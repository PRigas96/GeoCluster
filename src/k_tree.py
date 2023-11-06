import torch
import numpy as np
from src.models import Teacher, Student
from queue import Queue
from src.utils.functions import getUncertaintyArea, getE, NearestNeighbour
from src.metrics import Linf
import math as m
import matplotlib.pyplot as plt
from src.utils import plot_tools as pt


class Ktree:
    """Implements a k-tree class to use in Hierarchical Clustering"""

    def __init__(self, threshold, data, teacher_args, un_args, student_args):
        # self.boundary = boundary        #The given bounding box
        self.threshold = threshold      # Minimum number of data (objects) in a node.
        self.data = data                # The input data (objects).
        self.teacher_args = teacher_args
        self.un_args = un_args
        self.student_args = student_args
        self.divided = False
        root_parent = None
        self.root = self.Node(self.data, "0", self, root_parent)
        # self.root.create_student()   # Extract the trained student model for the root node.

    def create_tree(self, save_path_prefix="", plot=False):
        queue = Queue()
        queue.put(self.root)

        while not queue.empty():
            node = queue.get()
            # If a save path is set, append the node index to it.
            save_path_index_prefix = "" if save_path_prefix == "" else save_path_prefix + node.index

            # Create the student and divide the node if the data has size less than the defined threshold.
            if len(node.data) > self.threshold:
                node.create_student(save_path_index_prefix, plot)
                node.divide()
                for i in range(4):
                    queue.put(node.children[i])

    def query(self, query_point):
        query_point = query_point if torch.is_tensor(query_point) else torch.tensor(query_point)
        node = self.root
        while not node.isLeaf():
            pred = node.student(query_point)
            z = pred.argmax()
            node = node.children[z]

        return node.query(query_point)

    class Node:
        """Implements a node class to use in a tree."""

        def __init__(self, data, index, ktree, parent):
            """Initialise the class."""
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # never 
            self.device = "cpu"
            self.data = data
            self.student = None
            self.children = []
            self.ktree = ktree
            self.index = index
            self.best_z = torch.empty(0)
            self.parent = parent
            self.energies = []

        def create_student(self, save_path_prefix="", plot=False):
            # If the node has no data, do nothing.
            if len(self.data) == 0:
                return

            # First calculate the area of the data.
            size_sup = 2 * max(self.data[:, 2])
            x_lim = [m.floor(min(self.data[:, 0] - size_sup)), m.ceil(max(self.data[:, 0] + size_sup))]
            y_lim = [m.floor(min(self.data[:, 1] - size_sup)), m.ceil(max(self.data[:, 1] + size_sup))]

            """
            Teacher model.
            """
            
            # Create and train the teacher model.
            teacher = Teacher(4, 2, 400, len(self.index) - 1, self.parent).to(self.device)
            # Populate the optimizer with the model parameters and the learning rate.
            teacher_args = self.ktree.teacher_args.copy()
            teacher_args["optimizer"] = torch.optim.Adam(teacher.parameters(), lr=teacher_args["optimizer_lr"])
            del teacher_args["optimizer_lr"]
            # Train the teacher model and assign the best state found during training.
            teacher.train_(train_data=torch.from_numpy(self.data).float().to(self.device),
                           x_lim=x_lim, y_lim=y_lim, **teacher_args)
            teacher.load_state_dict(teacher.best_model_state)
            
            # Save the model and some training results.
            if save_path_prefix != "":
                # Save best model state to .pt format.
                torch.save(teacher.best_model_state, f"{save_path_prefix}_teacher_config.pt")
                print(f"Saved teacher config to {save_path_prefix}_teacher_config.pt")

                # Save training results to .npy format.
                teacher_results = {
                    "best_model_state": teacher.best_model_state,
                    "best_outputs": teacher.best_outputs,
                    "best_z": teacher.best_z,
                    "best_lat": teacher.best_lat,
                    "best_epoch": teacher.best_epoch,
                    "p_p": teacher.p_p,
                    "p_c": teacher.p_c,
                    "reg_proj_array": teacher.reg_proj_array,
                    "reg_latent_array": teacher.reg_latent_array,
                    "memory": teacher.memory,
                    "cost_array": teacher.cost_array
                }
                np.save(f"{save_path_prefix}_teacher_training_results.npy", teacher_results)
                print(f"Saved teacher training results to {save_path_prefix}_teacher_training_results.npy")
            
            # Plot training results.
            if plot:
                # Plot the Amplitude demodulation of the signal (costs array).
                signal = teacher.cost_array
                upper_signal, lower_signal, filtered_signal = pt.AM_dem(signal, fc=0.4 * len(signal),
                                                                        fs=2 * len(signal))
                pt.plot_AM_dem(upper_signal, lower_signal, filtered_signal, signal, teacher.best_epoch)
                # Plot the best model with the best outputs.
                manifold = pt.createManifold(teacher, teacher.best_outputs.cpu(), x_lim=x_lim, y_lim=y_lim)
                pt.plotManifold(self.data, manifold, teacher.best_outputs.cpu(), x_lim, y_lim)
                plt.show()

            """
            Uncertainty Area.
            """
            # Calculate the Uncertainty Area.
            m_points = getUncertaintyArea(outputs=teacher.best_outputs.cpu().detach().numpy(),
                                          x_area=x_lim, y_area=y_lim, model=None, **self.ktree.un_args)
            m_points = np.array(m_points)
            
            # Plot the Uncertainty Area.
            if plot:
                # Plot the m points that are in the uncertainty area.
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(m_points[:, 0], m_points[:, 1], s=20, c='royalblue',
                           alpha=0.5, marker='*', label='Uncertainty Area')
                ax.contourf(manifold[:, :, 0], manifold[:, :, 1], manifold[:, :, 2],
                            levels=200, cmap='viridis', alpha=0.2)
                plt.legend()
                plt.show()

            """
            Labeling.
            """
            qp = np.random.permutation(m_points)
            qp = torch.tensor(qp)
            F, z, F_sq, z_sq = getE(teacher, teacher.best_outputs.cpu(), qp, self.data)
            # Initialize the pseudo clusters.
            # Append data that their z_sq is i for each centroid.
            pseudo_clusters = [self.data[z_sq == i] for i in range(teacher.n_centroids)]
            # Create labels.
            outputs_shape = (qp.shape[0], teacher.n_centroids)
            F_ps = torch.zeros(outputs_shape)
            z_ps = torch.zeros(outputs_shape)
            for i in range(outputs_shape[0]):
                if i % 1000 == 0:
                    print(f"Labeled {i}/{outputs_shape[0]} points.")
                for j in range(outputs_shape[1]):
                    qpoint = qp[i].cpu().detach().numpy()
                    F_ps[i, j], z_ps[i, j] = torch.tensor(NearestNeighbour(qpoint, pseudo_clusters[j]))
            print(f"Labeled all {outputs_shape[0]}/{outputs_shape[0]} points.")
            
            # Plot the labels.
            if plot:
                # plot qp
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                plt_qp = qp.cpu().detach().numpy()
                new_labels = F_ps.min(1)[1].cpu().detach().numpy()
                ax.scatter(plt_qp[:, 0], plt_qp[:, 1], c=new_labels, s=50)
                # plot best_outputs
                plt_bo = teacher.best_outputs.cpu().detach().numpy()
                c = np.linspace(0, plt_bo.shape[0], plt_bo.shape[0])
                ax.scatter(plt_bo[:, 0], plt_bo[:, 1], c=c, s=200)
                pt.plot_data_on_manifold(fig, ax, self.data, size=10, limits=x_lim + y_lim)
                plt.show()

            """
            Student model.
            """
            # Create and train the student model and assign the best state found during training.
            student = Student(4, 2, 2).to(self.device)  # initialize the voronoi network
            student_args = self.ktree.student_args
            student.train_(optimizer=torch.optim.Adam(student.parameters(), lr=student_args["optimizer_lr"]),
                           epochs=student_args["epochs"],
                           device=self.device,
                           qp=qp,
                           F_ps=F_ps)
            student.load_state_dict(student.best_vor_model_state)
            
            # Save the model and some training results.
            if save_path_prefix != "":
                # Save voronoi model.
                torch.save(student.best_vor_model_state, f"{save_path_prefix}_student_config.pt")
                print(f"Saved student config to {save_path_prefix}_student_config.pt")

                # Save training results to .npy format.
                student_results = {
                    "best_vor_model_state": student.best_vor_model_state,
                    "cost_ll": student.cost_ll,
                    "acc_l": student.acc_l,
                    "es": student.es
                }
                np.save(f"{save_path_prefix}_student_training_results.npy", student_results)
                print(f"Saved teacher training results to {save_path_prefix}_student_training_results.npy")
            
            # Plot the student training results.
            if plot:
                student.plot_accuracy_and_loss(student_args["epochs"])
                plt.show()

            self.student = student
            self.best_z = teacher.best_z

        def create_student_from_config(self, path):
            student = Student(4, 2, 2).to(self.device)
            student.load_state_dict(torch.load(path))
            student.eval()
            self.student = student

        def isLeaf(self):
            return len(self.children) == 0

        # Create 4 child nodes, where every child stores one of the four clusters produced.
        def divide(self):
            # Retrieve the data that exist in each of the clusters.
            unique_clusters = torch.unique(self.best_z)
            for cluster in unique_clusters:
                cluster_data = self.data[self.best_z == cluster]
                # Create a child node with the corresponding data.
                self.children.append(Ktree.Node(cluster_data, f"{self.index}{cluster.item()}", self.ktree, self))

        def query(self, query_point):
            dists = np.array([Linf(self.data[i], query_point)[0] for i in range(len(self.data))])
            min_dist_index = dists.argmin()
            return self.data[min_dist_index]
