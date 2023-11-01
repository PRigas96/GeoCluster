import torch
import numpy as np
from src.models import LVGEBM, Voronoi
from src.utils.functions import getUncertaintyArea, getE, NearestNeighbour
import math as m
import matplotlib.pyplot as plt
from src.utils import plot_tools as pt


class QuadTree:
    """Implements a quadtree class to use in Hierarchical Clustering"""

    def __init__(self, threshold, data, teacher_args, un_args, student_args):
        # self.boundary = boundary        #The given bounding box
        self.threshold = threshold      # Minimum number of data (objects) in a node.
        self.data = data                # The input data (objects).
        self.teacher_args = teacher_args
        self.un_args = un_args
        self.student_args = student_args
        self.divided = False
        self.root = self.Node(self.data, 0, self)
        # self.root.create_student()   # Extract the trained student model for the root node.

    class Node:
        """Implements a node class to use in a tree."""

        def __init__(self, data, index, quadtree):
            """Initialise the class."""
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.data = data
            self.student = None
            self.children = []
            self.quadtree = quadtree
            self.index = index
            self.best_z = torch.empty(0)

        def create_student(self, plot=False):
            # First calculate the area of the data.
            x_lim = [m.floor(min(self.data[:, 0])), m.ceil(max(self.data[:, 0]))]
            y_lim = [m.floor(min(self.data[:, 1])), m.ceil(max(self.data[:, 1]))]

            # Create and train the teacher model.
            teacher = LVGEBM(4, 2, 400).to(self.device)
            # Populate the optimizer with the model parameters and the learning rate.
            teacher_args = self.quadtree.teacher_args
            self.quadtree.teacher_args["optimizer"] = torch.optim.Adam(teacher.parameters(),
                                                                       lr=teacher_args["optimizer_lr"])
            del teacher_args["optimizer_lr"]
            # Train the teacher model and assign the best state found during training.
            teacher.train_(**teacher_args)
            teacher.load_state_dict(teacher.best_model_state)
            # Plot training results.
            if plot:
                # Plot the Amplitude demodulation of the signal (costs array).
                signal = teacher.cost_array
                upper_signal, lower_signal, filtered_signal = pt.AM_dem(signal, fc=0.4 * len(signal),
                                                                        fs=2 * len(signal))
                pt.plot_AM_dem(upper_signal, lower_signal, filtered_signal, signal, teacher.best_epoch)
                # Plot the best model with the best outputs.
                manifold = pt.createManifold(teacher, teacher.best_outputs.cpu())
                pt.plotManifold(self.data, manifold, teacher.best_outputs.cpu(), x_lim, y_lim)
                plt.show()

            # Calculate the Uncertainty Area.
            m_points = getUncertaintyArea(outputs=teacher.best_outputs.cpu().detach().numpy(),
                                          x_area=x_lim, y_area=y_lim, model=None, **self.quadtree.un_args)
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

            # Labeling.
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

            # Create and train the student model and assign the best state found during training.
            student = Voronoi(4, 2, 2).to(self.device)  # initialize the voronoi network
            student_args = self.quadtree.student_args
            student.train_(optimizer=torch.optim.Adam(student.parameters(), lr=student_args["optimizer_lr"]),
                           epochs=student_args["epochs"],
                           device=self.device,
                           qp=qp,
                           F_ps=F_ps)
            student.load_state_dict(student.best_vor_model_state)
            # Plot the student training results.
            if plot:
                student.plot_accuracy_and_loss(student_args["epochs"])
                plt.show()

            self.student = student
            self.best_z = teacher.best_z

        def create_student_from_config(self, path):
            student = Voronoi(4, 2, 2).to(self.device)
            student.load_state_dict(torch.load(path))
            student.eval()
            self.student = student

        def isLeaf(self, data):
            return len(data) <= self.quadtree.threshold

        # Create 4 child nodes, where every child stores one of the four clusters produced.
        def divide(self):
            # Retrieve the data that exist in each of the clusters.
            unique_clusters = torch.unique(self.best_z)
            for cluster in unique_clusters:
                cluster_data = self.data[self.best_z == cluster]
                # Create a child node with the corresponding data.
                self.children.append(QuadTree.Node(cluster_data, cluster.item(), self.quadtree))
