import torch
import numpy as np
from src.models import Teacher, Student
from queue import Queue
from src.utils.functions import getUncertaintyArea, getE, NearestNeighbour
from src.ebmUtils import loss_functional
import math as m
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils import plot_tools as pt


class Ktree:
    """
        Implements a k-tree class to use in Hierarchical Clustering

        Parameters:
            threshold (int): The minimum number of data (objects) in a node.
            data (list): The input data (objects).
            teacher_args (dict): The arguments for the teacher model.
            un_args (dict): The arguments for the uncertainty area.
            student_args (dict): The arguments for the student model.

        Attributes:
            threshold (int): The minimum number of data (objects) in a node.
            data (list): The input data (objects).
            teacher_args (dict): The arguments for the teacher model.
            un_args (dict): The arguments for the uncertainty area.
            student_args (dict): The arguments for the student model.
            divided (bool): Whether the tree has been divided or not.
            root (Node): The root node of the tree.
            
        Methods:
            create_tree (save_path_prefix="", plot=False): Creates the tree.
            query (query_point): Returns the nearest neighbour of the query point.
            Node:
                __init__ (data, index, ktree, parent): Initialises the node class.
                create_student (save_path_prefix="", plot=False): Creates the student model.
                isLeaf (): Returns whether the node is a leaf or not.
                divide (): Divides the node.
                query (query_point): Returns the nearest neighbour of the query point.
                
    """

    def __init__(self, threshold,
                 data,
                 metric,
                 teacher_args,
                 un_args,
                 student_args,
                 device,
                 dim=2):
        # self.boundary = boundary        #The given bounding box
        self.threshold = threshold      # Minimum number of data (objects) in a node.
        self.data = data                # The input data (objects).
        self.metric = metric
        self.teacher_args = teacher_args
        self.un_args = un_args
        self.student_args = student_args
        self.divided = False
        root_parent = None
        self.root = self.Node(self.data, "0", self, root_parent, device=device)
        self.dim = dim
        self.device = device
        self.number_of_nodes = 1
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
                print()
                print("="*20)
                print(f"Creating student for node {node.index} that has {len(node.data)} data, which is more than the threshold {self.threshold}.")
                node.create_student(save_path_index_prefix, plot)
                if node.student is not None:
                    node.divide()
                    self.number_of_nodes += node.student.n_centroids
                    for i in range(node.student.n_centroids):
                        queue.put(node.children[i])

    def create_tree_from_config(self, path_prefix):
        queue = Queue()
        queue.put(self.root)

        while not queue.empty():
            node = queue.get()
            path = Path(path_prefix + node.index + "_student_config.pt")

            # If there is no student config, the node is a leaf so do nothing.
            if not path.is_file():
                continue

            node.create_student_from_config(str(path))

            # Calculate best_z indices in order to divide the node data afterward.
            # See relevant comments in "Node.create_student" post teacher training.
            centroids = np.load(path_prefix + node.index + "_teacher_training_results.npy",
                                allow_pickle=True).item()["best_outputs"]
            e = loss_functional(centroids, torch.from_numpy(node.data).float().to(self.device), self.metric)
            _, node.best_z = e.min(1)
            best_z_unique, node.best_z = torch.unique(node.best_z, return_inverse=True)

            # Divide the node and add its children to the queue.
            node.divide()
            self.number_of_nodes += node.student.n_centroids
            for child in node.children:
                queue.put(child)

    def get_leaves(self, node=None):
        """Returns a list of all the tree's leaf nodes (ordered "left to right").

        Returns:
            list[Node]: A list of all the leaf nodes, ordered "left to right".
        """
        node = node if node is not None else self.root

        if node.isLeaf():
            return [node] # Return a list with the leaf node itself as its only element.

        leaves = []
        for i in range(len(node.children)):
            leaves += self.get_leaves(node.children[i])

        return leaves

    def query(self, query_point):
        """A query in the k-tree structure for a given point.

        Args:
            query_point (torch.tensor): A point vector (in the objects' dimension).
        Returns:
            object: The nearest neighbor data object with respect to the query point.
        """
        return self.query_verbose(query_point)["nn"]

    def query_verbose(self, query_point):
        """A verbose query in the k-tree structure for a given point.
            Use this method to add more query results.

        Args:
            query_point (torch.tensor): A point vector (in the objects' dimension).
        Returns:
            dict: A dictionary with properties
            - nn (object): The nearest neighbor data object with respect to the query point.
            - cluster_index (str): The index property of the leaf node the nearest neighbour belongs to.
        """
        predictions_per_layer = []
        query_point = query_point if torch.is_tensor(query_point) else torch.tensor(query_point)
        # to device
        query_point = query_point.to(self.device)
        #print(f"Querying point {query_point}...")
        #print(self.device)

        node = self.root
        
        while not node.isLeaf():
            pred = node.student(query_point)
            #print(f"Predictions for node {node.index} are {pred}")
            z = pred.argmax()
            node = node.children[z]
            predictions_per_layer.append(node.query(query_point))

        #print(f"Query point {query_point} belongs to node {node.index} ")
        predicted_nn = node.query(query_point)
        predictions_per_layer.append(predicted_nn)
        return {
            "nn": predicted_nn,
            "cluster_index": node.index,
            "predictions per layer": predictions_per_layer
        }

    def query_maxsum(self, query_points):
        leaves = self.get_leaves()
        leaf_sums = self.root.get_leaf_sums(query_points)
        #print("Leaf_sums are:",leaf_sums)
        _, maxsum_indices = leaf_sums.max(1)
        #print("Maxsum_indices are:",maxsum_indices)
        #print(f"Nn is: {leaves[maxsum_indices[0]].query(query_points[0])}")
        return [leaves[maxsum_indices[i]].query(query_points[i]) for i in range(len(query_points))]
    
    def querry_knn_per_layer(self, query_points, k, eps = 0):
        """
            #TODO: implement this
            Get k best choices per layer, or eps
            
            Args:
                query_points (torch.tensor): A point vector (in the objects' dimension).
                k (int): The number of best choices per layer.
                eps (int): Sensitivity parameter.
        """
        if eps == 0:
            pass
        else:
            # at each layer, get the choices and pick the ones inside eps ball
            layer = self.root
            layers = []
            while not layer.isLeaf():
                pred = layer.student(query_points)
                _, z_ordered = pred.topk(k) # Get the indices of prediction
                # check if 2 predictions are inside ball
                diff = pred - max(pred)
                for i in range(len(diff)):
                    if diff[i] < eps:
                        #TODO: if 2 predictions are inside the ball, search them both
                        z_ordered = [z_ordered[0], z_ordered[i]]
                else:
                    # pick only the first
                    z_ordered = z_ordered[0]
                # If the child's data are empty, continue to the next z.
                # Otherwise, set that child as current node and break (the for loop).
                for z in z_ordered:
                    if len(layer.children[z].data) == 0:
                        continue
                    layer = layer.children[z]
                    break
                layers.append(layer)
            # return the best choice

    def query_maxcumsum(self, query_points):
        leaves = self.get_leaves()
        leaves_sum = self.root.get_leaf_cumsums(query_points)
        #print("Leaf_sums are(cumsum):",leaves_sum)
        _, maxcumsum_indices = leaves_sum.max(1)
        #print("Maxcumsum_indices are:",maxcumsum_indices)
        #print(f"Nn is: {leaves[maxcumsum_indices[0]].query(query_points[0])}")
        return [leaves[maxcumsum_indices[i]].query(query_points[i]) for i in range(len(query_points))]

    def plot_leaf_clusters(self, n_samples=2000):
        """Plot the cluster spaces defined by the tree's leaf nodes.
            Take as samples an equal split of the space in each dimension (final number of samples will be reduced),
            predict their cluster and plot them colored accordingly by converting node index to integer.

        Args:
            n_samples (int): The number of the points to sample.
        """
        n_samples_dim = int(n_samples ** (1 / self.dim))
        bounding_box = self.root.get_bounding_box()
        samples_linspace = np.array([np.linspace(bounding_box[d][0], bounding_box[d][1], n_samples_dim)
                                     for d in range(self.dim)])
        samples = np.array(np.meshgrid(*samples_linspace)).T.reshape(-1, self.dim)

        # Predict the sample point, get the cluster node index and convert it to an integer using as base
        # the number of centroids k. Add "1" at the beginning to distinguish e.g. "01" from "001".
        samples_cluster_indices = [self.query_verbose(torch.from_numpy(sample).float())["cluster_index"]
                                   for sample in samples]
        sample_clusters_ids = [int("1" + samples_cluster, self.teacher_args["number_of_centroids"])
                               for samples_cluster in samples_cluster_indices]
        ax = plt.axes(projection='3d' if self.dim == 3 else None)
        ax.scatter(*tuple(samples[:, i] for i in range(samples.shape[1])), c=sample_clusters_ids)

    def plot_leaf_clusters_voronoi(self, n_samples=2000):
        """Plot the voronoi diagram (by sampling) shared among the data
            in the cluster spaces defined by the tree's leaf nodes.
            Take as samples an equal split of the space in each dimension (final number of samples will be reduced),
            find their nearest neighbour (by brute force) and the cluster they're in
            and plot them colored accordingly by converting node index to integer.

        Args:
            n_samples (int): The number of the points to sample.
        """
        n_samples_dim = int(n_samples ** (1 / self.dim))
        bounding_box = self.root.get_bounding_box()
        samples_linspace = np.array([np.linspace(bounding_box[d][0], bounding_box[d][1], n_samples_dim)
                                     for d in range(self.dim)])
        samples = np.array(np.meshgrid(*samples_linspace)).T.reshape(-1, self.dim)

        samples_nn_z = [NearestNeighbour(sample, self.data, self.metric)[1] for sample in samples]
        samples_cluster_indices = ["0"] * len(samples)
        for i, nn_z in enumerate(samples_nn_z):
            node = self.root
            while not node.isLeaf():
                child_has_nn = [self.data[nn_z].tolist() in node.children[j].data.tolist()
                                for j in range(len(node.children))]
                node = node.children[child_has_nn.index(True)]
            samples_cluster_indices[i] = node.index

        # Get the cluster node index and convert it to an integer using as base the number of centroids k.
        # Add "1" at the beginning to distinguish e.g. "01" from "001".
        sample_clusters_ids = [int("1" + samples_cluster, self.teacher_args["number_of_centroids"])
                               for samples_cluster in samples_cluster_indices]
        ax = plt.axes(projection='3d' if self.dim == 3 else None)
        ax.scatter(*tuple(samples[:, i] for i in range(samples.shape[1])), c=sample_clusters_ids)

    def get_student_accuracies(self, query_points):
        queue = Queue()
        queue.put(self.root)
        correct_predictions_per_student = {}
        student_accuracies = {}

        while not queue.empty():
            node = queue.get()
            while not node.isLeaf():
                for query_point in query_points:
                    predicted_z = node.student(query_point)
                    z = predicted_z.argmax()
                    predicted_nn = node.children[z].query(query_point)
                    exact_nn = self.root.query(query_point)
                    if(np.equal(predicted_nn, exact_nn)):
                        correct_predictions_per_student[node.index] +=1
                student_accuracies[node.index] = correct_predictions_per_student[node.index]/len(query_points)

        return student_accuracies

    class Node:
        """Implements a node class to use in a tree."""

        def __init__(self, data, index, ktree, parent,device):
            """Initialise the class."""
            #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # never 
            self.device = device
            self.data = data
            self.student = None
            self.children = []
            self.ktree = ktree
            self.index = index
            self.best_z = torch.empty(0)
            self.best_reg = 0
            self.parent = parent
            self.energies = []
            self.un_points = None
            self.un_labels = None
            self.un_energy = None

        def get_bounding_box(self):
            """Calculate and return the bounding box of the data.

            Returns:
                list: A list of [min, max] pairs for each dimension of the data.
            """
            # The first "dim" columns have the centers, the second "dim" columns have the sizes.
            size_sup = 2 * np.max(self.data[:, self.ktree.dim:2 * self.ktree.dim])
            return [[m.floor(min(self.data[:, i] - size_sup)), m.ceil(max(self.data[:, i] + size_sup))]
                    for i in range(self.ktree.dim)]

        def create_student(self, save_path_prefix="", plot=False):
            # If the node has no data, do nothing.
            if len(self.data) == 0:
                return

            # First calculate the bounding box of the data.
            bounding_box = self.get_bounding_box()
            print(f"Bounding box for node {self.index}: {bounding_box}")

            """
            Teacher model.
            """
            
            # Create and train the teacher model.
            n_of_centroids = self.ktree.teacher_args["number_of_centroids"]
            print(f"Creating teacher for node {self.index} with {n_of_centroids} centroids.")
            if n_of_centroids == 0:
                n_of_centroids = 2**self.ktree.dim

            encoder_activation = self.ktree.teacher_args["encoder_activation"]
            encoder_depth = self.ktree.teacher_args["encoder_depth"]
            predictor_width = self.ktree.teacher_args["predictor_width"]
            predictor_depth = self.ktree.teacher_args["predictor_depth"]
            latent_size = self.ktree.teacher_args["latent_size"]
            latent_size = len(self.data) // 2
            teacher = Teacher(n_of_centroids,
                              self.ktree.dim,
                              encoder_activation,
                              encoder_depth,
                              predictor_width,
                              predictor_depth,
                              latent_size, 
                              self.index, 
                              self.parent, 
                              self.ktree.dim).to(self.device)
            # Populate the optimizer with the model parameters and the learning rate.
            teacher_args = self.ktree.teacher_args.copy()
            teacher_args["optimizer"] = torch.optim.Adam(teacher.parameters(), lr=teacher_args["optimizer_lr"])
            del teacher_args["optimizer_lr"]
            # Train the teacher model and assign the best state found during training.
            teacher.train_(train_data=torch.from_numpy(self.data).float().to(self.device),
                           metric=self.ktree.metric, bounding_box=bounding_box, **teacher_args)
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
                manifold = pt.createManifold(teacher, teacher.best_outputs.cpu(), self.ktree.metric,
                                             x_lim=bounding_box[0], y_lim=bounding_box[1])
                manifold = manifold.cpu().detach().numpy()
                pt.plotManifold(self.data, manifold, teacher.best_outputs.cpu(), bounding_box[0], bounding_box[1])
                plt.show()

            # Recalculate best_z indices since teacher.best_z is fuzzy from training.
            e = loss_functional(teacher.best_outputs, torch.from_numpy(self.data).float().to(self.device),
                                self.ktree.metric)
            _, self.best_z = e.min(1)
            # Keep only the centroids that appear in the data predictions.
            # Also update the best_z indices to index only the kept centroids
            # e.g. convert (1, 0, 3) to (1, 0, 2); since centroid 2 is missing we count index 3 as 2.
            # Do so using the inverse indexing tensor of torch.unique().
            best_z_unique, self.best_z = torch.unique(self.best_z, return_inverse=True)
            centroids = teacher.best_outputs[best_z_unique]
            n_of_centroids = len(centroids)
            # Finally store the regularised projection from the best epoch.
            self.best_reg = teacher.reg_proj_array[teacher.best_epoch]

            # If the division is fully unbalanced, i.e. all data is put into one child, no need for a student.
            if n_of_centroids == 1:
                return

            """
            Uncertainty Area.
            """
            # Calculate the Uncertainty Area.
            m_points = getUncertaintyArea(outputs=centroids.cpu().detach().numpy(),
                                          bounding_box=bounding_box, **self.ktree.un_args)
            m_points = np.array(m_points)
            self.un_points = m_points
            
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
            # Initialize the pseudo clusters.
            # Append data that their best_z is i for each centroid.
            pseudo_clusters = [self.data[self.best_z == i] for i in range(n_of_centroids)]
            # Create labels.
            outputs_shape = (qp.shape[0], n_of_centroids)
            F_ps = torch.zeros(outputs_shape)
            z_ps = torch.zeros(outputs_shape)
            for i in range(outputs_shape[0]):
                if i % 1000 == 0:
                    print(f"Labeled {i}/{outputs_shape[0]} points.")
                for j in range(outputs_shape[1]):
                    qpoint = qp[i].cpu().detach().numpy()
                    F_ps[i, j], z_ps[i, j] = torch.tensor(NearestNeighbour(qpoint, pseudo_clusters[j], self.ktree.metric))
            print(f"Labeled all {outputs_shape[0]}/{outputs_shape[0]} points.")
            self.un_labels = z_ps
            self.un_energy = F_ps
            
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
                pt.plot_data_on_manifold(fig, ax, self.data, size=10, limits=bounding_box[0] + bounding_box[1])
                plt.show()

            """
            Student model.
            """
            # Create and train the student model and assign the best state found during training.
            
            student_args = self.ktree.student_args
            width = student_args["width"]
            depth = student_args["depth"]
            print(f"Creating student for node {self.index} with {n_of_centroids} centroids.")
            print("Device is:", self.device)
            student = Student(n_of_centroids, self.ktree.dim, self.ktree.dim, width, depth).to(self.device)  # initialize the voronoi network
            student.train_(optimizer=torch.optim.Adam(student.parameters(), lr=student_args["optimizer_lr"]),
                           epochs=student_args["epochs"],
                           device=self.device,
                           qp=qp,
                           F_ps=F_ps
                           )
            student.eval()
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

            # Store the trained student.
            self.student = student

        def create_student_from_config(self, path):
            width = self.ktree.student_args["width"]
            depth = self.ktree.student_args["depth"]

            # Get the model parameters from the state dict.
            state_dict = torch.load(path)
            # Last predictor layer has bias shape (n_centroids,).
            n_centroids = state_dict[f"predictor.{2 * depth - 2}.bias"].shape[0]

            # Build the student object and assign it to the node.
            student = Student(n_centroids, self.ktree.dim, self.ktree.dim, width, depth).to(self.device)  # initialize the voronoi network
            student.load_state_dict(torch.load(path))
            student.eval()
            self.student = student

        def isLeaf(self):
            return len(self.children) == 0

        def divide(self):
            """Retrieve the data that exist in each of the clusters.
                Create child nodes where every child stores one cluster, each containing
                the data that the teacher predicted each centroid is closest (best_z).
            """
            for cluster in range(self.student.n_centroids):
                cluster_data = self.data[self.best_z == cluster]
                # Create a child node with the corresponding data.
                #         def __init__(self, data, index, ktree, parent,device):
                self.children.append(Ktree.Node(cluster_data, f"{self.index}{cluster}", self.ktree, self, self.device))

        def query(self, query_point, k=1):
            # if it doesnt work try this:
            # dists = np.array([self.ktree.metric(torch.from_numpy(self.data[i]).double(), query_point) for i in range(len(self.data))])
            query_point.to(self.device)
            query_point = torch.tensor(query_point) if not torch.is_tensor(query_point) else query_point
            #print("Query device is:", query_point.device)
            #print("Data device are: ")
            #print(torch.from_numpy(self.data[0]).double().to(self.device))
            #print("Data[0] device are: ", self.data[0].device)
            #dists = np.array([self.ktree.metric(torch.from_numpy(self.data[i]).double().to(self.device), query_point) for i in range(len(self.data))])
            #dists = np.array([self.ktree.metric(torch.from_numpy(self.data[i]).double().to(self.device), query_point) for i in range(len(self.data))])
            dists = torch.tensor([self.ktree.metric(torch.from_numpy(self.data[i]).double().to(self.device), query_point) for i in range(len(self.data))])
            # dists should be tensor


            # min_dist_index = dists.argmin() #returns the exact nearest neighbor

            # _, min_dist_indices = torch.topk(dists, k, largest=False) #we want to return the k nearest for now
            # min_dist_indices = min_dist_indices.sort().indices
            k_smallest_indices = np.argsort(dists)[:k]


            # return self.data[min_dist_index]

            k_nearest = [self.data[i] for i in k_smallest_indices]

            return k_nearest

        def get_leaf_sums(self, query_points):
            """Recursive function to calculate the energy from the node's student predictions
                of given query points and add it to each of its children's respective energy.

            Args:
                query_points (torch.Tensor): The query points.

            Returns:
                torch.Tensor: A tensor containing the totally summed energies of the query point predictions
                    for each leaf node (equivalently for each path on the tree).
            """
            # Base case, for a leaf return 0s as its energies are calculated on the parent.
            if self.isLeaf():
                return torch.zeros((len(query_points), 1))

            # The student predictions are the children energies.
            y_pred_children = self.student(query_points)
            # Each y_pred_children column corresponds to the energies of a child node, so add that column
            # to each child's leaf sums, i.e. the tensor with their leaves' predictions.
            sums = [y_pred_children[:, i].reshape(-1, 1) + self.children[i].get_leaf_sums(query_points)
                    for i in range(len(self.children))]
            return torch.hstack(tuple(sums))

        def get_leaf_cumsums(self, query_points, y_pred=None):
            """Recursive function to calculate the cumulative energy from the node's student predictions
                of given query points and add it to each of its children's respective energy.

            Args:
                query_points (torch.Tensor): The query points.
                y_pred (torch.Tensor): The prediction energy column (from the parent) for the current node.

            Returns:
                torch.Tensor: A tensor containing the totally cumulatively summed energies of the query point
                    predictions for each leaf node (equivalently for each path on the tree).
            """
            # Base case, for a leaf return 0s as its energies are calculated on the parent.
            if self.isLeaf():
                return torch.zeros((len(query_points), 1))

            # y_pred validation.
            y_pred = y_pred if y_pred is not None else torch.zeros((len(query_points), 1))
            y_pred = y_pred if torch.is_tensor(y_pred) else torch.tensor(y_pred)
            y_pred.reshape(-1, 1)  # Convert to a column tensor.
            y_pred.to(self.device)

            # The children energies for each query point is the sum of
            # the node energies (y_pred) and the student predictions.
            y_pred_children = y_pred + self.student(query_points)
            # Each
            cumsums = [y_pred_children[:, i].reshape(-1, 1) +
                       self.children[i].get_leaf_cumsums(query_points, y_pred_children[:, i])
                       for i in range(len(self.children))]
            return torch.hstack(tuple(cumsums))