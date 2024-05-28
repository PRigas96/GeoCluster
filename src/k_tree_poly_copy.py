import torch
import numpy as np
from src.models import Clustering, ClusteringLS,ClusteringPLS, Critic, ClusteringNew
from queue import Queue
from src.utils.functions import NearestNeighbour
from src.utils.embeddings import loss_functional, getUncertaintyArea
import math as m
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils import plot_tools as pt


class Ktree:
    """
        Implements a k-tree class to use in Hierarchical Clustering

        Attributes:
            threshold (int): the minimum number of data (objects) in a node
            data (np.array): the input data (objects)
            metric (callable): the metric function to use to compute the distance between the data and given points
            clustering_args (dict): the arguments for the clustering model
            un_args (dict): the arguments for the uncertainty area sampler
            critic_args (dict): the arguments for the critic model
            divided (bool): True is the tree has been divided, False otherwise
            root (Node): The root node of the tree.
            dim (int): the dimensionality of the data
            device (torch.device): the currently selected device for torch
            number_of_nodes (int): the number of nodes in the k-tree

        Methods:
            create_tree (save_path_prefix="", plot=False): Creates the tree.
            create_tree_from_config (path_prefix): Creates the tree from a set of pre-saved models.
            get_leaves: Returns a list of all the tree's leaf nodes (ordered "left to right").
            query (query_point): Runs a query in the k-tree structure for a given point.
            query_verbose (query_point): Runs a verbose query in the k-tree structure for a given point.
            query_maxsum (query_points): Runs a query in the k-tree structure for a given list of points
                using the max sum criterion.
            query_knn_per_layer (query_points, k, eps): Gets the best k choices per layer, or eps.
            query_maxcumsum (query_points): Runs a query in the k-tree structure for a given list of points
                using the max cumulative sum criterion.
            plot_leaf_clusters (n_samples): Plots the cluster spaces defined by the tree's leaf nodes.
            plot_leaf_clusters_voronoi (n_samples): Plots the voronoi diagram (by sampling) shared among the data
                in the cluster spaces defined by the tree's leaf nodes.
            get_critic_accuracies (query_points): Calculates the prediction accuracy for each (non-leaf) node critic.
    """

    def __init__(self, threshold,
                 data,
                 metric,
                 clustering_args,
                 un_args,
                 critic_args,
                 device,
                 dim=2):
        """
            Initialises a Ktree object.
    
            Parameters:
                threshold (int): the minimum number of data (objects) in a node
                data (np.array): the input data (objects)
                metric (callable): the metric function to use to compute the distance between the data and given points
                clustering_args (dict): the arguments for the clustering model
                un_args (dict): the arguments for the uncertainty area sampler
                critic_args (dict): the arguments for the critic model
                device (torch.device): the currently selected device for torch
                dim (int): the dimensionality of the data
        """
        self.threshold = threshold  # Minimum number of data (objects) in a node.
        self.data = data  # The input data (objects).
        self.metric = metric
        self.clustering_args = clustering_args
        self.un_args = un_args
        self.critic_args = critic_args
        self.divided = False
        root_parent = None
        self.root = self.Node(self.data, "0", self, root_parent, device=device,metric=self.metric)
        self.dim = dim
        self.device = device
        self.number_of_nodes = 1

    def create_tree(self, save_path_prefix="", plot=False):
        """
            Creates the k-tree. Starting from root node it creates a critic model and
                iteratively repeats the process to its children until the stop criterion is reached
                i.e. the data size is less than the defined threshold.

            See Also:
                Node.create_critic: Creates the critic model.

            Parameters:
                save_path_prefix (str): if set, the trained models and training parameters
                    will be saved in the path specified here, appended by each node's index
                plot (bool): if set, plots about the trained models will be shown on runtime
        """
        queue = Queue()
        queue.put(self.root)

        while not queue.empty():
            node = queue.get()
            # If a save path is set, append the node index to it.
            save_path_index_prefix = "" if save_path_prefix == "" else save_path_prefix + node.index

            # Create the critic and divide the node if the data has size less than the defined threshold.
            if len(node.data) > self.threshold:
                print()
                print("=" * 20)
                print(
                    f"Creating critic for node {node.index} that has {len(node.data)} data, which is more than the threshold {self.threshold}.")
                node.create_critic(save_path_index_prefix, plot)
                if node.critic is not None:
                    node.divide()
                    self.number_of_nodes += node.critic.n_centroids
                    for i in range(node.critic.n_centroids):
                        queue.put(node.children[i])

    def create_tree_from_config(self, path_prefix):
        """
            Creates the tree from a set of model configurations saved in files.
                The model configurations must exist in the same directory and must be named
                as 'path_prefix' + 'node.index' + '_critic_config.pt'.

            See Also:
                Node.create_critic_from_config: Creates a critic model from a model config saved in a file.

            Parameters:
                path_prefix (str): the path in which to look for the model config files
        """
        queue = Queue()
        queue.put(self.root)

        while not queue.empty():
            node = queue.get()
            path = Path(path_prefix + node.index + "_critic_config.pt")

            # If there is no critic config, the node is a leaf so do nothing.
            if not path.is_file():
                continue

            node.create_critic_from_config(str(path))

            # Calculate best_z indices in order to divide the node data afterward.
            # See relevant comments in "Node.create_critic" post clustering training.
            centroids = np.load(path_prefix + node.index + "_clustering_training_results.npy",
                                allow_pickle=True).item()["best_centroids"]
            e = loss_functional(centroids, torch.from_numpy(node.data).float().to(self.device), self.metric)
            _, node.best_z = e.min(1)
            best_z_unique, node.best_z = torch.unique(node.best_z, return_inverse=True)

            # Divide the node and add its children to the queue.
            node.divide()
            self.number_of_nodes += node.critic.n_centroids
            for child in node.children:
                queue.put(child)

    def get_leaves(self, node=None):
        """
            Returns a list of all the tree's leaf nodes (ordered "left to right").

            Returns:
                list[Node]: A list of all the leaf nodes, ordered "left to right".
        """
        node = node if node is not None else self.root

        if node.isLeaf():
            return [node]  # Return a list with the leaf node itself as its only element.

        leaves = []
        for i in range(len(node.children)):
            leaves += self.get_leaves(node.children[i])

        return leaves

    def query(self, query_point):
        """
            Runs a query in the k-tree structure for a given point.

            Parameters:
                query_point (torch.Tensor): a point vector (in the objects' dimension)

            Returns:
                object: the nearest neighbor data object with respect to the query point
        """
        return self.query_verbose(query_point)["nn"]

    def query_verbose(self, query_point):
        """
            Runs a verbose query in the k-tree structure for a given point.
                Use this method to add more query results.

            Parameters:
                query_point (torch.Tensor): a point vector (in the objects' dimension)

            Returns:
                dict: A dictionary with properties
                - nn (object): the nearest neighbor data object with respect to the query point
                - cluster_index (str): the index property of the leaf node the nearest neighbour belongs to
                - predictions per layer (list): a list with a prediction for each node visited from the query
        """
        predictions_per_layer = []
        query_point = query_point if torch.is_tensor(query_point) else torch.tensor(query_point)
        # to device
        query_point = query_point.to(self.device)
        # print(f"Querying point {query_point}...")
        # print(self.device)

        node = self.root

        while not node.isLeaf():
            pred = node.critic(query_point)
            z = pred.argmax()
            node = node.children[z]
            predictions_per_layer.append(node.query(query_point))

        return {
            "nn": predictions_per_layer[-1],
            "cluster_index": node.index,
            "predictions per layer": predictions_per_layer
        }

    def query_maxsum(self, query_points):
        """
            Runs a query in the k-tree structure for a given list of points using the max sum criterion.
                In each layer the summed prediction (i.e. the sum of its own prediction
                and all its parents' predictions) is calculated for each node in the layer
                and the one selected is the leaf with the highest summed prediction.

            Parameters:
                query_points (torch.Tensor): a list of point vectors (in the objects' dimension)

            Returns:
                list[object]: a list of the nearest neighbor data object for each query point
        """
        leaves = self.get_leaves()
        leaf_sums = self.root.get_leaf_sums(query_points)
        # print("Leaf_sums are:",leaf_sums)
        _, maxsum_indices = leaf_sums.max(1)
        # print("Maxsum_indices are:",maxsum_indices)
        # print(f"Nn is: {leaves[maxsum_indices[0]].query(query_points[0])}")
        return [leaves[maxsum_indices[i]].query(query_points[i]) for i in range(len(query_points))]

    def query_knn_per_layer(self, query_points, k, eps=0):
        """
            Gets the best k choices per layer, or eps.
            
            Parameters:
                query_points (torch.Tensor): a point vector (in the objects' dimension).
                k (int): the number of best choices per layer to select
                eps (int): sensitivity parameter
        """
        if eps == 0:
            pass
        else:
            # at each layer, get the choices and pick the ones inside eps ball
            layer = self.root
            layers = []
            while not layer.isLeaf():
                pred = layer.critic(query_points)
                _, z_ordered = pred.topk(k)  # Get the indices of prediction
                # check if 2 predictions are inside ball
                diff = pred - max(pred)
                for i in range(len(diff)):
                    if diff[i] < eps:
                        # TODO: if 2 predictions are inside the ball, search them both
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
        """
            Runs a query in the k-tree structure for a given list of points using the max cumulative sum criterion.
                In each layer the cumulative sum prediction (i.e. the sum of its own prediction
                and all its parents' cumulative predictions) is calculated for each node in the layer
                and the one selected is the leaf with the highest cumulative summed prediction.

            Parameters:
                query_points (torch.Tensor): a list of point vectors (in the objects' dimension)

            Returns:
                list[object]: a list of the nearest neighbor data object for each query point
        """
        leaves = self.get_leaves()
        leaves_sum = self.root.get_leaf_cumsums(query_points)
        # print("Leaf_sums are(cumsum):",leaves_sum)
        _, maxcumsum_indices = leaves_sum.max(1)
        # print("Maxcumsum_indices are:",maxcumsum_indices)
        # print(f"Nn is: {leaves[maxcumsum_indices[0]].query(query_points[0])}")
        return [leaves[maxcumsum_indices[i]].query(query_points[i]) for i in range(len(query_points))]

    def plot_leaf_clusters(self, n_samples=2000):
        """
            Plots the cluster spaces defined by the tree's leaf nodes.
                Take as samples an equal split of the space in each dimension (final number of samples will be reduced),
                predict their cluster and plot them colored accordingly by converting node index to integer.

            Parameters:
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
        sample_clusters_ids = [int("1" + samples_cluster, self.clustering_args["number_of_centroids"])
                               for samples_cluster in samples_cluster_indices]
        ax = plt.axes(projection='3d' if self.dim == 3 else None)
        ax.scatter(*tuple(samples[:, i] for i in range(samples.shape[1])), c=sample_clusters_ids)

    def plot_leaf_clusters_voronoi(self, n_samples=2000):
        """
            Plots the voronoi diagram (by sampling) shared among the data
                in the cluster spaces defined by the tree's leaf nodes.
                Take as samples an equal split of the space in each dimension (final number of samples will be reduced),
                find their nearest neighbour (by brute force) and the cluster they're in
                and plot them colored accordingly by converting node index to integer.

            Parameters:
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
        sample_clusters_ids = [int("1" + samples_cluster, self.clustering_args["number_of_centroids"])
                               for samples_cluster in samples_cluster_indices]
        ax = plt.axes(projection='3d' if self.dim == 3 else None)
        ax.scatter(*tuple(samples[:, i] for i in range(samples.shape[1])), c=sample_clusters_ids)

    def get_critic_accuracies(self, query_points):
        """
            Calculates the prediction accuracy for each (non-leaf) node critic.

            Parameters:
                query_points (torch.Tensor): a list of point vectors (in the objects' dimension)

            Returns:
                dict: a dictionary with a key for each non-leaf node index,
                    each valued with the ratio of successful query predictions
        """
        queue = Queue()
        queue.put(self.root)
        correct_predictions_per_critic = {}
        critic_accuracies = {}

        while not queue.empty():
            node = queue.get()
            if not node.isLeaf():
                for query_point in query_points:
                    predicted_z = node.critic(query_point)
                    z = predicted_z.argmax()
                    predicted_nn = node.children[z].query(query_point)[0]
                    exact_nn = node.query(query_point)[0]
                    # if np.array_equal(predicted_nn, exact_nn):
                    # print(np.equal(predicted_nn, exact_nn))
                    # print(torch.equal(predicted_nn, exact_nn))
                    if torch.equal(predicted_nn, exact_nn):
                        if node.index not in correct_predictions_per_critic:
                            correct_predictions_per_critic[node.index] = 0
                        else:
                            correct_predictions_per_critic[node.index] += 1
                print(correct_predictions_per_critic)
                critic_accuracies[node.index] = correct_predictions_per_critic[node.index] / len(query_points)

                for child in node.children:
                    queue.put(child)

        return critic_accuracies

    class Node:
        """
            Implements a node class to use in a tree.
            
            Attributes:
                device (torch.device): the currently selected device for torch
                data (np.array): the data (objects) belonging to this node
                critic (Critic|None): the critic model
                children (list[Ktree.Node]): a list of the children Node objects
                ktree (Ktree): the k-tree to which the node belongs
                index (str): the index of the node, a string of integers describing the child path from root
                best_z (torch.Tensor): a list of equal length as the data containing the index of the closest centroid
                best_reg (int): the regularised projection of the centroids
                parent (Ktree.Node): the parent node of this node, None for the k-tree root node
                un_points (np.array|None): the points returned by the uncertainty area sampler
                un_labels (torch.Tensor|None): the labels of the uncertainty area points
                un_energy (torch.Tensor|None): the predicted energies of the uncertainty area points

            Methods:
                get_bounding_box (): Calculates and returns the bounding box of the data.
                create_critic (save_path_prefix="", plot=False): Creates the critic model.
                create_critic_from_config (path): Creates the node critic from file configuration.
                isLeaf (): Returns whether the node is a leaf or not.
                divide (): Creates child nodes where every child stores the data that each centroid is closest (best_z).
                query (query_point): Returns (by brute force) the k nearest neighbours of the query point.
                get_leaf_sums (query_points): Recursive function to calculate the energy from the node's critic
                    predictions of given query points and add it to each of its children's respective energy.
                get_leaf_cumsums (query_points): Recursive function to calculate the cumulative energy from the node's
                    critic predictions of given query points and add it to each of its children's respective energy.
        """

        def __init__(self, data, index, ktree, parent, device,metric):
            """
                Initialises a node object.

                Parameters:
                    data (np.array): the data (objects) belonging to this node
                    index (str): the index of the node, a string of integers describing the child path from root
                    ktree (Ktree): the k-tree to which the node belongs
                    parent (Ktree.Node): the parent node of this node, None for the k-tree root node
                    device (torch.device): the currently selected device for torch
            """
            # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # never
            self.device = device
            self.data = data
            self.critic = None
            self.children = []
            self.ktree = ktree
            self.index = index
            self.best_z = torch.empty(0)
            self.best_reg = 0
            self.parent = parent
            self.un_points = None
            self.un_labels = None
            self.un_energy = None
            self.metric = metric
            if self.metric is None:
                raise ValueError("Metric is not defined.")
            # pass data to device
            if type(self.data) is torch.Tensor:
                self.data = self.data.to(self.device)
            else:
                self.data = torch.from_numpy(self.data).float().to(self.device)
            

        def get_bounding_box(self):
            """
                Calculates and returns the bounding box of the data.

                Returns:
                    list: A list of [min, max] pairs for each dimension of the data.
            """
            # The first "dim" columns have the centers, the second "dim" columns have the sizes.
            # data are [batch, N, dim] (e.g. 100, 4,2) where 100 is the batch size, 4 is the number of points and 2 is the dimension
            self.object_id = self.ktree.clustering_args["object_id"]
            if self.object_id == "squares":
                x = self.data[:,0]
                y = self.data[:,1]
                l = self.data[:,2]
                bounding_box = [[x.min().item()-l.max().item(), x.max().item()+l.max().item()], [y.min().item()-l.max().item(), y.max().item()+l.max().item()]]
            elif self.object_id == "cuboids":
                x = self.data[:,0]
                y = self.data[:,1]
                z = self.data[:,2]
                l = self.data[:,3]
                w = self.data[:,4]
                h = self.data[:,5]
                bounding_box = [[x.min().item()-l.max().item(), x.max().item()+l.max().item()], [y.min().item()-w.max().item(), y.max().item()+w.max().item()], [z.min().item()-h.max().item(), z.max().item()+h.max().item()]]
            elif self.object_id == "ellipses":
                a = self.data[:,0]
                b = self.data[:,1]
                x = self.data[:,2]
                y = self.data[:,3]
                bounding_box = [[x.min().item()-a.max().item(), x.max().item()+a.max().item()], [y.min().item()-b.max().item(), y.max().item()+b.max().item()]]
            else:
                raise ValueError("Object ID not recognized.")
            
            # # bounding_box = [[min(centers[:, i] - sizes[:, i]), max(centers[:, i] + sizes[:, i])]
            # #                 for i in range(self.ktree.dim)]
            # if self.ktree.dim == 3:
            #     sizes = self.data[:, 3].to("cpu")
            #     bounding_box = [[min(centers[:, i] - sizes), max(centers[:, i] + sizes)] for i in range(self.ktree.dim)]
            # bounding_box = torch.tensor(bounding_box).to(self.device)
            return bounding_box
           

        def create_critic(self, save_path_prefix="", plot=False):
            """
                Creates the critic model. Main pipeline of the algorithm.
                    First a clustering model is trained and predicts the best centroids to fit the data.
                    Then points are sampled from areas where the best centroid is uncertain.
                    Finally, a critic model is trained to predict a query point to its best centroid.

                Parameters:
                    save_path_prefix (str): if set, the trained model and training parameters
                        will be saved in the path specified here, appended by the node's index
                    plot (bool): if set, plots about the trained models will be shown on runtime
            """
            # If the node has no data, do nothing.
            if len(self.data) == 0:
                return

            # First calculate the bounding box of the data.
            bounding_box = self.get_bounding_box()
            print(f"Bounding box for node {self.index}: {bounding_box}")

            """
            Clustering model.
            """

            # Create and train the clustering model.
            n_of_centroids = self.ktree.clustering_args["number_of_centroids"]
            print(f"Creating clustering for node {self.index} with {n_of_centroids} centroids.")
            if n_of_centroids == 0:
                n_of_centroids = 2 ** self.ktree.dim
            dim = self.ktree.clustering_args["dimension"]
            object_id = self.ktree.clustering_args["object_id"]
            if type(self.data) is not torch.Tensor:
                train_data = torch.from_numpy(self.data).float().to(self.device)
            else:
                train_data = self.data.float().to(self.device)
            clustering = ClusteringNew(train_data,
                                      n_of_centroids,
                                      dim,
                                      object_id,
                                      self.metric)

            epochs = self.ktree.clustering_args["epochs"]
            pre_processing = self.ktree.clustering_args["pre_processing"]
            clustering.fit(epochs,pre_processing)
            # Recalculate best_z indices since clustering.best_z is fuzzy from training.
            centroids = clustering.centroids
            self.best_z = clustering.labels
            # e = loss_functional(clustering.best_centroids, torch.from_numpy(self.data).float().to(self.device),
            #                     self.ktree.metric)
            # _, self.best_z = e.min(1)

            # Keep only the centroids that appear in the data predictions.
            # Also update the best_z indices to index only the kept centroids
            # e.g. convert (1, 0, 3) to (1, 0, 2); since centroid 2 is missing we count index 3 as 2.
            # Do so using the inverse indexing tensor of torch.unique().
            # best_z_unique, self.best_z = torch.unique(self.best_z, return_inverse=True)
            # centroids = clustering.best_centroids[best_z_unique]
            n_of_centroids = len(centroids)
            # Finally store the regularised projection from the best epoch.
            # self.best_reg = clustering.reg_proj_array[clustering.best_epoch]

            # If the division is fully unbalanced, i.e. all data is put into one child, no need for a critic.
            if n_of_centroids == 1:
                return

            """
            Uncertainty Area.
            """
            # Calculate the Uncertainty Area.
            m_points = getUncertaintyArea(centroids=centroids.cpu().detach().numpy(),
                                          bounding_box=bounding_box, **self.ktree.un_args)
            m_points = np.array(m_points)
            self.un_points = m_points

            # # Plot the Uncertainty Area.
            # if plot:
            #     # Plot the m points that are in the uncertainty area.
            #     fig = plt.figure(figsize=(10, 10))
            #     ax = fig.add_subplot(111)
            #     ax.scatter(m_points[:, 0], m_points[:, 1], s=20, c='royalblue',
            #                alpha=0.5, marker='*', label='Uncertainty Area')
            #     ax.contourf(manifold[:, :, 0], manifold[:, :, 1], manifold[:, :, 2],
            #                 levels=200, cmap='viridis', alpha=0.2)
            #     plt.legend()
            #     plt.show()

            """
            Labeling.
            """
            qp = np.random.permutation(m_points)
            qp = torch.tensor(qp)
            # Initialize the pseudo clusters.
            # Append data that their best_z is i for each centroid.
            pseudo_clusters = [self.data[self.best_z == i].cpu() for i in range(n_of_centroids)]
            # Create labels.
            outputs_shape = (qp.shape[0], n_of_centroids)
            F_ps = torch.zeros(outputs_shape)
            z_ps = torch.zeros(outputs_shape)
            for i in range(outputs_shape[0]):
                if i % 1000 == 0:
                    print(f"Labeled {i}/{outputs_shape[0]} points.")
                for j in range(outputs_shape[1]):
                    qpoint = qp[i].cpu().detach().numpy()
                    F_ps[i, j], z_ps[i, j] = torch.tensor(
                        NearestNeighbour(qpoint, pseudo_clusters[j], self.ktree.metric))
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
                # plot best_centroids
                # plt_bc = clustering.best_centroids.cpu().detach().numpy()
                # c = np.linspace(0, plt_bc.shape[0], plt_bc.shape[0])
                ax.scatter(plt_bc[:, 0], plt_bc[:, 1], c=c, s=200)
                pt.plot_data_on_manifold(fig, ax, self.data, size=10, limits=bounding_box[0] + bounding_box[1])
                plt.show()

            """
            Critic model.
            """
            # Create and train the critic model and assign the best state found during training.

            critic_args = self.ktree.critic_args
            width = critic_args["width"]
            depth = critic_args["depth"]
            print(f"Creating critic for node {self.index} with {n_of_centroids} centroids.")
            print("Device is:", self.device)
            critic = Critic(n_of_centroids, self.ktree.dim, self.ktree.dim, width, depth).to(
                self.device)  # initialize the voronoi network
            critic.train_(optimizer=torch.optim.Adam(critic.parameters(), lr=critic_args["optimizer_lr"]),
                          epochs=critic_args["epochs"],
                          device=self.device,
                          qp=qp,
                          F_ps=F_ps
                          )
            critic.eval()
            critic.load_state_dict(critic.best_vor_model_state)

            # Save the model and some training results.
            if save_path_prefix != "":
                # Save voronoi model.
                torch.save(critic.best_vor_model_state, f"{save_path_prefix}_critic_config.pt")
                print(f"Saved critic config to {save_path_prefix}_critic_config.pt")

                # Save training results to .npy format.
                critic_results = {
                    "best_vor_model_state": critic.best_vor_model_state,
                    "cost_ll": critic.cost_ll,
                    "acc_l": critic.acc_l,
                    "es": critic.es
                }
                np.save(f"{save_path_prefix}_critic_training_results.npy", critic_results)
                print(f"Saved critic training results to {save_path_prefix}_critic_training_results.npy")

            # Plot the critic training results.
            if plot:
                critic.plot_accuracy_and_loss(critic_args["epochs"])
                plt.show()

            # Store the trained critic.
            self.critic = critic

        def create_critic_from_config(self, path):
            """
                Creates the node critic from file configuration.

                Parameters:
                    path (str): the path in which to look for the model config file
            """
            width = self.ktree.critic_args["width"]
            depth = self.ktree.critic_args["depth"]

            # Get the model parameters from the state dict.
            state_dict = torch.load(path)
            # Last predictor layer has bias shape (n_centroids,).
            n_centroids = state_dict[f"predictor.{2 * depth - 2}.bias"].shape[0]

            # Build the critic object and assign it to the node.
            critic = Critic(n_centroids, self.ktree.dim, self.ktree.dim, width, depth).to(
                self.device)  # initialize the voronoi network
            critic.load_state_dict(torch.load(path))
            critic.eval()
            self.critic = critic

        def isLeaf(self):
            """
                Returns whether the node is a leaf or not.

                Returns:
                    bool: True if the node is a leaf, False otherwise
            """
            return len(self.children) == 0

        def divide(self):
            """Creates child nodes where every child stores the data that each centroid is closest (best_z)."""
            for cluster in range(self.critic.n_centroids):
                cluster_data = self.data[self.best_z == cluster]
                # Create a child node with the corresponding data.
                self.children.append(Ktree.Node(cluster_data, f"{self.index}{cluster}", self.ktree, self, self.device, self.metric))

        def query(self, query_point, k=1):
            """
                Returns (by brute force) the k nearest neighbours of the query point.

                Parameters:
                    query_point (torch.Tensor): a point vector (in the objects' dimension)
                    k (int): the number of the nearest neighbours to return

                Returns:
                    list[object]: the k nearest neighbours of the query point
            """
            query_point.to(self.device)
            query_point = torch.tensor(query_point) if not torch.is_tensor(query_point) else query_point
            # print("Query device is:", query_point.device)
            # print("Data device are: ")
            # print(torch.from_numpy(self.data[0]).double().to(self.device))
            # print("Data[0] device are: ", self.data[0].device)
            # prepare data
            if type(self.data) is not torch.Tensor:
                dists = torch.tensor(
                    [self.ktree.metric(torch.from_numpy(self.data[i]).double().to(self.device), query_point) for i in
                     range(len(self.data))])
            else:
                dists = torch.tensor(
                    [self.ktree.metric(self.data[i], query_point) for i in range(len(self.data))])
            
            # dists = torch.tensor(
            #     [self.ktree.metric(torch.from_numpy(self.data[i]).double().to(self.device), query_point) for i in
            #      range(len(self.data))])
            # k_smallest_indices = np.argsort(dists)[:k]
            k_smallest_indices = torch.argsort(dists)[:k]
            k_nearest = [self.data[i] for i in k_smallest_indices]

            return k_nearest

        def get_leaf_sums(self, query_points):
            """
                Recursive function to calculate the energy from the node's critic predictions
                    of given query points and add it to each of its children's respective energy.

                Parameters:
                    query_points (torch.Tensor): The query points.

                Returns:
                    torch.Tensor: A tensor containing the totally summed energies of the query point predictions
                        for each leaf node (equivalently for each path on the tree).
            """
            # Base case, for a leaf return 0s as its energies are calculated on the parent.
            if self.isLeaf():
                return torch.zeros((len(query_points), 1))

            # The critic predictions are the children energies.
            y_pred_children = self.critic(query_points)
            # Each y_pred_children column corresponds to the energies of a child node, so add that column
            # to each child's leaf sums, i.e. the tensor with their leaves' predictions.
            sums = [y_pred_children[:, i].reshape(-1, 1) + self.children[i].get_leaf_sums(query_points)
                    for i in range(len(self.children))]
            return torch.hstack(tuple(sums))

        def get_leaf_cumsums(self, query_points, y_pred=None):
            """
                Recursive function to calculate the cumulative energy from the node's critic predictions
                    of given query points and add it to each of its children's respective energy.

                Parameters:
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
            # the node energies (y_pred) and the critic predictions.
            y_pred_children = y_pred + self.critic(query_points)
            # Each
            cumsums = [y_pred_children[:, i].reshape(-1, 1) +
                       self.children[i].get_leaf_cumsums(query_points, y_pred_children[:, i])
                       for i in range(len(self.children))]
            return torch.hstack(tuple(cumsums))
