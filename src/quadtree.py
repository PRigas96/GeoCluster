import torch
import numpy as np
from src.models import LVGEBM, Voronoi
from src.utils.functions import getUncertaintyArea, getE, NearestNeighbour

class QuadTree:
    """Implements a quadtree class to use in Hierarchical Clustering"""

    def __init__(self, threshold, data, teacher_args, un_args, student_args):
        # self.boundary = boundary        #The given bounding box
        self.threshold = threshold      #Minimum number of data (objects) in a node
        self.data = data                #The input data (objects)
        self.divided = False
        self.root = self.Node(self.data, 0, self)
        self.root.create_student(teacher_args, un_args, student_args)   #Extract the trained student model for the root node

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

        def create_student(self, teacher_args, un_args, student_args, plot=False):
            # Create and train the teacher model.
            teacher = LVGEBM(4, 2, 400).to(self.device)
            # Populate the optimizer with the model parameters and the learning rate.
            teacher_args["optimizer"] = torch.optim.Adam(teacher.parameters(), lr=teacher_args["optimizer_lr"])
            del teacher_args["optimizer_lr"]
            teacher.train_(**teacher_args)
            # Get best outputs.
            best_outputs = teacher(teacher.z)
            # TODO: Save model and plot training data.

            # Uncertainty Area.
            m_points = getUncertaintyArea(outputs=best_outputs.cpu().detach().numpy(), **un_args)
            m_points = np.array(m_points)

            # Labeling.
            qp = np.random.permutation(m_points)
            qp = torch.tensor(qp)
            
            # TODO: Pass squares to getE as an argument.
            F, z, F_sq, z_sq = getE(teacher, teacher.best_outputs.cpu(), qp, self.data)
            # TODO: Plot Uncertainty Area.

            # Initialize the pseudo clusters.
            # Append data that their z_sq is i for each centroid.
            pseudo_clusters = [self.data[z_sq == i] for i in range(teacher.n_centroids)]

            # Create labels.
            outputs_shape = (qp.shape[0], teacher.n_centroids)
            F_ps = torch.zeros(outputs_shape)
            z_ps = torch.zeros(outputs_shape)
            for i in range(outputs_shape[0]):
                for j in range(outputs_shape[1]):
                    qpoint = qp[i].cpu().detach().numpy()
                    F_ps[i, j], z_ps[i, j] = torch.tensor(NearestNeighbour(qpoint, pseudo_clusters[j]))
            # TODO: Plot labels.

            # Create and train the student model.
            student = Voronoi(4, 2, 2).to(self.device)  # initialize the voronoi network
            student.train_(optimizer=torch.optim.Adam(student.parameters(), lr=student_args["optimizer_lr"]),
                        epochs=student_args["epochs"],
                        device=self.device,
                        qp=torch.tensor(qp),
                        F_ps=F_ps)
            # TODO: Save model and plot training data.

            self.student = student
            self.best_z = teacher.best_z

        def create_student_from_config(self, path):
            student = Voronoi(4, 2, 2).to(self.device)
            student.load_state_dict(torch.load(path))
            student.eval()
            self.student = student

        def isLeaf(self, data):
            return len(data) <= self.quadtree.threshold

        #Create 4 child nodes, where every child stores one of the four clusters produced
        def divide(self):
            #Retrieve the data that exist in each of the clusters
            unique_clusters = torch.unique(self.teacher.best_z)
            for cluster in unique_clusters:
                cluster_data = self.data[self.teacher.best_z == cluster]
                #Create a child node with the corresponding data
                self.children.append(QuadTree.Node(cluster_data, cluster.item(), self.quadtree))
