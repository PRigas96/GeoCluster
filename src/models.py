import torch
import numpy as np
from torch import nn
from torch.nn.functional import one_hot
from torch.distributions import Dirichlet
from src.utils.embeddings import Reg, RegLatent, loss_functional
import matplotlib.pyplot as plt
from copy import deepcopy
import math as m
from src.metrics import get_dist_matrix_ls, compute_distances_2d, compute_distances_3d


class Clustering(nn.Module):
    """
        Latent Variable Generative Energy Based Model

        Parameters:
            n_centroids: number of centroids
            output_dim: output dimension
            latent_size: size of the latent space
            
        Args:
            z: latent space
            decoder: decoder
            projector: projector
            z_l: latent space without noise
            z_fuzzy: latent space with noise
            c: projected latent space
            c_fuzzy: projected latent space with noise
            
        Returns:
            forward: projected latent space
            sample_z: sample latent space
            col_one_hot: one hot encoding
            get_z: get latent space

    """

    def __init__(self,
                 n_centroids,
                 output_dim,
                 encoder_activation,
                 encoder_depth,
                 projector_width,
                 projector_depth,
                 latent_size=400,
                 node_index="0",
                 parent_node=None,
                 dim=2):
        super(Clustering, self).__init__()
        self.n_centroids = n_centroids
        self.output_dim = output_dim
        self.latent_size = latent_size
        self.node_index = node_index
        self.parent_node = parent_node
        # decoder is a linear layer
        enc_layer = []
        for i in range(encoder_depth):
            # encoder will start from latent size and go to output dim in encoder_depth steps.  No width is needed
            times = latent_size // encoder_depth  # e.g. 400 // 4 = 100
            times = times if times != 0 else 1
            flag = encoder_activation
            if i == 0:
                enc_layer.append(nn.Linear(latent_size, latent_size // (times * (i + 1))))
                if flag:
                    enc_layer.append(nn.ReLU())
            elif i == encoder_depth - 1:
                enc_layer.append(nn.Linear(latent_size // (times * (i)), output_dim))
            else:
                enc_layer.append(nn.Linear(latent_size // (times * (i)), latent_size // (times * (i + 1))))
                if flag:
                    enc_layer.append(nn.ReLU())

        self.encoder = nn.Sequential(*enc_layer)
        # self.encoder = nn.Sequential(nn.Linear(latent_size, output_dim, bias=False),)
        # init z to one-hot descrete latent variable
        self.z = nn.Parameter(
            Dirichlet(torch.ones(n_centroids)).sample((latent_size,)).T
            , requires_grad=True
        )
        predictor_layer = []
        for i in range(projector_depth):
            if i == 0:
                predictor_layer.append(nn.Linear(n_centroids, projector_width // 2))
                predictor_layer.append(nn.ReLU())
                predictor_layer.append(nn.Linear(projector_width // 2, projector_width))
            elif i == projector_depth - 1:
                predictor_layer.append(nn.Linear(projector_width, n_centroids))
            else:
                predictor_layer.append(nn.Linear(projector_width, projector_width))
                predictor_layer.append(nn.ReLU())
        self.predictor = nn.Sequential(*predictor_layer)

        self.projector = nn.Sequential(
            nn.Linear(n_centroids, n_centroids * 10),
            nn.ReLU(),
            nn.Linear(n_centroids * 10, n_centroids * 100),
            nn.ReLU(),
            nn.Linear(n_centroids * 100, n_centroids * 10),
            nn.ReLU(),
            nn.Linear(n_centroids * 10, n_centroids),
        )
        self.z_l = None
        self.z_fuzzy = None
        self.c = None
        self.c_fuzzy = None

        # Store training variables.
        self.best_model_state = None  # best model state
        self.best_centroids = None  # best centroids
        self.best_z = None  # best latent
        self.best_lat = None  # best latent space
        self.best_epoch = None  # best epoch
        self.p_p = None  # saved centroids
        self.p_c = None  # saved costs
        self.reg_proj_array = None  # saved projection regularizers
        self.reg_latent_array = None  # saved latent regularizers
        self.memory = None  # saved memory
        self.cost_array = None  # saved costs
        self.dim = dim

    def col_one_hot(self, z):
        # one hot encoding
        # z = torch.argmax(z, dim=0)
        z = torch.argmin(z, dim=0)
        z = one_hot(z, num_classes=self.n_centroids).T
        if self.n_centroids == self.latent_size:
            # make z eye matrix
            z = torch.eye(self.n_centroids)

        return z.float()

    def get_z(self):
        return self.z

    # sample z from without one-hot encoding
    def sample_z(self, regularize=False):
        if regularize:
            z = self.z
        else:
            z = torch.bernoulli(self.z)
        y = self.encoder(z)
        return y, z

    def pump(self):
        self.c

    def forward(self, z):
        z = self.col_one_hot(z)
        # make z dtype float
        y = self.encoder(z)
        self.z_l = y
        # make y fuzzy
        y_std = torch.randn(y.shape) * 1
        y_std = y_std.to(y.device)
        y = y + y_std
        self.z_fuzzy = y
        # project y_hat to space of y
        y_hat = self.predictor(y.T).T
        self.c = y_hat

        return y_hat

    # debugging number of centroids
    def train_(self, optimizer, epochs, times, train_data, bounding_box, metric,
               number_of_centroids,
               latent_size,
               encoder_activation,
               encoder_depth,
               predictor_width,
               predictor_depth,
               alpha=10,
               beta=10,
               gamma=10,
               delta=0.1,
               f_clk=2,
               scale=1e-2,
               scale_flag=False,
               bound_for_saving=6000):
        """
            Train the clustering model

            Parameters:
                optimizer: optimizer to be used
                epochs: number of epochs
                times: number of times to print
                train_data: data to be trained on
                alpha: regularizer for projection module
                beta: regularizer for latent space
                f_clk: frequency of the clock
                scale: scale of the noise
                bound_for_saving: bound for saving the data
                bounding_box (list[list]): list of [min, max] limits for each coordinate
        """
        print("=" * 20)
        print("Training Clustering Model")
        p_times = epochs // times  # print times
        y_prep = train_data
        num = 0
        shuffle_flag = False
        if y_prep.shape[0] > 5000:
            print("Shuffling data")
            shuffle_flag = True
            num = 700
        elif y_prep.shape[0] > 1000:
            print("Shuffling data")
            shuffle_flag = True
            num = m.floor(0.1 * y_prep.shape[0])
        p_p, p_c = [], []
        reg_proj_array = []
        reg_latent_array = []
        memory = []
        cost_array = []
        best_model_state = {}
        best_cost = torch.inf
        reg_latent = torch.tensor(0.0)
        for epoch in range(epochs):
            # forward
            if shuffle_flag:
                y = y_prep[torch.randperm(y_prep.shape[0])][:num]
            else:
                y = y_prep
            y_pred = self(self.z)
            k = y_pred.shape[0]
            dim = y_pred.shape[1]
            # add pump
            if epoch % f_clk == 0 and epoch != 0:
                if scale_flag:
                    scale = len(self.data)
                y_pred_std = torch.randn(y_pred.shape) * memory[-1] * scale * delta
                y_pred_std = y_pred_std.to(y_pred.device)
                y_pred = y_pred + y_pred_std

            reg_proj = Reg(y_pred, bounding_box, self.node_index, self.parent_node)  # regularize projection module
            if reg_proj == 0:
                reg_proj = torch.tensor(0.0)
            reg_proj_array.append(reg_proj)  # save reg_proj

            if epoch > 1:
                reg_latent = RegLatent(self.z_l)  # regularize latent space
                reg_latent_array.append(reg_latent.item())  # save reg_latent

            # print("y_pred device is: ", y_pred.device)
            # print("y device is: ", y.device)
            e = loss_functional(y_pred, y, metric)

            F, z = e.min(1)  # get energy and latent
            div = torch.mean(F).item()  # get average energy
            memory.append(div)  # save memory
            # create a repulsive force between the centroids
            F_r = torch.zeros(self.n_centroids, self.n_centroids)
            centroids = y_pred
            for i in range(self.n_centroids):
                for j in range(self.n_centroids):
                    if i != j:
                        F_r[i, j] = torch.max(torch.abs(centroids[i] - centroids[j]))
            # if 1 centroid is near another i wanna penaliize it
            cost = torch.mean(abs(F)) + alpha * reg_proj + beta * reg_latent  # add regularizers

            # i wanna penalize small F_r, the smaller the more penalized

            fr = torch.zeros(1)
            for i in range(self.n_centroids):
                for j in range(self.n_centroids):
                    if i != j:
                        fr += 1 / F_r[i, j]
            f_rep = gamma * 0.5 * torch.sum(fr) * dim
            cost += f_rep

            cost_array.append(cost.item())  # save cost

            # backward
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if cost < best_cost:
                best_cost = cost
                best_model_state = deepcopy(self.state_dict())
                best_centroids = y_pred
                best_z = z
                best_lat = self.z_l
                best_epoch = epoch
            if cost < bound_for_saving:
                p_p.append(y_pred)
                p_c.append(cost)
            if (epoch + 1) % p_times == 0:
                print("=" * 20)
                # print("Centroids: ", y_pred)
                # print("torch.mean(F): ", torch.mean(F).item())
                # print("reg_proj: ", alpha*reg_proj.item())
                # print("reg_latent: ", beta * reg_latent.item())
                # print("Repulsive: ", f_rep.item())

                # print
                print("Epoch: {}/{}.. \n".format(epoch + 1, epochs),
                      "Training loss: {:.5f}.. \n".format(cost.item()),
                      "torch.mean(F): {:.5f}.. \n".format(div),
                      "Reg Proj: {:.5f}.. \n".format(alpha * reg_proj.item()),
                      "Reg Latent: {:.5f}.. \n".format(beta * reg_latent.item()),
                      "Repulsive: {:.5f}.. \n".format(f_rep.item()),
                      "Memory: {:.5f}.. \n".format(memory[-1] * delta),
                      "Memory: {:.5f}.. \n".format(div * delta),
                      "Output: \n", y_pred.cpu().detach().numpy(),
                      )

        # Store the training variables to the model.
        self.best_model_state = best_model_state
        self.best_centroids = best_centroids
        self.best_z = best_z
        self.best_lat = best_lat
        self.best_epoch = best_epoch
        self.p_p = p_p
        self.p_c = p_c
        self.reg_proj_array = reg_proj_array
        self.reg_latent_array = reg_latent_array
        self.memory = memory
        self.cost_array = cost_array

class ClusteringLS:
    def __init__(self, data, n_clusters, dim, dist_function):
        self.data = data
        self.n_clusters = n_clusters
        self.dim = dim
        self.dist_function = compute_distances_2d if dim == 2 else compute_distances_3d

    def kmeans_pp_greedy(
        self,
        random_state=None,
        n_trials=None,
    ):
        # check parameters
        if type(self.data) is not torch.Tensor:
            self.data = torch.tensor(self.data)
        n_samples, n_features = self.data.shape
        if self.n_clusters > n_samples:
            raise ValueError(
                "n_clusters should be smaller or equal to the number of centroids"
            )
        if type(self.n_clusters) is not torch.Tensor:
            selfn_clusters = torch.tensor(self.n_clusters)
        if n_trials is None:
            n_trials = 2 + int(torch.log(self.n_clusters))
        # set random state
        if random_state is not None:
            torch.manual_seed(random_state)
        # initialize centroids
        centroids = torch.zeros(self.n_clusters, self.dim)
        # choose first centroid
        first_centroid_idx = torch.randint(n_samples, (1,))
        # print(f"First centroid index: {first_centroid_idx}")
        data_oi = self.data[first_centroid_idx]
        if self.dim == 2:
            x0, y0, l, theta = data_oi[0]
            centroids[0] = torch.tensor([x0, y0])
            centroids[0] += torch.tensor([0.5 * l * torch.cos(theta), l * torch.sin(theta)])
        elif self.dim == 3:
            x0, y0, z0, l, theta, phi = data_oi[0]
            centroids[0] = torch.tensor([x0, y0, z0])
            centroids[0] += torch.tensor(
                [
                    0.5 * l * torch.sin(theta) * torch.cos(phi),
                    0.5 * l * torch.sin(theta) * torch.sin(phi),
                    0.5 * l * torch.cos(theta),
                ]
            )
        else:
            raise ValueError("dim should be 2 or 3")
        # create a vector of minus ones of shape (n_samples,)
        indices = -torch.ones(n_samples)
        # # init dist matrix
        # print(f"Data shape: {data.shape}")
        # print(f"Centroids shape: {centroids.shape}")
        # print(f"Dist function: {dist_function}")
        dist_matrix = get_dist_matrix_ls(self.data, centroids[:1], self.dist_function)

        for i in range(1, self.n_clusters):
            # choose the next centroid
            freq = torch.zeros(n_trials)
            for _ in range(n_trials):
                # choose a centroid with probability proportional to the distance
                # to the closest centroid
                dists = torch.min(dist_matrix, dim=1).values
                probs = dists / torch.sum(dists)
                next_centroid_idx = torch.multinomial(probs, 1)
                # update best_dist
                freq[_] = next_centroid_idx
            # pick randomly a vlaue from freq
            idx = torch.randint(n_trials, (1,))
            next_centroid_idx = freq[idx].int()
            data_oi = self.data[next_centroid_idx]
            if self.dim == 2:
                x0, y0, l, theta = data_oi[0]
                centroids[i] = torch.tensor([x0, y0])
                centroids[i] += torch.tensor(
                    [0.5 * l * torch.cos(theta), l * torch.sin(theta)]
                )
            elif self.dim == 3:
                x0, y0, z0, l, theta, phi = data_oi[0]
                centroids[i] = torch.tensor([x0, y0, z0])
                centroids[i] += torch.tensor(
                    [
                        0.5 * l * torch.sin(theta) * torch.cos(phi),
                        0.5 * l * torch.sin(theta) * torch.sin(phi),
                        0.5 * l * torch.cos(theta),
                    ]
                )
            else:
                raise ValueError("dim should be 2 or 3")
            # update dist_matrix
            dist_matrix = get_dist_matrix_ls(self.data, centroids[: i + 1], self.dist_function)
        return centroids

    def get_points_from_emb(self):
        """
        Get the points from the embedding

        Returns
        -------
        torch.Tensor
            The points in the space [x0,y0,x1,y1,...]
        """
        data_points = torch.zeros(self.data.shape[0], 2 * self.dim)
        if self.dim == 2:  # [x0,y0,l,theta]
            data_points[:, 0] = self.data[:, 0]
            data_points[:, 1] = self.data[:, 1]
            data_points[:, 2] = self.data[:, 0] + self.data[:, 2] * torch.cos(self.data[:, 3])
            data_points[:, 3] = self.data[:, 1] + self.data[:, 2] * torch.sin(self.data[:, 3])
        elif self.dim == 3:  # [x0,y0,z0,l,theta,phi]
            data_points[:, 0] = self.data[:, 0]
            data_points[:, 1] = self.data[:, 1]
            data_points[:, 2] = self.data[:, 2]
            data_points[:, 3] = self.data[:, 0] + self.data[:, 3] * torch.sin(
                self.data[:, 4]
            ) * torch.cos(self.data[:, 5])
            data_points[:, 4] = self.data[:, 1] + self.data[:, 3] * torch.sin(
                self.data[:, 4]
            ) * torch.sin(self.data[:, 5])
            data_points[:, 5] = self.data[:, 2] + self.data[:, 3] * torch.cos(self.data[:, 4])

        else:
            raise ValueError("dim should be 2 or 3")
        return data_points

    def fit(self, n_iter, n_trials=10):
        self.data_points = self.get_points_from_emb(self.data, self.dim)
        # initialize using kmeans++ greedy
        centroids = self.kmeans_pp_greedy(
            self.data, self.k, self.dist_function, self.dim, n_trials=n_trials
        )
        # get initial divergence
        dist_matrix = get_dist_matrix_ls(self.data, centroids, self.dist_function)
        div = torch.sum(torch.min(dist_matrix, dim=1).values)
        print(f"Initial divergence: {div}")

        # do the iterations
        for i in range(n_iter):
            # get the distances
            dist_matrix = get_dist_matrix_ls(self.data, centroids, self.dist_function)
            # get the labels
            labels = torch.argmin(dist_matrix, dim=1)
            # update the centroids
            for j in range(self.k):
                dp = self.data_points[labels == j]
                centroids[j][0] = torch.mean(torch.concatenate([dp[:, 0], dp[:, 2]]))
                centroids[j][1] = torch.mean(torch.concatenate([dp[:, 1], dp[:, 3]]))
                if self.dim == 3:
                    centroids[j][2] = torch.mean(
                        torch.concatenate([dp[:, 2], dp[:, 4]])
                    )
            # get the divergence
            div = torch.sum(torch.min(dist_matrix, dim=1).values)
            print(f"Iteration {i+1}, divergence: {div}")
        self.centroids = centroids

    def predict(self, centroids):
        # get the distances
        dist_matrix = get_dist_matrix_ls(self.data, centroids, self.dist_function)
        # get the labels
        labels = torch.argmin(dist_matrix, dim=1)
        return labels

class Critic(nn.Module):
    """
        Voronoi Energy Based Model
        
        Parameters:
            n_centroids: number of centroids
            input_dim: input dimension
            output_dim: output dimension

        Args:
            predictor: predictor

        Returns:
            forward: predicted energy
    """

    def __init__(self, n_centroids, input_dim, output_dim, width, depth):
        super(Critic, self).__init__()
        self.n_centroids = n_centroids
        self.input_dim = input_dim
        self.output_dim = output_dim
        # inputs are datapoints and outputs are energies
        # initialize self.predictor
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, width))
                layers.append(nn.ReLU())
            elif i == depth - 1:
                layers.append(nn.Linear(width, n_centroids))
            else:
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())

        self.predictor = nn.Sequential(*layers)

        self.predictor_before = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_centroids)
        )

        # Store training variables.
        self.best_vor_model_state = None
        self.cost_ll = None
        self.acc_l = None
        self.es = None

    def forward(self, x):
        x = self.predictor(x)
        return x

    def train_(self,
               optimizer,
               epochs,
               device,
               qp,
               F_ps,
               times=10
               ):
        """
            Train the critic model

            Parameters:
                optimizer: optimizer to be used
                epochs: number of epochs
                device: device to be used
                qp: qp points
                F_ps: ???
        """
        print("=" * 20)
        print("Training Critic Model")
        cost_l = []
        cost_ll = []
        qp = qp if torch.is_tensor(qp) else torch.tensor(qp)
        qp = qp.float().to(device)

        ce = nn.CrossEntropyLoss()
        acc_l = []
        es = []
        best_vor_cost = torch.inf
        best_vor_model_state = None
        times = 10
        at = epochs // times
        for epoch in range(epochs):
            # get outputs
            outputs = self(qp)
            # pass outputs through a hard arg max
            F_ps_m, z_ps_m = F_ps.min(1)
            # send to device
            z_ps_m = z_ps_m.to(device)
            # get loss
            z_ = outputs
            # make z_ float
            z_ = z_.float()
            z_cost = ce(z_, z_ps_m)
            # uncomment for debugging
            # print("z_ is: ", z_)
            # print("z_ps_m is: ", z_ps_m)
            cost = 100 * z_cost
            cost_l.append(cost.item())
            # backward
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if cost < best_vor_cost:
                best_vor_cost = cost
                best_vor_model_state = deepcopy(self.state_dict())
            if epoch % at == 0:
                # lets check acc
                acc = 0
                for i in range(qp.shape[0]):
                    F_e, z_e = outputs[i].max(0)
                    if z_e == z_ps_m[i]:
                        acc += 1
                acc_l.append(acc / qp.shape[0])
                es.append(epoch)
                cost_ll.append(cost.item())
                print("Acc: ", acc / qp.shape[0])
                print("Epoch: ", epoch, "Cost: ", cost.item())

        # Store the training variables to the model.
        self.best_vor_model_state = best_vor_model_state
        self.cost_ll = cost_ll
        self.acc_l = acc_l
        self.es = es

    def plot_accuracy_and_loss(self, epochs):
        cost_ll_log = np.log(self.cost_ll)

        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax1.plot(self.es, self.acc_l, c='r', label='Accuracy', linestyle='dotted', linewidth=3)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='upper left')
        ax1.set_ylim([0, 1])
        # create another twin axis
        ax2 = ax1.twinx()
        ax2.plot(self.es, cost_ll_log, c='royalblue', label='Loss', linestyle='dashed', linewidth=3)
        ax2.set_ylabel('Log Loss')
        ax2.legend(loc='upper right')
        ax2.set_ylim([-1, 6])
        # set x_lim in both
        ax1.set_xlim([0, epochs])
        ax2.set_xlim([0, epochs])
        # add stars to the plot points
        ax1.scatter(self.es, self.acc_l, c='r', s=100, marker='*')
        ax2.scatter(self.es, cost_ll_log, c='royalblue', s=100, marker='*')
