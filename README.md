# GeoCluster

A Hierarchical Embedding Predictive Energy Based Model for Geometric Clustering in Continuous Space Taught In a Self-Supervised Regularized manner
![header](/dev/images/arch.jpg)
Figure: Illustration of first hierarchical training.

## Method

We aim to simplify and optimize the proccess of finding the nearest neighbor in a set of objects given a queery point, and a metric. To achive this we propose a hierarchical embedding predictive ebm that encodes the nearest neigbor information in a linear head (the student). This result in a graph of networks that is used for inference.

The teacher is trained in a self-supervised manner with regularizers to learn the underlying structure of the data while the student is trained to mimic the teacher and focus on the nearest neighbor information.

Key points of our method are:

- Input Agnostic.
- Simple, since only the metric needs to be defined.
- Fast, since the student is a linear model, and is run on the GPU.
- Great scaling and higly parallelizable

## Installation

Clone the repository and install the requirements by running the following in your terminal:

```[BASH]
git clone https://github.com/PRigas96/GeoCluster
cd GeoCluster
conda env create -f environment.yml
source activate GeoCluster
```

## Information

You can check the following folders:

- `data`: contains the data used in the experiments
- `models`: contains the trained models
- `results`: contains the results of the experiments
- `src`: contains the source code

TODO: add more information
