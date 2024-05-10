## Source Files
----------------
The source files are composed by the following python files: 
- 'ktree.py': contains the k-tree class and methods 
- 'metrics.py': contains the metric functions
- 'models.py': contains the model (Clustering, Critic) classes and methods

The folder [utils](utils) contains the utility functions used by the above files as well as functions used for the experiments and plots.
- 'objects': folder containing scripts to generate and load data for squares, cuboids and ellipses objects
- 'accuracy.py': contains functions to calculate the accuracy of the models
- 'embeddings.py': contains helper functions related to embeddings, i.e. regularise the outputs, calculate the model's loss functional and get the uncertainty area
- 'functions.py': contains generic functions, currently only the Nearest Neighbor search for a given metric
- 'plot_tools.py': contains functions to plot data, training results and model properties
