import numpy as np
import torch


def random_queries(ktree, n=300, times=4, k=1):
    leaves = ktree.get_leaves()
    height = max([len(leaf.index) for leaf in leaves])

    mean_acc_per_layer = np.zeros(height - 1)
    for time in range(times):
        np.random.seed(time)
        random_points = torch.zeros(n, ktree.dim)
        space = ktree.root.get_bounding_box()
        for i in range(n):
            random_points[i] = torch.tensor([np.random.uniform(space[d][0], space[d][1]) for d in range(ktree.dim)])

        # Run each query and store the number of correct predictions per layer.
        correct_predictions_per_layer = np.zeros(height - 1)
        n_queries_per_layer = np.zeros(height - 1)
        for random_point in random_points:
            predictions_per_layer = ktree.query_verbose(random_point)["predictions per layer"]
            k_nearest_neighbors = ktree.root.query(random_point, k=k)

            is_correct = True
            for i, pred in enumerate(predictions_per_layer):
                n_queries_per_layer[i] += 1
                if k == 1 and not np.array_equal(pred, k_nearest_neighbors):
                    is_correct = False
                    break
                elif k > 1 and not any(np.array_equal(pred[0], k_nearest_neighbors[j]) for j in range(k)):
                    is_correct = False
                    break
                correct_predictions_per_layer[i] += 1

            for i in range(len(predictions_per_layer), height - 1):
                correct_predictions_per_layer[i] += int(is_correct)

        accuracy_per_layer = correct_predictions_per_layer / n * 100
        mean_acc_per_layer += accuracy_per_layer
        print("The number of queries per layer are:")
        print(n_queries_per_layer)
        print(f"The percentage of correct predictions per layer is:")
        print(accuracy_per_layer)

    print(f"The mean percentage of correct predictions per layer is:")
    print(mean_acc_per_layer / times)


def serialised_queries(ktree, n=500, k=1):
    leaves = ktree.get_leaves()
    height = max([len(leaf.index) for leaf in leaves])

    # Build the serialised query points.
    num = int(np.sqrt(n))
    space = ktree.root.get_bounding_box()
    linspace = [torch.linspace(space[d][0], space[d][1], num) for d in range(ktree.dim)]
    serialised_points = torch.cartesian_prod(*linspace)

    # Run each query and store the number of correct predictions per layer.
    correct_predictions_per_layer = np.zeros(height - 1)
    n_queries_per_layer = np.zeros(height - 1)
    for serialised_point in serialised_points:
        predictions_per_layer = ktree.query_verbose(serialised_point)["predictions per layer"]
        k_nearest_neighbors = ktree.root.query(serialised_point, k=k)

        is_correct = True
        for i, pred in enumerate(predictions_per_layer):
            n_queries_per_layer[i] += 1
            if k == 1 and not np.array_equal(pred, k_nearest_neighbors):
                is_correct = False
                break
            elif k > 1 and not any(np.array_equal(pred[0], k_nearest_neighbors[j]) for j in range(k)):
                is_correct = False
                break
            correct_predictions_per_layer[i] += 1

        for i in range(len(predictions_per_layer), height - 1):
            correct_predictions_per_layer[i] += int(is_correct)

    accuracy_per_layer = correct_predictions_per_layer / n * 100
    print("The number of queries per layer are:")
    print(n_queries_per_layer)
    print(f"The percentage of correct predictions per layer is: ")
    print(accuracy_per_layer)
