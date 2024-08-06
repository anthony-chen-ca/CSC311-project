import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.
    See https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html for details.
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.
    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    transposed_matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = (nbrs.fit_transform(transposed_matrix)).T  # Fit transform and then transpose back to original
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def knn_impute_hybrid(matrix, valid_data, user_k, item_k, alpha):
    def compute_similarity_with_metadata(mat, meta):
        # Compute similarity matrix incorporating metadata
        pass  # Implement similarity computation with metadata
    # Apply user-based KNN
    user_nbrs = KNNImputer(n_neighbors=user_k)
    user_mat = user_nbrs.fit_transform(matrix)
    user_acc = sparse_matrix_evaluate(valid_data, user_mat)

    # Apply item-based KNN
    item_nbrs = KNNImputer(n_neighbors=item_k)
    transposed_matrix = matrix.T
    item_mat = (item_nbrs.fit_transform(transposed_matrix)).T
    item_acc = sparse_matrix_evaluate(valid_data, item_mat)

    # Combine both with weighting
    hybrid_mat = alpha * user_mat + (1 - alpha) * item_mat
    hybrid_acc = sparse_matrix_evaluate(valid_data, hybrid_mat)
    print(f"Hybrid Validation Accuracy: {hybrid_acc}")

    return hybrid_acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    question_meta = pd.read_csv("./data/question_meta.csv")
    student_meta = pd.read_csv("./data/student_meta.csv")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    print(question_meta.head())
    print(student_meta.head())

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # k_values = [1, 6, 11, 16, 21, 26]
    # user_accuracies = []
    # item_accuracies = []
    #
    # for k in k_values:
    #     print(f"Running KNN for k={k}:")
    #     # Compute the validation accuracies
    #     user_accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
    #     item_accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
    #     user_accuracies.append(user_accuracy)
    #     item_accuracies.append(item_accuracy)
    #
    # # Pick k* with the best performance
    # best_user_k = k_values[np.argmax(user_accuracies)]
    # best_item_k = k_values[np.argmax(item_accuracies)]
    # print(f"Best user k: {best_user_k}")
    # print(f"Best item k: {best_item_k}")
    #
    # # Evaluate on the test set using the best k
    # test_user_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_user_k)
    # test_item_accuracy = knn_impute_by_item(sparse_matrix, test_data, best_item_k)
    # print(f"Test User Accuracy for best k={best_user_k}: {test_user_accuracy}")
    # print(f"Test Item Accuracy for best k={best_item_k}: {test_item_accuracy}")
    #
    # # Plot the validation accuracy as a function of k
    # plt.plot(k_values, user_accuracies, marker='o')
    # plt.xlabel("k")
    # plt.ylabel("Validation Accuracy")
    # plt.title("(User) Validation Accuracy vs. k")
    # plt.savefig("knn-user.png")
    #
    # plt.clf()
    #
    # plt.plot(k_values, item_accuracies, marker='o')
    # plt.xlabel("k")
    # plt.ylabel("Validation Accuracy")
    # plt.title("(Item) Validation Accuracy vs. k")
    # plt.savefig("knn-item.png")

    user_k = 11
    item_k = 21
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    hybrid_accuracies = []

    for alpha in alpha_values:
        print(f"Running Hybrid KNN with alpha={alpha}:")
        hybrid_accuracy = knn_impute_hybrid(sparse_matrix, val_data, user_k, item_k, alpha)
        hybrid_accuracies.append(hybrid_accuracy)

    best_alpha = alpha_values[np.argmax(hybrid_accuracies)]
    print(f"Best alpha: {best_alpha}")

    # Evaluate on the test set using the best alpha
    test_hybrid_accuracy = knn_impute_hybrid(sparse_matrix, test_data, user_k, item_k, best_alpha)
    print(f"Test Hybrid Accuracy for best alpha={best_alpha}: {test_hybrid_accuracy}")

    # Plot the hybrid validation accuracy as a function of alpha
    plt.plot(alpha_values, hybrid_accuracies, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("Validation Accuracy")
    plt.title("Hybrid Validation Accuracy vs. alpha")
    plt.savefig("knn-hybrid.png")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
