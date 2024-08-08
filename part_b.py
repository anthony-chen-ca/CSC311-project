from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def process_student_metadata(student_meta):
    """Process student metadata and return a DataFrame with numerical features.
    These features include: gender, age, and premium.
    """
    # Gender: 0 = unspecified, 1 = male, 2 = female
    gender = pd.get_dummies(student_meta['gender'], prefix='gender', dummy_na=True)

    # Date of Birth: Calculate age, mark unknown ages as -1
    dob = "data_of_birth"  # typo from dataset
    current_year = datetime.now().year
    student_meta[dob] = pd.to_datetime(student_meta[dob], errors='coerce')
    student_meta['age'] = current_year - student_meta[dob].dt.year
    student_meta.loc[:, 'age'] = student_meta['age'].fillna(-1)

    # Premium Pupil: 0 = not premium, 1 = yes premium, -1 = unknown
    student_meta.loc[:, 'premium_pupil'] = student_meta['premium_pupil'].fillna(-1)

    # Combine all processed features (gender, age, premium) into a single DataFrame
    student_features = pd.concat([gender, student_meta['age'], student_meta['premium_pupil']], axis=1)
    student_features.index = student_meta['user_id']

    return student_features


def process_question_metadata(question_meta):
    """Process question metadata and return a DataFrame with numerical features.
    Each question ID row will have a one-hot encoded list representing subjects.
    """
    # Subject ID: Convert string representation of lists into actual lists, then one-hot encode
    question_meta['subject_id'] = question_meta['subject_id'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    question_features = question_meta['subject_id'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0)

    question_features.index = question_meta['question_id']

    return question_features


def compute_similarity_matrices(student_meta, question_meta):
    """Compute the similarity matrices for student metadata and question metadata.
    """
    # Process student metadata
    student_weights = process_student_metadata(student_meta)
    student_similarity_matrix = cosine_similarity(student_weights)

    # Process question metadata
    question_weights = process_question_metadata(question_meta)
    question_similarity_matrix = cosine_similarity(question_weights)

    return student_similarity_matrix, question_similarity_matrix


def weight_function(distances, similarity_matrix, indices):
    """Custom weight function that takes an array of distances and returns an array containing the weights.
    Each entry in weight array corresponds to the metadata similarity between a point and its neighbor.
    """
    weights = np.zeros_like(distances)
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors):
            weights[i, j] = similarity_matrix[i, neighbor]
    # Normalize weights to ensure they sum to 1 for each row
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


def custom_weights(d, weights):
    """Custom weight function built for KNNImputer.
    """
    sorted_indices = np.argsort(d, axis=1)
    row_indices = np.arange(len(d))[:, None]
    return weights[row_indices, sorted_indices]


def preprocess_matrix(matrix, k):
    """Imputes NaN values using a very basic KNNImputer, so we can use NearestNeighbors.
    """
    imputer = KNNImputer(n_neighbors=k)
    return imputer.fit_transform(matrix)


def weighted_knn_impute(matrix, similarity_matrix, valid_data, k, transposed=False):
    """Weighted KNN with imputation. The k nearest neighbors are chosen based on student or question similarity, and
    then the chosen neighbors are given weights based on metadata. Finally, fills in the missing values using those
    adjusted weights.

    Returns imputed matrix and accuracy.
    """
    if transposed:
        matrix = matrix.T
    preprocessed_matrix = preprocess_matrix(matrix, k)
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(preprocessed_matrix)
    # Get the distances and indices of the k nearest neighbors
    distances, indices = nbrs.kneighbors(preprocessed_matrix, return_distance=True)

    # Perform KNN Imputation with custom weights
    weights = weight_function(distances, similarity_matrix, indices)
    weighted_nbrs = KNNImputer(n_neighbors=k, weights=lambda d: custom_weights(d, weights))
    imputed_matrix = weighted_nbrs.fit_transform(matrix)  # original matrix
    if transposed:
        imputed_matrix = imputed_matrix.T
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    return imputed_matrix, acc


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
    return mat, acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    transposed_matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = (nbrs.fit_transform(transposed_matrix)).T  # Fit transform and then transpose back to original
    acc = sparse_matrix_evaluate(valid_data, mat)
    return mat, acc


def hybrid_weighted_knn_impute(matrix, valid_data, student_similarity_matrix, question_similarity_matrix,
                               user_k, item_k, alpha):
    """Hybrid weighted KNN imputation using both user and item. Returns the hybrid accuracy.
    """
    # Apply user-based weighted KNN
    user_mat, user_acc = weighted_knn_impute(matrix, student_similarity_matrix, valid_data, user_k)
    print(f"Weighted User Accuracy: {user_acc}")

    # Apply item-based weighted KNN
    item_mat, item_acc = weighted_knn_impute(matrix, question_similarity_matrix, valid_data, item_k, transposed=True)
    print(f"Weighted Item Accuracy: {item_acc}")

    # Combine both with weighting
    hybrid_mat = alpha * user_mat + (1 - alpha) * item_mat
    hybrid_acc = sparse_matrix_evaluate(valid_data, hybrid_mat)
    print(f"Hybrid Weighted Accuracy: {hybrid_acc}")

    return hybrid_acc


def hybrid_knn_impute(matrix, valid_data, user_k, item_k, alpha):
    """Hybrid KNN imputation using both user and item. Returns the hybrid accuracy.
    """
    # Apply user-based KNN
    user_mat, user_acc = knn_impute_by_user(matrix, valid_data, user_k)
    print(f"User-based Accuracy: {user_acc}")

    # Apply item-based KNN
    item_mat, item_acc = knn_impute_by_item(matrix, valid_data, item_k)
    print(f"Item-based Accuracy: {item_acc}")

    # Combine both with hybrid weighting
    hybrid_mat = alpha * user_mat + (1 - alpha) * item_mat
    hybrid_acc = sparse_matrix_evaluate(valid_data, hybrid_mat)
    print(f"Hybrid Accuracy: {hybrid_acc}")

    return hybrid_acc


def tune_user_k(sparse_matrix, val_data, student_similarity_matrix, question_similarity_matrix,
                user_k_values, best_item_k, best_alpha):
    """Return the best user k.
    """

    hybrid_accuracies = []
    hybrid_weighted_accuracies = []
    for user_k in user_k_values:
        print(f"Now running Hybrid KNN with user_k={user_k}:")
        hybrid_accuracy = hybrid_knn_impute(sparse_matrix, val_data, user_k, best_item_k, best_alpha)
        hybrid_accuracies.append(hybrid_accuracy)
        print(f"Complete. Now running Hybrid Weighted KNN with user_k={user_k}:")
        hybrid_weighted_accuracy = hybrid_weighted_knn_impute(sparse_matrix, val_data, student_similarity_matrix,
                                                              question_similarity_matrix,
                                                              user_k, best_item_k, best_alpha)
        hybrid_weighted_accuracies.append(hybrid_weighted_accuracy)
    best_user_k = user_k_values[np.argmax(hybrid_weighted_accuracies)]
    print(f"Best user_k: {best_user_k}")
    return best_user_k


def tune_item_k(sparse_matrix, val_data, student_similarity_matrix, question_similarity_matrix,
                best_user_k, item_k_values, best_alpha):
    """Return the best item k.
    """

    hybrid_accuracies = []
    hybrid_weighted_accuracies = []
    for item_k in item_k_values:
        print(f"Now running Hybrid KNN with item_k={item_k}:")
        hybrid_accuracy = hybrid_knn_impute(sparse_matrix, val_data, best_user_k, item_k, best_alpha)
        hybrid_accuracies.append(hybrid_accuracy)
        print(f"Complete. Now running Hybrid Weighted KNN with item_k={item_k}:")
        hybrid_weighted_accuracy = hybrid_weighted_knn_impute(sparse_matrix, val_data, student_similarity_matrix,
                                                              question_similarity_matrix,
                                                              best_user_k, item_k, best_alpha)
        hybrid_weighted_accuracies.append(hybrid_weighted_accuracy)
    best_item_k = item_k_values[np.argmax(hybrid_weighted_accuracies)]
    print(f"Best item_k: {best_item_k}")
    return best_item_k


def tune_alpha(sparse_matrix, val_data, student_similarity_matrix, question_similarity_matrix,
               best_user_k, best_item_k, alpha_values):
    """Return the best alpha.
    """

    hybrid_accuracies = []
    hybrid_weighted_accuracies = []
    for alpha in alpha_values:
        print(f"Now running Hybrid KNN with alpha={alpha}:")
        hybrid_accuracy = hybrid_knn_impute(sparse_matrix, val_data, best_user_k, best_item_k, alpha)
        hybrid_accuracies.append(hybrid_accuracy)
        print(f"Complete. Now running Hybrid Weighted KNN with alpha={alpha}:")
        hybrid_weighted_accuracy = hybrid_weighted_knn_impute(sparse_matrix, val_data, student_similarity_matrix,
                                                              question_similarity_matrix,
                                                              best_user_k, best_item_k, alpha)
        hybrid_weighted_accuracies.append(hybrid_weighted_accuracy)
    best_alpha = alpha_values[np.argmax(hybrid_weighted_accuracies)]
    print(f"Best alpha: {best_alpha}")

    # Plot the hybrid validation accuracy as a function of alpha
    # plt.plot(alpha_values, hybrid_accuracies, marker='o')
    # plt.xlabel("alpha")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Hybrid Validation Accuracy vs. alpha")
    # plt.savefig("knn-hybrid.png")

    return best_alpha


def main():
    start_time = time.time()  # Start the timer

    # Load all the data
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    question_meta = pd.read_csv("./data/question_meta.csv")
    student_meta = pd.read_csv("./data/student_meta.csv")

    # Verify data loading
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    print("Question metadata:")
    print(question_meta.head())
    print("Student metadata:")
    print(student_meta.head())

    # Compute similarity matrices
    student_similarity_matrix, question_similarity_matrix = compute_similarity_matrices(student_meta, question_meta)

    # Hyperparameters
    user_k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    best_user_k = 7
    item_k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    best_item_k = 21
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_alpha = 0.1

    # # user_k tuning
    # best_user_k = tune_user_k(sparse_matrix, val_data, student_similarity_matrix, question_similarity_matrix,
    #                           user_k_values, best_item_k, best_alpha)
    #
    # # item_k tuning
    # best_item_k = tune_item_k(sparse_matrix, val_data, student_similarity_matrix, question_similarity_matrix,
    #                           best_user_k, item_k_values, best_alpha)
    # # Alpha tuning
    # best_alpha = tune_alpha(sparse_matrix, val_data, student_similarity_matrix, question_similarity_matrix,
    #                         best_user_k, best_item_k, alpha_values)

    # Final evaluation on test set
    print("Hyperparameters:")
    print(f"Best user_k: {best_user_k}")
    print(f"Best item_k: {best_item_k}")
    print(f"Best alpha: {best_alpha}")
    final_hybrid_test_accuracy = hybrid_knn_impute(sparse_matrix, test_data, best_user_k, best_item_k, best_alpha)
    final_weighted_test_accuracy = hybrid_weighted_knn_impute(sparse_matrix, test_data, student_similarity_matrix,
                                                              question_similarity_matrix,
                                                              best_user_k, best_item_k, best_alpha)
    print(f"FINAL HYBRID TEST ACCURACY: {final_hybrid_test_accuracy}")
    print(f"FINAL HYBRID WEIGHTED TEST ACCURACY: {final_weighted_test_accuracy}")

    """
    PAST RESULTS:
    Best user_k: 7
    Best item_k: 21
    Best alpha: 0.1
    FINAL HYBRID TEST ACCURACY: 0.6892464013547841
    FINAL HYBRID WEIGHTED TEST ACCURACY: 0.6788032740615297
    """

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total time taken: {int(minutes)} minutes {int(seconds)} seconds")


if __name__ == "__main__":
    main()
