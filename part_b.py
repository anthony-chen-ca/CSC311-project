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

    # Combine all processed features into a single DataFrame
    student_features = pd.concat([gender, student_meta['age'], student_meta['premium_pupil']], axis=1)
    student_features.index = student_meta['user_id']

    return student_features


def process_question_metadata(question_meta):
    """Process question metadata and return a DataFrame with numerical features.
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
    """Weight function that takes an array of distances and returns an array containing the weights.
    Each entry in weight array corresponds to the metadata similarity between a point and its neighbor.
    """
    weights = np.zeros_like(distances)
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors):
            weights[i, j] = similarity_matrix[i, neighbor]
    return weights


def preprocess_matrix(matrix):
    """Gets rid of NaN values using a very basic KNNImputer, so we can use NearestNeighbors.
    """
    imputer = KNNImputer(n_neighbors=5)
    return imputer.fit_transform(matrix)


def weighted_knn_impute(matrix, similarity_matrix, valid_data, k, transposed=False):
    """Weighted KNN with imputation. The k nearest neighbors are chosen based on similar answers, and then the
    chosen neighbors are given weights based on metadata. Finally, performs imputation using those adjusted weights.

    Returns imputed matrix and accuracy.
    """
    if transposed:
        matrix = matrix.T
    matrix = preprocess_matrix(matrix)
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(matrix)
    # Get the distances and indices of the k nearest neighbors
    distances, indices = nbrs.kneighbors(matrix, return_distance=True)

    # Adjust weights using metadata similarity
    weights = weight_function(distances, similarity_matrix, indices)

    # Perform KNN Imputation with custom weights
    weighted_nbrs = KNNImputer(n_neighbors=k, weights=lambda d: weights)
    imputed_matrix = weighted_nbrs.fit_transform(matrix)
    if transposed:
        imputed_matrix = imputed_matrix.T
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    return imputed_matrix, acc


def knn_impute_hybrid(matrix, valid_data, student_similarity_matrix, question_similarity_matrix, user_k, item_k, alpha):
    """Hybrid weighted KNN imputation using both user and item. Returns the hybrid accuracy."""
    # Apply user-based KNN
    user_mat, user_acc = weighted_knn_impute(matrix, student_similarity_matrix, valid_data, user_k)
    print(f"User-based Accuracy: {user_acc}")

    # Apply item-based KNN
    item_mat, item_acc = weighted_knn_impute(matrix, question_similarity_matrix, valid_data, item_k, transposed=True)
    print(f"Item-based Accuracy: {item_acc}")

    # Combine both with weighting
    hybrid_mat = alpha * user_mat + (1 - alpha) * item_mat
    hybrid_acc = sparse_matrix_evaluate(valid_data, hybrid_mat)
    print(f"Hybrid Accuracy: {hybrid_acc}")

    return hybrid_acc


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
    user_k = 11
    item_k = 21
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # best alpha is 0.7 with 0.68 acc on test data set

    hybrid_accuracies = []
    for alpha in alpha_values:
        print(f"Now running Hybrid KNN with alpha={alpha}:")
        hybrid_accuracy = knn_impute_hybrid(sparse_matrix, val_data, student_similarity_matrix,
                                            question_similarity_matrix, user_k, item_k, alpha)
        hybrid_accuracies.append(hybrid_accuracy)

    best_alpha = alpha_values[np.argmax(hybrid_accuracies)]
    print(f"Best alpha: {best_alpha}")

    # Evaluate on the test set using the best alpha
    test_hybrid_accuracy = knn_impute_hybrid(sparse_matrix, test_data, student_similarity_matrix,
                                             question_similarity_matrix, user_k, item_k, best_alpha)
    print(f"FINAL TEST ACCURACY for best alpha={best_alpha}: {test_hybrid_accuracy}")

    # Plot the hybrid validation accuracy as a function of alpha
    # plt.plot(alpha_values, hybrid_accuracies, marker='o')
    # plt.xlabel("alpha")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Hybrid Validation Accuracy vs. alpha")
    # plt.savefig("knn-hybrid.png")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Total time taken: {int(minutes)} minutes {int(seconds)} seconds")


if __name__ == "__main__":
    main()
