from datetime import datetime
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


def enrich_matrix(matrix, question_meta, student_meta):
    """Enrich feature matrix with metadata.
    """
    # Process metadata
    student_weights = process_student_metadata(student_meta)
    question_weights = process_question_metadata(question_meta)

    # Initialize enriched matrix as a copy of the original matrix
    enriched_matrix = matrix.copy()

    num_students, num_questions = matrix.shape

    for student_idx in range(num_students):
        for question_idx in range(num_questions):
            student_id = student_idx
            question_id = question_idx

            # Skip NaN values in the original matrix
            if np.isnan(matrix[student_idx, question_idx]):
                continue

            # Get metadata values
            student_meta_row = student_weights.loc[student_id].values if student_id in student_weights.index else None
            question_meta_row = question_weights.loc[question_id].values if question_id in question_weights.index else None

            # If both student and question metadata are missing, we will skip
            if student_meta_row is None and question_meta_row is None:
                continue

            # Calculate student metadata weights
            student_weight = 0
            if student_meta_row is not None:
                valid_student_features = [feature for feature in student_meta_row if feature != -1]
                num_valid_student_features = len(valid_student_features)
                if num_valid_student_features > 0:
                    student_weight = sum(valid_student_features) / num_valid_student_features

            # Calculate question metadata weights
            question_weight = 0
            if question_meta_row is not None:
                num_question_features = len(question_meta_row)
                question_weight = sum(question_meta_row) / num_question_features

            # Combine weights from student and question metadata
            combined_weight = (student_weight * 0.5) + (question_weight * 0.5)

            # Adjust matrix entry
            enriched_matrix[student_idx, question_idx] *= (1 + combined_weight / 2)

    return enriched_matrix


def sample_data(matrix, sample_size, random_state):
    """Randomly sample student-question pairs from the matrix.
    """
    np.random.seed(random_state)
    total_pairs = matrix.shape[0]

    # Generate random indices
    sampled_indices = np.random.choice(total_pairs, size=sample_size, replace=False)

    # Sample the matrix
    sampled_matrix = matrix[sampled_indices]

    return sampled_matrix


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
    print("Running knn (impute by user)...")
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
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
    print("Running knn (impute by item)...")
    transposed_matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = (nbrs.fit_transform(transposed_matrix)).T  # Fit transform and then transpose back to original
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return mat, acc


def knn_impute_hybrid(matrix, valid_data, user_k, item_k, alpha):
    """Hybrid KNN imputation using both user and item."""
    # def compute_similarity_with_metadata(mat, meta):
    #     # Compute similarity matrix incorporating metadata
    #     pass  # Implement similarity computation with metadata

    # Apply user-based KNN
    user_mat, user_acc = knn_impute_by_user(matrix, valid_data, user_k)

    # Apply item-based KNN
    item_mat, item_acc = knn_impute_by_item(matrix, valid_data, item_k)

    # Combine both with weighting
    hybrid_mat = alpha * user_mat + (1 - alpha) * item_mat
    hybrid_acc = sparse_matrix_evaluate(valid_data, hybrid_mat)
    print(f"Hybrid Validation Accuracy: {hybrid_acc}")

    return hybrid_acc


def main():
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

    # Hyperparameters
    user_k = 11  # Will need to see if other k values are better, given a new algorithm
    item_k = 21  # Will need to see if other k values are better, given a new algorithm
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Best alpha is 0.1 given the current settings
    # The other major hyperparameter tuning comes from the weight calculation in enrich_matrix
    # If you have a better idea on calculating weights, feel free to make changes

    # Enrich features using metadata
    enriched_sparse_matrix = enrich_matrix(sparse_matrix, question_meta, student_meta)
    print("Enriched sparse matrix:")
    print(enriched_sparse_matrix)
    print("Shape of enriched sparse matrix:")
    print(enriched_sparse_matrix.shape)

    # # Uncomment for sampling (probably does not work)
    # sample_size = 50000  # Original dataset contains 542 students and 1774 questions (961,508 student-question pairs)
    # random_state = 123
    # sampled_enriched_matrix = sample_data(enriched_sparse_matrix, sample_size, random_state)
    # print("Sample enriched sparse matrix:")
    # print(sampled_enriched_matrix)
    # print("Sampled enriched matrix shape:")
    # print(sampled_enriched_matrix.shape)

    hybrid_accuracies = []
    for alpha in alpha_values:
        print(f"Running Hybrid KNN with alpha={alpha}:")
        hybrid_accuracy = knn_impute_hybrid(enriched_sparse_matrix, val_data, user_k, item_k, alpha)
        hybrid_accuracies.append(hybrid_accuracy)

    best_alpha = alpha_values[np.argmax(hybrid_accuracies)]
    print(f"Best alpha: {best_alpha}")

    # Evaluate on the test set using the best alpha
    test_hybrid_accuracy = knn_impute_hybrid(enriched_sparse_matrix, test_data, user_k, item_k, best_alpha)
    print(f"Test Hybrid Accuracy for best alpha={best_alpha}: {test_hybrid_accuracy}")

    # Plot the hybrid validation accuracy as a function of alpha
    # plt.plot(alpha_values, hybrid_accuracies, marker='o')
    # plt.xlabel("alpha")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Hybrid Validation Accuracy vs. alpha")
    # plt.savefig("knn-hybrid.png")


if __name__ == "__main__":
    main()
