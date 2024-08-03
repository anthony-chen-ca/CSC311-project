from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for x, j in enumerate(data["question_id"]):
        i = data["user_id"][x]
        c = data["is_correct"][x]
        log_lklihood += c * (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    for x, j in enumerate(data["question_id"]):
        i = data["user_id"][x]
        c = data["is_correct"][x]
        p = sigmoid(theta[i] - beta[j])
        grad_theta[i] += c - p
        grad_beta[j] += p - c

    # Parameter update
    theta += lr * grad_theta
    beta -= lr * grad_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # Random values for theta and beta from 0 to 1
    np.random.seed(1)
    theta = np.random.uniform(0, 1, max(data["user_id"]) + 1)
    beta = np.random.uniform(0, 1, max(data["question_id"]) + 1)

    print(f"Initial Average Theta (how good the students are): {np.mean(theta)}")
    print(f"Initial Average Beta (how difficult the questions are): {np.mean(beta)}")

    # Validation accuracy, training log-likelihoods, validation log-likelihoods
    val_acc_lst = []
    train_ll = []
    val_ll = []

    best_val_acc = 0
    best_i = 0
    patience = 3

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_ll.append(neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_ll.append(val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("Iteration: {} \t NLLK: {} \t Score: {}".format(i + 1, neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

        # Early stopping
        if score > best_val_acc:
            best_val_acc = score
            best_i = i
        elif i - best_i >= patience:
            print(f"Early stopping at iteration {i + 1}")
            return theta, beta, val_acc_lst, train_ll, val_ll, i + 1

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_ll, val_ll, iterations


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Hyperparameters
    lr = 0.001
    iterations = 25

    # Train the model
    theta, beta, val_acc_lst, train_ll, val_ll, iterations = irt(train_data, val_data, lr, iterations)

    print(f"Final Average Theta (how good the students are): {np.mean(theta)}")
    print(f"Final Average Beta (how difficult the questions are): {np.mean(beta)}")

    # Plot the training and validation log-likelihoods
    plt.plot(list(range(1, iterations + 1)), train_ll, label='Training Log-Likelihood')
    plt.plot(list(range(1, iterations + 1)), val_ll, label='Validation Log-Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Training and Validation Log-Likelihood')
    plt.legend()
    plt.savefig("irt.png")

    final_val_score = evaluate(data=val_data, theta=theta, beta=beta)
    final_test_score = evaluate(data=test_data, theta=theta, beta=beta)
    print(f"Final Validation Score: {final_val_score}")
    print(f"Final Test Score: {final_test_score}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1, j2, j3 = 1, 2, 3

    avg_theta = np.mean(theta)
    theta_range = np.linspace(avg_theta - 0.5, avg_theta + 0.5, 100)

    prob_j1 = [sigmoid(theta - beta[j1]) for theta in theta_range]
    prob_j2 = [sigmoid(theta - beta[j2]) for theta in theta_range]
    prob_j3 = [sigmoid(theta - beta[j3]) for theta in theta_range]

    plt.figure()
    plt.plot(theta_range, prob_j1, label=f'Question {j1}')
    plt.plot(theta_range, prob_j2, label=f'Question {j2}')
    plt.plot(theta_range, prob_j3, label=f'Question {j3}')
    plt.xlabel('Theta (Ability)')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response vs. Theta for Selected Questions')
    plt.legend()
    plt.savefig("probability_vs_theta.png")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
