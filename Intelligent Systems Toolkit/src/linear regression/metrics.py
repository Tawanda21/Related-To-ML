import numpy as np


def _to_native_scalar(value):
    """Convert NumPy scalar values to native Python scalars for clean output."""
    return value.item() if isinstance(value, np.generic) else value

def mean_squared_error(y_true, y_pred):
    """MSE = (1/n) * Σ(y_true - y_pred)^2
    OR
    MSE = Average of the squared differences between true and predicted values.
    Lower is actually better. Sensitive to outliers.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """MAE = (1/n) * Σ|y_true - y_pred|
    OR
    MAE = Average of the absolute differences between true and predicted values.
    Lower is better. Less sensitive to outliers than MSE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """RMSE = sqrt(MSE)
    OR
    RMSE = Square root of the average of the squared differences between true and predicted values.
    Lower is better. Same units as the target variable.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    """R^2 = 1 - (SS_residual / SS_total)
    OR
    R^2 = Proportion of variance in the dependent variable that is predictable from the independent variable(s).
    Higher is better. Ranges from 0 to 1 (or negative if the model is worse than a horizontal line).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    OR
    MAPE = Average of the absolute percentage differences between true and predicted values.
    Lower is better. Can be problematic if y_true contains zeros.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.inf  # If all true values are zero, MAPE is undefined

    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def accuracy_score(y_true, y_pred):
    """Accuracy = (TP + TN) / (TP + TN + FP + FN) or correct predictions / total predictions
    OR
    Accuracy = Proportion of correct predictions (both true positives and true negatives) among the total number of cases.
    Higher is better. Can be misleading in imbalanced datasets.
    Best for balanced datasets.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)

    return correct_predictions / total_predictions if total_predictions > 0 else 0

def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion Matrix = [[TP, FP], [FN, TN]]
    OR
    Confusion Matrix = A table used to evaluate the performance of a classification model by comparing true labels with predicted labels.
    Provides counts of true positives, false positives, true negatives, and false negatives.
    Useful for understanding the types of errors made by the model.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    label_to_index = {label: index for index, label in enumerate(labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_index and pred_label in label_to_index:
            matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

    return matrix

def precision_score(y_true, y_pred):
    """Precision = TP / (TP + FP)
    OR
    Precision = Proportion of true positive predictions among all positive predictions.
    Higher is better. Focuses on the accuracy of positive predictions.
    Best for scenarios where false positives are costly.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    """Recall = TP / (TP + FN)
    OR
    Recall = Proportion of true positive predictions among all actual positives.
    Higher is better. Focuses on the ability to find all positive cases.
    Best for scenarios where false negatives are costly.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    OR
    F1 Score = Harmonic mean of precision and recall.
    Higher is better. Balances precision and recall.
    Best for scenarios with imbalanced classes.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def specificity_score(y_true, y_pred):
    """Specificity = TN / (TN + FP)
    OR
    Specificity = Proportion of true negative predictions among all actual negatives.
    Higher is better. Focuses on the ability to identify negative cases.
    Best for scenarios where false positives are costly.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tn / (tn + fp) if (tn + fp) > 0 else 0

def classification_report(y_true, y_pred):
    """Classification Report = A summary of precision, recall, F1 score, and support for each class.
    OR
    Classification Report = A detailed report that provides precision, recall, F1 score, and support for each class in a classification problem.
    Higher is better for precision, recall, and F1 score. Support indicates the number of true instances for each class.
    Useful for evaluating the performance of classification models, especially in imbalanced datasets.
    """
    return {
        'accuracy': _to_native_scalar(accuracy_score(y_true, y_pred)),
        'precision': _to_native_scalar(precision_score(y_true, y_pred)),
        'recall': _to_native_scalar(recall_score(y_true, y_pred)),
        'f1_score': _to_native_scalar(f1_score(y_true, y_pred)),
        'specificity': _to_native_scalar(specificity_score(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def evaluate_regression(y_true, y_pred, model_name="Regression Model"):
    """Evaluate Regression = A dictionary containing MSE, MAE, RMSE, R^2, and MAPE.
    OR
    Evaluate Regression = A comprehensive evaluation of regression model performance using multiple metrics.
    Lower is better for MSE, MAE, RMSE, and MAPE. Higher is better for R^2.
    Provides a holistic view of the model's performance across different aspects.
    """
    return {
        'mean_squared_error': _to_native_scalar(mean_squared_error(y_true, y_pred)),
        'mean_absolute_error': _to_native_scalar(mean_absolute_error(y_true, y_pred)),
        'root_mean_squared_error': _to_native_scalar(root_mean_squared_error(y_true, y_pred)),
        'r2_score': _to_native_scalar(r2_score(y_true, y_pred)),
        'mean_absolute_percentage_error': _to_native_scalar(mean_absolute_percentage_error(y_true, y_pred))
    }

def evaluate_classification(y_true, y_pred, model_name="Classification Model"):
    """Evaluate Classification = A dictionary containing accuracy, precision, recall, F1 score, specificity, and confusion matrix.
    OR
    Evaluate Classification = A comprehensive evaluation of classification model performance using multiple metrics.
    Higher is better for accuracy, precision, recall, F1 score, and specificity. The confusion matrix provides insight into the types of errors made by the model.
    Provides a holistic view of the model's performance across different aspects.
    """
    return classification_report(y_true, y_pred)


def print_evaluation_list(title, evaluation):
    """Print evaluation metrics as a clean, list-style console output."""
    print(f"{title}:")
    for metric, value in evaluation.items():
        if metric == 'confusion_matrix' and isinstance(value, list):
            print(f"- {metric}:")
            for row in value:
                print(f"  - {row}")
        else:
            print(f"- {metric}: {value}")

# Example usage for regression evaluation
y_true_reg = [3, -0.5, 2, 7]
y_pred_reg = [2.5, 0.0, 2, 8]
regression_evaluation = evaluate_regression(y_true_reg, y_pred_reg)
print_evaluation_list("Regression Evaluation", regression_evaluation)
print()

# Example usage for classification evaluation
y_true_clf = [0, 1, 1, 0, 1]
y_pred_clf = [0, 1, 0, 0, 1]
classification_evaluation = evaluate_classification(y_true_clf, y_pred_clf)
print_evaluation_list("Classification Evaluation", classification_evaluation)
