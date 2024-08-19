from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from fraud_detection.utils.concatenate_transformer import ConcatenateTransformer


def combine_features(features: list, window_ids: list) -> list:
    """
    Combines features belonging to the same window ID into one array.

    Args:
        X: List of feature arrays.
        window_ids: List of window IDs corresponding to each feature array.

    Returns:
        A list of combined feature arrays.
    """

    combined_features = []
    for window_id in set(window_ids):
        window_features = [x for x, id in zip(features, window_ids) if id == window_id]
        combined_features.append(ConcatenateTransformer().transform(window_features))

    return combined_features


def select_single_label(labels: list, window_ids: list) -> list:
    """
    Selects a single label per window.

    Args:
        y: List of labels.
        window_ids: List of window IDs corresponding to each label.

    Returns:
        A list of selected labels.
    """

    selected_labels = []
    for window_id in set(window_ids):
        window_labels = [label for label, id in zip(labels, window_ids) if id == window_id]
        selected_labels.append(window_labels[0])

    return selected_labels