import pandas as pd

import fraud_detection.utils.fraud_selection_utils as fsu


def prepare_train_data(raw_data: pd.DataFrame, train_set_params: dict) -> pd.DataFrame:
    """
    function prepares training data by selecting true and false windows for different
    clients in the population:
        for fraudulent transaction it will select n transactions before the transaction
        for non-fradulent clients it will select random transactions

    Args:
        raw_data (pd.DataFrame): train data used to build the training dataset
        sample_params (dict): dictionary with sampling parameters

    Returns:
        a dataframe with sampled transactions
    """

    # unpack parameters
    seed = train_set_params.get("seed")
    n_true_samples = train_set_params.get("n_true")
    n_false_samples = train_set_params.get("n_false")
    window_size = train_set_params.get("window_size")
    max_samples = train_set_params.get("max_normal_samples")

    id_cols = train_set_params.get('columns', {}).get('id_columns')
    feature_cols = train_set_params.get('columns', {}).get('feature_columns')
    final_cols = [*id_cols, *['label'], *feature_cols]

    # select transactions
    fraud_clients = fsu.select_cases(raw_data, True, n_true_samples, seed, True)
    normal_clients = fsu.select_cases(raw_data, False, n_false_samples, seed, True)

    fraud_transactions = fsu.select_fraud_transactions(fraud_clients, window_size)
    normal_transactions = fsu.select_non_fraud_transactions(normal_clients, window_size, max_samples)

    # combine true and false label data
    return_data = pd.concat(
        [
            fraud_transactions[[*final_cols]],
            normal_transactions[[*final_cols]],
        ]
    )

    return return_data