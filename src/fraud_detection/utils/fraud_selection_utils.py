import numpy as np
import pandas as pd
from tqdm import tqdm


def select_cases(
    data: pd.DataFrame,
    is_fraud: bool,
    n_sample: int = None,
    seed: int=20240128,
    add_label_col: bool=False
) -> pd.DataFrame:
    '''
    selects only frauds from the data

    Args:
        data (pd.DataFrame): data to select from
        is_fraud (bool): indicates whether we want to select frauds or not
        n_sample (int): number of samples to select, if not provied, all items will be returned
        seed (int): seed used to sample elements

    Returns:
        pd.DataFrame: frauds or not
    '''
    selected_ids = data[data['is_fraud'] == is_fraud]['cc_num'].unique()
    if n_sample:
        if n_sample < len(selected_ids):
            np.random.seed(seed)
            selected_ids = np.random.choice(selected_ids, n_sample, replace=False)
    
    return_data = data[data['cc_num'].isin(selected_ids)].sort_values('trans_date_trans_time')
    if add_label_col:
        return_data['label'] = is_fraud
    return return_data


def select_fraud_transactions(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    selects fraud windows from the data. the idea is for every fraud transaction
    select a list of preceding transactions to it

    Args:
        data (pd.DataFrame): data to select from
        window_size (int): size of the window

    Returns:
        pd.DataFrame: fraud windows
    """
    fraud_windows = []

    for name, group in tqdm(data.groupby('cc_num')):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            if row['is_fraud'] == 1:
                start = max(0, idx-window_size-1)
                selected_window = group.loc[start:idx].copy()
                selected_window['window_id'] = f'{name}_{idx}'
                fraud_windows.append(selected_window)

    fraud_windows = pd.concat(fraud_windows)
    return fraud_windows


def select_non_fraud_transactions(data: pd.DataFrame, window_size: int, max_samples: int) -> pd.DataFrame:
    """
    selects normal windows from the data

    Args:
        data (pd.DataFrame): data to select from
        window_size (int): size of the window
        max_samples (int): maximum number of samples to select

    Returns:
        pd.DataFrame: normal windows
    """
    normal_windows = []
    np.random.seed(0)  # Set the seed for reproducibility
    sample_indices = np.random.choice(data.index, size=min(max_samples, len(data)), replace=False)

    for idx in tqdm(sample_indices):
        name = data.loc[idx, 'cc_num']
        group = data[data['cc_num'] == name]
        group_idx = group.index.get_loc(idx)  # Get the index of the selected transaction in the group
        group = group.reset_index(drop=True)  # Reset the index after getting the group index
        start = max(0, group_idx - window_size)
        selected_window = group.loc[start:group_idx].copy()
        selected_window['window_id'] = f'{name}_{group_idx}'
        normal_windows.append(selected_window)

    normal_windows = pd.concat(normal_windows)
    return normal_windows