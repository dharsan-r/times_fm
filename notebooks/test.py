import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

import timesfm
import gc
import numpy as np
import pandas as pd
from timesfm import patched_decoder
from timesfm import data_loader
import load_data
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import dataclasses
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

combined_Vabs_df, combined_Pabs_df, control_Pabs_data_df, control_Vabs_data_df, stroke_Pabs_data_df, stroke_Vabs_data_df, combined_headers = load_data.return_data(576)


def split_data(df):
    total_columns = df.columns.tolist()
    num_columns = len(total_columns)

    # Calculate split point
    split_point = int(num_columns * 0.8)

    # Randomly shuffle the columns
    np.random.seed(42)  # for reproducibility
    shuffled_columns = np.random.permutation(total_columns)

    # Split columns into train and test sets
    train_columns = shuffled_columns[:split_point]
    test_columns = shuffled_columns[split_point:]

    # Create train and test DataFrames
    train_df = df[train_columns]
    test_df = df[test_columns]

    test_gt = test_df.tail(64)

    # Remove the last 64 rows from the original DataFrame
    test_df = test_df.iloc[:-64]

    return train_df, test_df, test_gt

def morph_data(test_df):


    # Assuming combined_Vabs_df exists and only value columns are to be used
    # Step 1: Combine all values from the DataFrame into a single column
    flattened_values = test_df.values.flatten()

    # Step 2: Create the corresponding unique_id for every 576 rows
    num_rows = len(flattened_values)
    num_groups = int(np.ceil(num_rows / 512))

    unique_ids = [f'T{i+1}' for i in range(num_groups) for _ in range(512)]
    unique_ids = unique_ids[:num_rows]  # Trim to match number of values

    # Step 3: Create the new DataFrame
    augmented_df = pd.DataFrame({
        'unique_id': unique_ids,
        'value': flattened_values
    })

    # Assuming combined_Vabs_df already exists
    start_time = pd.to_datetime("2016-07-01 00:00:00")
    num_rows = len(augmented_df)

    # Create the date range
    date_range = pd.date_range(start=start_time, periods=num_rows, freq='S')

    # Insert the date column at the beginning
    augmented_df.insert(0, 'ds', date_range)

    return augmented_df

print("\ncontrol Vabs")
train_df, test_df, test_gt = split_data(control_Vabs_data_df)
num_subjects  = test_df.shape[1]
augmented_df = morph_data(test_df)


DATA_DICT = {
    "ettm1": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "../datasets/ETT-small/ETTm1.csv",
        "freq": "15min",
    },
    "test": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "../datasets/ETT-small/ETTm1.csv",
        "freq": "1s",
    },
    
}


dataset = "ettm1"
data_path = DATA_DICT[dataset]["data_path"]
freq = DATA_DICT[dataset]["freq"]
int_freq = timesfm.freq_map(freq)
boundaries = DATA_DICT[dataset]["boundaries"]

data_df = pd.read_csv(open(data_path, "r"))

data_df2 = augmented_df


print(data_df)


ts_cols = [col for col in data_df.columns if col != "date"]
num_cov_cols = None
cat_cov_cols = None

context_len = 512
pred_len = 64

num_ts = len(ts_cols)
batch_size = 8

dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col="date",
      num_cov_cols=num_cov_cols,
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundaries[0]],
      val_range=[boundaries[0], boundaries[1]],
      test_range=[boundaries[1], boundaries[2]],
      hist_len=context_len,
      pred_len=pred_len,
      batch_size=num_ts,
      freq=freq,
      normalize=True,
      epoch_len=None,
      holiday=False,
      permute=True,
  )

train_batches = dtl.tf_dataset(mode="train", shift=1).batch(batch_size)
val_batches = dtl.tf_dataset(mode="val", shift=pred_len)
test_batches = dtl.tf_dataset(mode="test", shift=pred_len)

print(train_batches)
