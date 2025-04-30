import timesfm
import pandas as pd
import numpy as np
import load_data
from sklearn.metrics import mean_squared_error

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



# Loading the timesfm-2.0 checkpoint:
# For PAX
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=8,
          horizon_len=64,
          num_layers=10,
          context_len=512,

          use_positional_embedding=False,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
  )


print("\ncontrol Vabs")
train_df, test_df, test_gt = split_data(control_Vabs_data_df)
num_subjects  = test_df.shape[1]

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

augmented_df = morph_data(test_df)

forecast_df = tfm.forecast_on_df(
    inputs=augmented_df,
    freq="m",  # monthly
    value_name="value",
    num_jobs=-1,
)


# Initialize lists to store individual store RMSEs
store_rmses = []

# Reset t to 1 before starting
t = 1

# Iterate through columns in test_gt
for column_name in test_gt.columns:
    # Filter forecast data for the current unique_id
    unique_id = "T" + str(t)
    filtered_df = forecast_df[forecast_df['unique_id'] == unique_id]
    
    # Get ground truth data and forecast values
    gt_data = test_gt[column_name].values  # Convert to numpy array
    values = filtered_df['timesfm'].values  # Convert to numpy array
    
    # Compute RMSE for this store
    rmse = np.sqrt(mean_squared_error(gt_data, values))
    
    # Store the RMSE
    store_rmses.append(rmse)
    
    # Increment t for next iteration
    t += 1

# Compute average RMSE across all stores
avg_rmse = np.mean(store_rmses)

# Calculate the standard deviation of RMSEs
std_dev_rmses = np.std(store_rmses)


print(f"Average RMSE across all stores: {avg_rmse}")
print(f"\nSTD Dev RMSE across all stores: {std_dev_rmses}")

