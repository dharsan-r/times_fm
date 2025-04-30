import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import base_model
from praxis import optimizers
from praxis import schedules
from praxis import base_hyperparams
from praxis import base_layer
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import checkpoints
from paxml import learners
from paxml import partitioning
from paxml import checkpoint_types
import load_data
from sklearn.metrics import mean_squared_error


import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

import timesfm
import gc
import numpy as np
import pandas as pd
from timesfm import patched_decoder
from timesfm import data_loader

from tqdm import tqdm
import dataclasses
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

combined_Vabs_df, combined_Pabs_df, control_Pabs_data_df, control_Vabs_data_df, stroke_Pabs_data_df, stroke_Vabs_data_df, combined_headers = load_data.return_data(576)

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor
NpTensor = pytypes.NpTensor
WeightedScalars = pytypes.WeightedScalars
instantiate = base_hyperparams.instantiate
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
AuxLossStruct = base_layer.AuxLossStruct

AUX_LOSS = base_layer.AUX_LOSS
template_field = base_layer.template_field

# Standard prng key names
PARAMS = base_layer.PARAMS
RANDOM = base_layer.RANDOM

key = jax.random.PRNGKey(seed=1234)

timesfm_backend = "gpu"  # @param

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

def get_RMSE(forecast_df, test_gt):

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


tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend=timesfm_backend,
          per_core_batch_size=32,
          horizon_len=64,
          num_layers=20,
          # Se this to True for v1.0 checkpoints
          use_positional_embedding=False,
          # Note that we could set this to as high as 2048 but keeping it 512 here so that
          # both v1.0 and 2.0 checkpoints work
          context_len=512,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
  )

# The following code bellow is for zero-shot predictions

print("\ncontrol Vabs")
train_df, test_df, test_gt = split_data(control_Vabs_data_df)
num_subjects  = test_df.shape[1]
augmented_df = morph_data(test_df)

forecast_df = tfm.forecast_on_df(
    inputs=augmented_df,
    freq="m",  # monthly
    value_name="value",
    num_jobs=-1,
)

get_RMSE(forecast_df, test_gt)


model = pax_fiddle.Config(
    patched_decoder.PatchedDecoderFinetuneModel,
    name='patched_decoder_finetune',
    core_layer_tpl=tfm.model_p,
)




@pax_fiddle.auto_config
def build_learner() -> learners.Learner:
  return pax_fiddle.Config(
      learners.Learner,
      name='learner',
      loss_name='avg_qloss',
      optimizer=optimizers.Adam(
          epsilon=1e-7,
          clip_threshold=1e2,
          learning_rate=1e-2,
          lr_schedule=pax_fiddle.Config(
              schedules.Cosine,
              initial_value=1e-3,
              final_value=1e-4,
              total_steps=40000,
          ),
          ema_decay=0.9999,
      ),
      # Linear probing i.e we hold the transformer layers fixed.
      bprop_variable_exclusion=['.*/stacked_transformer_layer/.*'],
  )



task_p = tasks_lib.SingleTask(
    name='ts-learn',
    model=model,
    train=tasks_lib.SingleTask.Train(
        learner=build_learner(),
    ),
)



task_p.model.ici_mesh_shape = [1, 1, 1]
task_p.model.mesh_axis_names = ['replica', 'data', 'mdl']

DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
MESH = jax.sharding.Mesh(DEVICES, ['replica', 'data', 'mdl'])

num_devices = jax.local_device_count()
print(f'num_devices: {num_devices}')
print(f'device kind: {jax.local_devices()[0].device_kind}')


jax_task = task_p
key, init_key = jax.random.split(key)
