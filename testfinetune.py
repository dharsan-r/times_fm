

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




timesfm_backend = "gpu"  # @param

tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend=timesfm_backend,
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=20,
          # Se this to True for v1.0 checkpoints
          use_positional_embedding=False,
          # Note that we could set this to as high as 2048 but keeping it 512 here so that
          # both v1.0 and 2.0 checkpoints work
          context_len=512,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m"),
  )


DATA_DICT = {
    "ettm2": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "../datasets/ETT-small/ETTm2.csv",
        "freq": "15min",
    },
    "ettm1": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "../datasets/ETT-small/ETTm1.csv",
        "freq": "15min",
    },
    "etth2": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "../datasets/ETT-small/ETTh2.csv",
        "freq": "H",
    },
    "etth1": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "../datasets/ETT-small/ETTh1.csv",
        "freq": "H",
    },
    "elec": {
        "boundaries": [18413, 21044, 26304],
        "data_path": "../datasets/electricity/electricity.csv",
        "freq": "H",
    },
    "traffic": {
        "boundaries": [12280, 14036, 17544],
        "data_path": "../datasets/traffic/traffic.csv",
        "freq": "H",
    },
    "weather": {
        "boundaries": [36887, 42157, 52696],
        "data_path": "../datasets/weather/weather.csv",
        "freq": "10min",
    },
}


dataset = "ettm1"
data_path = DATA_DICT[dataset]["data_path"]
freq = DATA_DICT[dataset]["freq"]
int_freq = timesfm.freq_map(freq)
boundaries = DATA_DICT[dataset]["boundaries"]

data_df = pd.read_csv(open(data_path, "r"))
print()
print("yeahhhhh we're reaching here finally")
print()
ts_cols = [col for col in data_df.columns if col != "date"]
num_cov_cols = None
cat_cov_cols = None

context_len = 512
pred_len = 96

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