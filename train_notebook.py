import os
import gc
from glob import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import exploratory_data_analysis as data_nb
from exploratory_data_analysis import Aggregator

data_nb.Aggregator.group_aggregators = [pl.n_unique] #pl.max, pl.min, 
# data_nb.Aggregator.str_aggregators = [pl.max, pl.min, pl.first, pl.last, pl.n_unique]
# train_df = data_nb.prepare_df(data_nb.CFG.train_dir)
base_files = [
    "_static_cb_0.parquet",
    "_static_0_*.parquet",
    "_applprev_1_*.parquet",
    "_tax_registry_a_1.parquet",
    "_tax_registry_b_1.parquet",
    "_tax_registry_c_1.parquet",
    "_other_1.parquet",
    "_person_1.parquet",
    "_deposit_1.parquet",
    "_debitcard_1.parquet",
    "_credit_bureau_b_1.parquet",
    "_credit_bureau_b_2.parquet",
]
base_agg = Aggregator(
    num_aggregators = [pl.max, pl.min, pl.first, pl.last, pl.mean],
    str_aggregators = [pl.max, pl.min, pl.first, pl.last],
    group_aggregators = [pl.max, pl.min, pl.first, pl.last],
    str_mode = True
)
breakpoint()
train_df = data_nb.prepare_df(base_files, data_nb.CFG.train_dir, base_agg)

# train_df = pd.read_parquet('/kaggle/usr/lib/home_credit_baseline_data/train_full.parquet')
cat_cols = list(train_df.select_dtypes("category").columns)
print(train_df.describe())