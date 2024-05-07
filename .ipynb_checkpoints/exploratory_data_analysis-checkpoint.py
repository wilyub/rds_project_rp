import os
import gc
from glob import glob
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)

class CFG:
    root_dir = Path("../data/")
    train_dir = Path("../data/parquet_files/train")
    test_dir = Path("../data/parquet_files/test")

# if __name__ == '__main__':
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'max_colwidth', 400): 
#         enhanced_feat_def_df = pd.read_parquet("/kaggle/input/home-credit-enhanced-feature-definitions/feature_definitions_dtypes_tables.parquet")
#         display(enhanced_feat_def_df)
def reduce_mem_usage(df, float16_as32=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)                    
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))            

        return df
    
    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                
        df = df.drop("date_decision", "MONTH")

        return df
    
    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df

class Aggregator:
    def __init__(
        self, 
        num_aggregators=[pl.max, pl.min, pl.first, pl.last, pl.mean], 
        str_aggregators=[pl.max, pl.min, pl.first, pl.last],  # n_unique
        group_aggregators=[pl.max, pl.min, pl.first, pl.last],
        str_mode=True
    ):
        self.num_aggregators = num_aggregators
        self.str_aggregators = str_aggregators
        self.group_aggregators = group_aggregators
        self.str_mode=str_mode
    
    def num_expr(self, df_cols):
        cols = [col for col in df_cols if col[-1] in ("P", "A")]
        expr_all = []
        for method in self.num_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]
            expr_all += expr

        return expr_all

    def date_expr(self, df_cols):
        cols = [col for col in df_cols if col[-1] in ("D",)]
        expr_all = []
        for method in self.num_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr

        return expr_all

    def str_expr(self, df_cols):
        cols = [col for col in df_cols if col[-1] in ("M",)]
        
        expr_all = []
        for method in self.str_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr
            
        if self.str_mode:
            expr_mode = [
                pl.col(col)
                .drop_nulls()
                .mode()
                .first()
                .alias(f"mode_{col}")
                for col in cols
            ]
        else:
            expr_mode = []

        return expr_all + expr_mode

    def other_expr(self, df_cols):
        cols = [col for col in df_cols if col[-1] in ("T", "L")]
        
        expr_all = []
        for method in self.str_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr

        return expr_all
    
    def count_expr(self, df_cols):
        cols = [col for col in df_cols if "num_group" in col]

        expr_all = []
        for method in self.group_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr

        return expr_all

    def get_exprs(self, df_cols):
        exprs = (
            self.num_expr(df_cols) + 
            self.date_expr(df_cols) + 
            self.str_expr(df_cols) + 
            self.other_expr(df_cols) + 
            self.count_expr(df_cols)
        )

        return exprs

def read_files_by_path(pattern_path, aggregator, depth=None, agg_chunks=False, num_group1_filter=None):
    chunks = []
    for i, path in enumerate(glob(str(pattern_path))):
        print("  chunk: ", i)
        chunk = pl.read_parquet(path).pipe(Pipeline.set_table_dtypes)
        if agg_chunks:
            if num_group1_filter != None:
                chunk = chunk.filter((pl.col("num_group1") == num_group1_filter)).drop(columns=["num_group1"])
            chunk = chunk.group_by("case_id").agg(aggregator.get_exprs(chunk.columns))
        chunks.append(chunk)
    df = pl.concat(chunks, how="vertical_relaxed")
    
    if depth in [1, 2]:
        print(f"  agg, depth {depth}")
        if num_group1_filter != None:
            df = df.filter((pl.col("num_group1") == num_group1_filter)).drop(columns=["num_group1"])
        df = df.group_by("case_id").agg(aggregator.get_exprs(df.columns))
    return df

def read_files(files_arr, data_dir, aggregator, mode="train", agg_chunks=False, num_group1_filter=None):
    base_file_name = f"{mode}_base.parquet"
    print("  files: ", base_file_name)
    feats_df = read_files_by_path(
        data_dir / base_file_name, data_dir, mode
    )
    
    for i, file_name in enumerate(files_arr):
        depth = re.findall("\w_(\d)", file_name)
        if len(depth) > 0:
            depth = depth[0]
        else:
            continue
        print("  files: ", file_name, f"(depth: {depth})")
        files_df = read_files_by_path(
            data_dir / f"{mode}{file_name}", 
            aggregator, int(depth), agg_chunks, 
            num_group1_filter=num_group1_filter
        )
        feats_df = feats_df.join(
            files_df, 
            how="left", on="case_id", suffix=f"_{depth}_{i}"
        )
        del files_df
        gc.collect()

    return feats_df

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data

def prepare_df(
    files_arr, data_dir, aggregator, 
    mode="train", cat_cols=None, train_cols=[], 
    agg_chunks=False, feat_eng=True, num_group1_filter=None
):
    print()
    print("Collecting data...")
    feats_df = read_files(files_arr, data_dir, aggregator, mode=mode, agg_chunks=agg_chunks)
    print("  feats_df shape:\t", feats_df.shape)
    
    if feat_eng:
        print("Feature Engineering...")
        feats_df = feats_df.with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
        neworder = feats_df.columns
        neworder.remove("month_decision")
        neworder.remove("weekday_decision")
        neworder.insert(5, "month_decision")
        neworder.insert(6, "weekday_decision")
        feats_df = feats_df.select(neworder)
    
    feats_df = feats_df.pipe(Pipeline.handle_dates)
#     print("  feats_df shape:\t", feats_df.shape)
    
    print("Filter cols...")
    if mode == "train":
        feats_df = feats_df.pipe(Pipeline.filter_cols)
    else:
        train_cols = feats_df.columns if len(train_cols) == 0 else train_cols
        feats_df = feats_df.select([col for col in train_cols if col != "target"])
    print("  feats_df shape:\t", feats_df.shape)
    
    print("Convert to pandas...")
    feats_df = to_pandas(feats_df, cat_cols)
    return feats_df

if __name__ == '__main__':

    credit_bureau_a_1_files = [
        "_credit_bureau_a_1_*.parquet",
    ]
    credit_b_a_1_agg = Aggregator(
        num_aggregators = [pl.max, pl.min, pl.first, pl.last, pl.mean],
        str_aggregators = [pl.max, pl.min, pl.first, pl.last], # n_unique
        group_aggregators = [pl.max, pl.min, pl.first, pl.last]
    )

    credit_bureau_a_1_train_df = prepare_df(
        credit_bureau_a_1_files, CFG.train_dir, credit_b_a_1_agg, feat_eng=False
    )
    gc.collect()
    cat_cols_credit_bureau_a_1 = list(credit_bureau_a_1_train_df.select_dtypes("category").columns)
    credit_bureau_a_1_train_df

    credit_bureau_a_2_files = [
        "_credit_bureau_a_2_*.parquet",
    ]
    credit_b_a_2_agg = Aggregator(
        num_aggregators = [pl.first],
        str_aggregators = [pl.first],
        group_aggregators = [pl.first],
        str_mode = False
    )
    credit_b_a_2_agg_2 = Aggregator(
        num_aggregators = [pl.max],
        str_aggregators = [pl.max],
        group_aggregators = [pl.max],
        str_mode = False
    )

    credit_bureau_a_2_train_df = prepare_df(
        credit_bureau_a_2_files, CFG.train_dir, credit_b_a_2_agg, agg_chunks=True, feat_eng=False
    )
    
    credit_bureau_a_2_train_df_2 = prepare_df(
        credit_bureau_a_2_files, CFG.train_dir, credit_b_a_2_agg_2, agg_chunks=True, feat_eng=False
    )
    credit_bureau_a_2_train_df_2.set_index("case_id", inplace=True)
    
    credit_bureau_a_2_train_df = credit_bureau_a_2_train_df.join(
        credit_bureau_a_2_train_df_2.drop(columns=["WEEK_NUM", "target"]), 
        how="left", on="case_id"
    )
    
    cat_cols_credit_bureau_a_2 = list(credit_bureau_a_2_train_df.select_dtypes("category").columns)
    del credit_bureau_a_2_train_df_2
    gc.collect()
    credit_bureau_a_2_train_df

    credit_bureau_a_2_train_df.describe()
    credit_bureau_a_2_train_df[credit_bureau_a_2_train_df.case_id == 405]

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
    train_base_df = prepare_df(base_files, CFG.train_dir, base_agg)
    cat_cols_base = list(train_base_df.select_dtypes("category").columns)
    breakpoint()

    test_base_df = prepare_df(
        base_files, CFG.test_dir, base_agg, mode="test", cat_cols=cat_cols_base, train_cols=train_base_df.columns
    )

    train_base_df.to_parquet("train_base.parquet")
    credit_bureau_a_1_train_df.to_parquet("credit_bureau_a_1_train_df.parquet")
    credit_bureau_a_2_train_df.to_parquet("credit_bureau_a_2_train_df.parquet")

    print("Train is duplicated:\t", train_base_df["case_id"].duplicated().any())
    print("Train Week Range:\t", (train_base_df["WEEK_NUM"].min(), train_base_df["WEEK_NUM"].max()))

    print()

    print("Test is duplicated:\t", test_base_df["case_id"].duplicated().any())
    print("Test Week Range:\t", (test_base_df["WEEK_NUM"].min(), test_base_df["WEEK_NUM"].max()))

    sns.lineplot(
        data=train_base_df,
        x="WEEK_NUM",
        y="target",
    )
    plt.savefig("test.png")