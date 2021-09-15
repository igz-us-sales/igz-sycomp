import os
import mlrun.feature_store as fs
from mlrun.feature_store.steps import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_X_y(df, target):
    X = df.drop(target, axis=1)
    y = df[target].to_frame()

    # Apply MinMax scaling to X
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])
    
    return X, y

def handler(context):
    # Get data from CSV
    df = context.get_input("dataset").as_df().set_index("patient_id")
    
    # X, y split
    X, y = get_X_y(df=df, target=context.get_param("target"))
    
    # X, y train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=context.get_param("test_size"), random_state=40, stratify=y
    )

    # Create output dir
    os.makedirs(context.artifact_path, exist_ok=True)
    
    # Log X, y to MLRun DB
    output_data = {"X_train" : X_train,
                   "X_test" : X_test,
                   "y_train" : y_train,
                   "y_test" : y_test}
    for key, df in output_data.items():
        context.log_dataset(key=key,
                            df=df,
                            format="csv",
                            artifact_path=context.artifact_subpath('data'))