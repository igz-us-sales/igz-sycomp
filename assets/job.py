import pandas as pd

def my_func(context):
    # Log scalar values
    context.logger.info("Logging value")
    value = context.get_param("value")
    context.log_result("value", value)
    
    # Log datasets
    df = pd.read_csv("assets/heart.csv")
    context.logger.info("Logging CSV")
    context.log_dataset(
        key="my_csv",
        df=df,
        format="csv",
        artifact_path=context.artifact_path
    )
    
    # Log models with optional metadata
    context.logger.info("Logging model")
    context.log_model(
        key="my_model",
        artifact_path=context.artifact_subpath(context.uid),
        model_file="assets/model.h5",
        metrics={"loss" : 0.23, "accuracy" : 0.94},
        tag="latest",
        parameters={
            "batch_size" : 64,
            "epochs" : 10
        },
        framework="keras",
        labels={"notes" : "noted"},
    )