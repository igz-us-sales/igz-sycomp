
import os
from kfp import dsl
from mlrun import mount_v3io, NewTask
import yaml
import nuclio

funcs = {}

# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for fn in functions.values():
        # Mount V3IO filesystem
        fn.apply(mount_v3io())

    functions["deploy-model"].spec.base_spec['spec']['loggerSinks'] = [{'level': 'info'}]
    functions["deploy-model"].spec.min_replicas = 1
    functions["deploy-model"].spec.max_replicas = 1

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="NLP Heart Disease Pipeline",
    description="Kubeflow Pipeline Demo to detect Heart Disease"
)
def kfpipeline(dataset:str,
               target:str="target",
               test_size:float=0.2,
               model_tag:str="latest",
               debug_logs:bool=True):    
    
    # Get data from feature store, prep, train/test split
    get_prep_data = funcs['get-prep-data'].as_step(
        handler="handler",
        inputs={"dataset" : dataset},
        params={"target" : target,
                "test_size" : test_size},
        outputs=["X_train", "X_test", "y_train", "y_test"],
        verbose=debug_logs)
    
    # Train/evaluate model
    train = funcs['train-eval-model'].as_step(
        handler="handler",
        inputs={"X_train" : get_prep_data.outputs['X_train'],
                "X_test" : get_prep_data.outputs['X_test'],
                "y_train" : get_prep_data.outputs['y_train'],
                "y_test" : get_prep_data.outputs['y_test']},
        hyperparams={'model_pkg_class': ["sklearn.ensemble.RandomForestClassifier", 
                                         "sklearn.linear_model.LogisticRegression",
                                         "sklearn.ensemble.AdaBoostClassifier"]},
        params={"model_tag" : model_tag},
        selector='max.accuracy',
        outputs=["model", "accuracy"],
        verbose=debug_logs)
    
    # Deploy model
    with dsl.Condition(name="model_accuracy_validation", condition=train.outputs['accuracy'] > 0.85):
        deploy = funcs["deploy-model"].deploy_step(models=[{"key": "heart_disease_model", "model_path": train.outputs['model']}])
