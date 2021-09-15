from mlrun.mlutils.models import gen_sklearn_model
from mlrun.utils.helpers import create_class
import numpy as np
from cloudpickle import dumps

def handler(context):
    # Get X, y
    X_train = context.get_input("X_train").as_df().set_index("patient_id")
    y_train = context.get_input("y_train").as_df().set_index("patient_id")
    X_test = context.get_input("X_test").as_df().set_index("patient_id")
    y_test = context.get_input("y_test").as_df().set_index("patient_id")
    
    # Get model class
    model_pkg_class = context.get_param("model_pkg_class")
    
    # Create model config
    model_config = gen_sklearn_model(model_pkg=model_pkg_class, skparams={})
    model_config["FIT"].update({"X": X_train,
                                "y": np.ravel(y_train)})
    ClassifierClass = create_class(model_config["META"]["class"])
    
    # Create model from config
    model = ClassifierClass(**model_config["CLASS"])
    
    # Train model
    model.fit(**model_config["FIT"])
    
    # Evaluate model
    accuracy = model.score(X_test, y_test)
    
    # Log metrics and model
    context.log_result("accuracy", accuracy)
    context.set_label('class', model_pkg_class)
    context.log_model("model", body=dumps(model),
                      artifact_path=context.artifact_path,
                      model_file="model.pkl",
                      metrics=context.results,
                      tag=context.get_param("model_tag"),
                      labels={"class": model_pkg_class})