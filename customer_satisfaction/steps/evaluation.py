import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "rmse"],
    Annotated[float, "r2"]
]:
    
    try:
        pred = model.predict(X_test)
        
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, pred)
        mlflow.log_metric("mse", mse)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, pred)
        mlflow.log_metric("rmse", rmse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, pred)
        mlflow.log_metric("r2", r2)

        return mse, rmse, r2

    except Exception as e:
        logging.error(f"Error while evaluating model : {e}")
        raise e