import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.core.multiarray import array as array
from sklearn.metrics import mean_absolute_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.array, y_pred: np.array):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.array, y_pred: np.array):
        
        try:
            logging.info("Calculating MSE")
            mse = mean_absolute_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE : {e}")
            raise e

class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.array, y_pred: np.array):
        
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_absolute_error(y_true, y_pred))
            logging.info(f"MSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RMSE : {e}")
            raise e
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.array, y_pred: np.array):
        
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2 : {e}")
            raise e