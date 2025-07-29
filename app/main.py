import os
import mlflow
import logging
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from dagshub import init


class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fetal Health API",
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get api health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                  }
              ])


def load_model():
    """
    Loads a pre-trained model from an MLflow server.

    This function connects to an MLflow server using the provided tracking URI, username,
     and password.
    It retrieves the latest version of the 'fetal_health' model registered on the server.
    The function then loads the model using the specified run ID and returns the loaded model.

    Returns:
        loaded_model: The loaded pre-trained model.

    Raises:
        None
    """
    logging.info('Reading model...')

    repo_owner = 'rafazv'
    repo_name = 'my-first-repo'

    init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    run_id = "134ca4b99a4e49a4ae6f27fbd1436ca2"
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    logging.info("Model successfully loaded!")

    return model


@app.on_event(event_type='startup')
def startup_event():
    """
    A function that is called when the application starts up. It loads a model into the
    global variable `loaded_model`.

    Parameters:
        None

    Returns:
        None
    """
    global loaded_model
    loaded_model = load_model()


@app.get(path='/',
         tags=['Health'])
def api_health():
    """
    A function that represents the health endpoint of the API.

    Returns:
        dict: A dictionary containing the status of the API, with the key "status" and
        the value "healthy".
    """
    return {"status": "healthy"}


@app.post(path='/predict',
          tags=['Prediction'])
def predict(request: FetalHealthData):
    """
    Predicts the fetal health based on the given request data.

    Args:
        request (FetalHealthData): The request data containing the fetal health parameters.

    Returns:
        dict: A dictionary containing the prediction of the fetal health.

    Raises:
        None
    """
    global loaded_model
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)
    logging.info(received_data)
    prediction = loaded_model.predict(received_data)
    logging.info(prediction)
    return {"prediction": str(np.argmax(prediction[0]))}
