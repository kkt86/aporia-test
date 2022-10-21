import joblib
import os

import aporia
from fastapi import BackgroundTasks, FastAPI


MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "iris_v1.joblib"))
FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

aporia.init(token=os.getenv("APORIA_API_KEY"),
            environment="local-dev",
            verbose=True)

apr_model = aporia.create_model_version(
    model_id="test",
    model_version="test-local",
    model_type="multiclass",
    features={feature: "numeric" for feature in FEATURES},
    predictions={"class": "categorical"}
)


app = FastAPI()


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/predict")
async def predict(sepal_length: float,
                  sepal_width: float,
                  petal_length: float,
                  petal_width: float,
                  background_tasks: BackgroundTasks) -> int:
    background_tasks.add_task(apr_model.flush)

    features = {feature_name: feature_value
                for feature_name, feature_value in zip(FEATURES,
                                                       [sepal_length, sepal_width, petal_length, petal_width])}

    y = int(MODEL.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0])

    apr_model.log_prediction(features=features, predictions={"class": y})

    return y


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=8090, reload=True)
