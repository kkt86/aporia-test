import joblib
import os
import pandas as pd

from fastapi import FastAPI


MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "iris_v1.joblib"))

app = FastAPI()



@app.get("/")
async def root():
    return {"status": "OK"}

@app.post("/predict")
async def predict(sepal_length: float = 0.5,
                  sepal_width: float = 0.5,
                  petal_length: float = 0.5,
                  petal_width: float = 0.5) -> int:

    x = pd.DataFrame({"sepal length (cm)": [petal_length],
                      "sepal width (cm)": [petal_width],
                      "petal length (cm)": [sepal_length],
                      "petal width (cm)": [sepal_width],
                      })
    y = MODEL.predict(x)

    return int(y[0])

if __name__=="__main__":
    import uvicorn

    uvicorn.run("main:app", port=8090, reload=True)
