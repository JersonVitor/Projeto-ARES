from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from .prediction import prediction

app = FastAPI()

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        gesture = await prediction(video)
        return {"gesture": gesture}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/response")
def data():
    return {"predict": "Sapo", "cofidence": 0.87}
