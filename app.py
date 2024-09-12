from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf


app = fastAPI()


# Função pra carregar e processar a imagem

def ler_imagem(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    image = ler_imagem(await file.read())

    # Passar a imagem para o modelo de Machine Learning
    results = predict_image(image)

    return JSONResponse(content={"results": results})

# Função de exemplo que simula a previsao
def predict_image(image):
    # Aqui carregaria um modelo e processaria a imagem
    # exemplo de previsao dummy
    return {"label": "cat", "confidence": 0.98}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)