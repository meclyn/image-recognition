from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf


app = FastAPI()


# Função pra carregar e processar a imagem
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def predict_image(image):
    image = image.resize((224, 224))
    
    #converter a imagem em um array numpy e pre-processar para o modelo
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # fazer a previsao
    predictions = model.predict(image_array)

    # decodificar os resultados
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)

    label = decoded_predictions[0][0][1]
    confidence = float(decoded_predictions[0][0][2])

    return {"label": label, "confidence": confidence}

def ler_imagem(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    image = ler_imagem(await file.read())

    # Passar a imagem para o modelo de Machine Learning
    results = predict_image(image)

    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)