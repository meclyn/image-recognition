from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf


app = FastAPI()

# Carregar o modelo pré-treinado MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Função para prever a imagem
def predict_image(image):
    image = image.resize((224, 224))
    
    # Converter a imagem em um array numpy e pré-processar para o modelo
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # Fazer a previsão
    predictions = model.predict(image_array)

    # Decodificar os resultados
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)

    label = decoded_predictions[0][0][1]
    confidence = float(decoded_predictions[0][0][2])

    return {"label": label, "confidence": confidence}

# Função para ler a imagem enviada
def ler_imagem(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# Rota para fazer o upload da imagem
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    image = ler_imagem(await file.read())

    # Passar a imagem para o modelo de Machine Learning
    results = predict_image(image)

    return JSONResponse(content={"results": results})

# Rota GET para verificar se a API está rodando
@app.get("/")
def read_root():
    return {"message": "API de reconhecimento de imagens está funcionando!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
