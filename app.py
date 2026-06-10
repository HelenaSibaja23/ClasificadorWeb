from flask import Flask
from flask import render_template
from flask import request

import os

import torch
import torch.nn as nn
from torchvision import models

import cv2
import numpy as np
import time

app = Flask(__name__)

device = torch.device("cpu")

modelo = models.mobilenet_v2(
    pretrained=False
)

modelo.classifier[1] = nn.Linear(
    modelo.classifier[1].in_features,
    2
)

modelo.load_state_dict(
    torch.load(
        "modelo_final.pth",
        map_location=device
    )
)

modelo.to(device)

modelo.eval()

print(
    "Modelo cargado correctamente"
)

print(
    "Dispositivo:",
    device
)

mean = np.array([
    121.63204420165984,
    122.54795731275061,
    157.2290834080282
], dtype=np.float32)

std = np.array([
    62.28263642767273,
    60.378799615415645,
    70.96031626452152
], dtype=np.float32)

def zscore_normalize(img):

    img = img.astype(
        np.float32
    )

    norm = (
        img - mean
    ) / (
        std + 1e-8
    )

    min_val = norm.min()

    max_val = norm.max()

    norm = (
        norm - min_val
    ) / (
        max_val - min_val + 1e-8
    )

    norm = (
        norm * 255
    ).clip(
        0,
        255
    )

    return norm.astype(
        np.uint8
    )

def dullrazor(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    result = cv2.inpaint(
        img,
        mask,
        6,
        cv2.INPAINT_TELEA
    )

    return result

def remove_reflective_artifacts_auto(
    img,
    ksize=7
):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    I = gray.astype(
        np.float32
    ) / 255.0

    I_avg = cv2.blur(
        I,
        (ksize, ksize)
    )

    delta = I - I_avg

    TR1 = (
        np.mean(I)
        +
        np.std(I)
    )

    TR2 = (
        np.mean(delta)
        +
        np.std(delta)
    )

    mask = np.logical_and(
        I > TR1,
        delta > TR2
    ).astype(np.uint8) * 255

    result = cv2.inpaint(
        img,
        mask,
        5,
        cv2.INPAINT_TELEA
    )

    return result

def clasificar_imagen(img):

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(
        np.float32
    )

    img = img / 255.0

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    tensor = tensor.to(device)

    inicio = time.perf_counter()

    with torch.no_grad():

        salida = modelo(
            tensor
        )

    fin = time.perf_counter()

    tiempo_inferencia = (
        fin - inicio
    ) * 1000

    probabilidades = torch.softmax(
        salida,
        dim=1
    )

    clase = torch.argmax(
        probabilidades,
        dim=1
    ).item()

    confianza = probabilidades[
        0,
        clase
    ].item()

    return (
        clase,
        confianza,
        tiempo_inferencia
    )

UPLOAD_FOLDER = "uploads"

from flask import send_from_directory

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

@app.route("/")
def inicio():

    return render_template(
        "index.html"
    )

@app.route(
    "/subir",
    methods=["POST"]
)
def subir():

    archivo = request.files["imagen"]

    ruta = os.path.join(
        UPLOAD_FOLDER,
        archivo.filename
    )

    archivo.save(
        ruta
    )

    img = cv2.imread(
        ruta
    )
    
    img = cv2.resize(
        img,
        (224,224)
    )
    
    img = dullrazor(
        img
    )
    
    img = zscore_normalize(
        img
    )
    
    img = remove_reflective_artifacts_auto(
        img
    )
    
    clase, confianza, tiempo = (
        clasificar_imagen(
            img
        )
    )

    if clase == 0:

        resultado = (
            "Lesión Benigna"
        )
    
        resultado_color = "#2E8B57"
    
    else:
    
        resultado = (
            "Lesión Maligna"
        )
    
        resultado_color = "#B22222"
    
    return render_template(
        "index.html",
        resultado=resultado,
        resultado_color=resultado_color,
        confianza=f"{confianza*100:.2f}%",
        tiempo=f"{tiempo:.2f} ms",
        imagen=archivo.filename
    )

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )