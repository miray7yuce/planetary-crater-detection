from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

from app.cv.planet import detect_planet
from app.cv.craters import detect_craters
from app.models.response import DetectionResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5277",  # Blazor frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    content = await file.read()
    npimg = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    planet = detect_planet(image)
    if planet is None:
        return DetectionResponse(
            success=False,
            message="Planet could not be detected in the image."
        )

    x, y, r = planet
    cv2.circle(image, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)

    result_img, coverage = detect_craters(image, planet)

    _, buffer = cv2.imencode(".png", result_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return DetectionResponse(
        success=True,
        coverage=coverage,
        image_base64=img_base64
    )
