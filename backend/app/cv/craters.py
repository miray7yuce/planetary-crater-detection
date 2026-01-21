import cv2
import numpy as np
import math

def detect_craters(image, planet_circle):
    x, y, R = planet_circle

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), R, 255, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 80, 160)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=20,
        param1=80,
        param2=18,
        minRadius=5,
        maxRadius=60
    )

    crater_area = 0
    overlay = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for cx, cy, cr in circles:
            crater_area += math.pi * cr * cr
            cv2.circle(overlay, (cx, cy), cr, (0, 255, 0), 1)

    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    planet_area = math.pi * R * R
    coverage = (crater_area / planet_area) * 100 if planet_area > 0 else 0

    return image, round(coverage, 2)
