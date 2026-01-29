import cv2
import numpy as np

def detect_craters(image, planet_circle):
    x, y, R = planet_circle

    # mask
    planet_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(planet_mask, (x, y), R, 255, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=planet_mask)

    # contrast
    clahe = cv2.createCLAHE(2.5, (8, 8))
    enhanced = clahe.apply(gray)

    _, thresh = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # only within planet area
    crater_mask = cv2.bitwise_and(thresh, thresh, mask=planet_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crater_mask)

    clean_mask = np.zeros_like(crater_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 50 < area < 5000: 
            clean_mask[labels == i] = 255

    # image overlay
    overlay = image.copy()
    overlay[clean_mask == 255] = (0, 255, 0)

    result = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    # area coverage calculation
    crater_area = np.count_nonzero(clean_mask)
    planet_area = np.count_nonzero(planet_mask)

    coverage = (crater_area / planet_area) * 100 if planet_area > 0 else 0

    return result, round(coverage, 2)
