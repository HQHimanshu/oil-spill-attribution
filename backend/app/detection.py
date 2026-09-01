import cv2
import numpy as np
from get_metadata import get_metadata_for_image

def detect_oil_spill(image_bytes: bytes):

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")


    # Resize for faster prototype processing
    max_size = 1200

    h, w = image.shape[:2]

    if max(h, w) > max_size:

        scale = max_size / max(h, w)

        image = cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale)
            )
        )


    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # Noise reduction
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # Dark-region detection
    _, mask = cv2.threshold(
        blurred,
        65,
        255,
        cv2.THRESH_BINARY_INV
    )


    # Remove noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    image_area = (
        image.shape[0] *
        image.shape[1]
    )


    # Keep meaningful regions
    min_region_area = max(
        50,
        image_area * 0.0001
    )


    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) >= min_region_area
    ]


    detected_area = sum(
        cv2.contourArea(c)
        for c in valid_contours
    )


    spill_percentage = (
        detected_area / image_area
    ) * 100


    # Limit extreme values
    spill_percentage = min(
        spill_percentage,
        100
    )


    spill_detected = (
        spill_percentage >= 3
    )


    # Confidence
    if spill_percentage < 1:

        confidence = 0.50

    elif spill_percentage < 3:

        confidence = 0.65

    elif spill_percentage < 8:

        confidence = 0.80

    elif spill_percentage < 15:

        confidence = 0.90

    else:

        confidence = 0.95


    # Largest detected region
    center_x = None
    center_y = None

    if valid_contours:

        largest = max(
            valid_contours,
            key=cv2.contourArea
        )

        moments = cv2.moments(largest)

        if moments["m00"] != 0:

            center_x = int(
                moments["m10"] /
                moments["m00"]
            )

            center_y = int(
                moments["m01"] /
                moments["m00"]
            )


    return {

        "spill_detected":
            spill_detected,

        "confidence":
            round(confidence, 2),

        "spill_percentage":
            round(
                spill_percentage,
                2
            ),

        "image_width":
            image.shape[1],

        "image_height":
            image.shape[0],

        "detected_regions":
            len(valid_contours),

        "spill_center_pixel": {

            "x": center_x,

            "y": center_y

        },

        "detection_method":
            "Prototype dark-region anomaly detection",

        "model_status":
            "Prototype"
    }