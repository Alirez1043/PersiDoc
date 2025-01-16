"""Image processing utility functions"""

import pytesseract
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import os

from orientation_modules.ImageOrientation import ImageOrientation
from skew_modules.IQRHLines import IQRHLines
from skew_modules.HoughBin import HoughBin
from skew_modules.HistScoreSkew import HistScoreSkew
from skew_modules.Furier import Furier

# Configure Tesseract path based on environment
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    DEFAULT_MODEL_PATH = "./models/model.onnx"
else:  # Linux/Docker
    pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
    DEFAULT_MODEL_PATH = "/app/models/model.onnx"

# Get model path from environment variable or use default
MODEL_PATH = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)

def perform_ocr(image):
    """Perform OCR on the given image"""
    return pytesseract.image_to_string(image, lang="fas")

def adjust_orientation(image, method):
    """Adjust image orientation using the specified method"""
    try:
        if method == "ResNET":
            orientation_model = ImageOrientation(model_path=MODEL_PATH)
            oriented_image, class_name = orientation_model.predict_orientation(image)
            return oriented_image, perform_ocr(oriented_image), class_name
        else:
            return image, perform_ocr(image), "No orientation correction applied"
    except Exception as e:
        st.error(f"Error processing image orientation with {method}: {str(e)}")
        return image, perform_ocr(image), "Error in orientation correction"

def adjust_skew(image, method):
    """Adjust image skew using the specified method"""
    try:
        # Convert PIL Image to OpenCV format for skew correction
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if method == "Furier":
            skew_detector = Furier()
        elif method == "HoughBin":
            skew_detector = HoughBin()
        elif method == "HistScoreSkew":
            skew_detector = HistScoreSkew()
        elif method == "IQRHLines":
            skew_detector = IQRHLines()
        else:
            return image, perform_ocr(image), 0, 0, "No skew correction applied"

        affined_image, degree, inf_time = skew_detector.skew(img_cv)

        # Convert back to PIL Image
        skewed_pil = Image.fromarray(cv2.cvtColor(affined_image, cv2.COLOR_BGR2RGB))
        return skewed_pil, perform_ocr(skewed_pil), degree, inf_time, f"Skew corrected by {degree:.2f}°"
    except Exception as e:
        st.error(f"Error processing image skew with {method}: {str(e)}")
        return image, perform_ocr(image), 0, 0, "Error in skew correction"
