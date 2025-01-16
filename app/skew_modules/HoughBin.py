import cv2
import numpy as np
import time
import math

class HoughBin:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    def affine_image(self, image, degree):
        """Rotate the image based on the specified degree."""
        height, width = image.shape[:2]
        heightNew = int(width * abs(math.sin(math.radians(degree))) + height * abs(math.cos(math.radians(degree))))
        widthNew = int(height * abs(math.sin(math.radians(degree))) + width * abs(math.cos(math.radians(degree))))

        M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        M[0, 2] += (widthNew - width) / 2
        M[1, 2] += (heightNew - height) / 2
        res = cv2.warpAffine(image, M, (widthNew, heightNew), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return res

    def skew(self, image):
        """Correct skew of the image using Hough transform."""
        start_time = time.time()  # Start timer
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binarize the image
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        erosion = cv2.erode(opening, self.kernel)
        
        # Detect edges
        edges = cv2.Canny(erosion, 80, 240, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=250, maxLineGap=70)
        if lines is None:
            return None, 0  # Skip if no lines are detected

        # Extract the lines and calculate their angles
        lines1 = lines[:, 0, :]
        Theta = np.arctan2(lines1[:, 1] - lines1[:, 3], lines1[:, 2] - lines1[:, 0]) * 180 / np.pi

        # Filter lines based on verticality
        valid_indices = np.where((Theta >= -35) & (Theta <= 35))
        Theta = Theta[valid_indices]

        if len(Theta) == 0:
            return None, 0  # Skip if no valid angles are found

        # Compute the dominant angle
        angle_i, angle_bins = np.histogram(Theta, bins=90)
        dominant_angle_index = angle_i.argmax()
        degree = -angle_bins[dominant_angle_index]

        end_time = time.time()  # End timer
        inference_time = end_time - start_time
        
        # Return corrected image, degree, and inference time
        return self.affine_image(image, degree), degree, inference_time
