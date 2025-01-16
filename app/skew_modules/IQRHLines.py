import cv2
import numpy as np
import time
import math

class IQRHLines:
    def __init__(self):
        pass

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
        """Detect horizontal lines and correct skew of the image."""
        start_time = time.time()  # Start timer
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        height, width = gray_image.shape[:2]
        
        # Threshold the image
        _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate the image to make lines more prominent
        dilate = cv2.dilate(threshed, (35, 35), iterations=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(dilate, 1, np.pi / 180, 200, None, 150, 10)
        
        rotation_angle = None
        if lines is not None:
            lines_array = np.squeeze(np.array(lines))
            diff_x = lines_array[:, 2] - lines_array[:, 0]
            diff_y = lines_array[:, 3] - lines_array[:, 1]

            slopes = diff_y / (diff_x + 1e-10)
            angles = np.degrees(np.arctan(slopes))

            # Filter horizontal lines
            horizontal_mask = np.abs(angles) < 30
            horizontal_lines = lines_array[horizontal_mask]
            filtered_angles = angles[horizontal_mask]

            # Remove outliers based on IQR
            Q1, Q3 = np.percentile(filtered_angles, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            non_outliers_mask = (filtered_angles >= lower_bound) & (filtered_angles <= upper_bound)
            filtered_angles = filtered_angles[non_outliers_mask]
            final_horizontal_lines = horizontal_lines[non_outliers_mask]

            if len(filtered_angles) != 0:
                degree = sum(filtered_angles) / len(filtered_angles)

        # Return corrected image and degree
        if degree is None:
            return None, None, None

        return self.affine_image(image, degree), degree, time.time() - start_time

