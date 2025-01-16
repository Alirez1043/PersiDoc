import cv2
import numpy as np
import math
import time
from multiprocessing import Pool
from scipy.ndimage import rotate

class HistScoreSkew:
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

    def determine_score(self, arr, angle):
        """Calculate the score for a given angle based on row sum differences."""
        data = rotate(arr, angle, reshape=False, order=0)  # ensures that the array's shape remains unchanged (reshape=False).
        histogram = np.sum(data, axis=1, dtype=float)  # sum for each row along axis 1
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    def determine_score_wrapper(self, args):
        """Wrapper to call determine_score in parallel."""
        return self.determine_score(*args)

    def skew(self, image):
        """Method to compute the most optimal angle for image skew correction."""
        start_time = time.time()  # Start timer

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        limit = 20
        step = 0.2
        angles = np.arange(-limit, limit + step, step)

        # Using Pool for parallel computation
        with Pool() as pool:
            scores = pool.map(self.determine_score_wrapper, [(thresh, angle) for angle in angles])

        # Extract histograms and scores
        histograms, actual_scores = zip(*scores)

        # Find the angle with the highest score
        best_index = np.argmax(actual_scores)
        degree = angles[best_index]   # Negate the degree to correct the skew

        end_time = time.time()  # End timer
        inference_time = end_time - start_time

        # Return the corrected image, the computed degree, and the inference time
        return self.affine_image(image, degree), degree, inference_time
