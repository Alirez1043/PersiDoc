import cv2
import numpy as np
import time


class Furier() :

    def skew(self, image):
        h, w, c = image.shape
        x_center, y_center = (w // 2, h // 2)

        # Find angle to rotate image
        st = time.time()
        rotation_angle = self.get_skewed_angle(image)
        et = time.time()
        M = cv2.getRotationMatrix2D((x_center, y_center), rotation_angle, 1.0)
        borderValue = (255, 255, 255)

        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=borderValue)
        return rotated_image, rotation_angle*-1 , et-st

    def get_skewed_angle(self, image, angle_max = None, search_ratio = 5):
        """Getting angle from a given document image.

        image : np.ndarray
        vertical_image_shape : int
        resize image as preprocessing
        angle_max : float
        maximum angle to searching
        """
        assert isinstance(image, np.ndarray), image

        if angle_max is None:
            angle_max = 5

        m = self._get_fft_magnitude(image)
        a = self._get_angle_radial_projection(m, angle_max=angle_max, num = search_ratio)
        return a 

    def _ensure_gray(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            pass
        return image


    def _ensure_optimal_square(self, image):
        assert image is not None, image
        nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
        output_image = cv2.copyMakeBorder(
            src=image,
            top=0,
            bottom=nh - image.shape[0],
            left=0,
            right=nw - image.shape[1],
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )
        return output_image

    def _ensure_gray(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _get_fft_magnitude(self, image):
        gray = self._ensure_gray(image)
        opt_gray = self._ensure_optimal_square(gray)

        opt_gray = cv2.adaptiveThreshold(
            ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
        )

        dft = np.fft.fft2(opt_gray)
        shifted_dft = np.fft.fftshift(dft)
        magnitude = np.abs(shifted_dft)
        return magnitude

    def _get_angle_radial_projection(self,m, angle_max=None, num=None):
        """Get angle via radial projection.

        Arguments:
        ------------
        angle_max : float
        num : int
        number of angles to generate between 1 degree
        """
        assert m.shape[0] == m.shape[1]
        r = c = m.shape[0] // 2

        if angle_max is None:
            pass

        if num is None:
            num = int((angle_max - (-angle_max)) / 0.5 + 1)

        tr = np.linspace(-1 * angle_max, angle_max, int(angle_max * num * 2)) / 180 * np.pi
        profile_arr = tr.copy()

        def f(t):
            _f = np.vectorize(
                lambda x: m[c + int(x * np.cos(t)), c + int(-1 * x * np.sin(t))]
            )
            _l = _f(range(0, r))
            val_init = np.sum(_l)
            return val_init

        vf = np.vectorize(f)
        li = vf(profile_arr)

        a = tr[np.argmax(li)] / np.pi * 180

        if a == -1 * angle_max:
            return 0
        return a
