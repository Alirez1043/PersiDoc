import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

class ImageOrientation:
    def __init__(self, model_path, input_size=(224, 224)):
        """
        Initialize the model with the path to the ONNX model and image input size.
        """
        self.model_path = model_path
        self.input_size = input_size
        self.ort_session = ort.InferenceSession(model_path)
        self.class_mapping = {'down2up': 0, 'up2down': 3, 'left2right': 1, 'right2left': 2}
        self.pred2class = {v: k for k, v in self.class_mapping.items()}
        self.orient2degree = {'right2left': 270, 'left2right': 90, 'up2down': 180, 'down2up': 0}
        self.model_label2name = {0: 'down2up', 3: 'up2down', 1: 'left2right', 2: 'right2left'}

    def preprocess_image(self, image):
        """
        Preprocess the image: resize, normalize, and convert to suitable format for the model.
        """
        if isinstance(image, Image.Image):
            # Ensure the image is in RGB format
            image = image.convert("RGB")
        else:
            raise ValueError("Input must be a PIL.Image object.")

        # Resize image
        image = image.resize(self.input_size, Image.Resampling.BICUBIC)

        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        # Reorder dimensions to match model input
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def get_oriented_image(self, image, orientation_degree):
        """
        Rotate the image based on the predicted orientation degree.
        """
        if orientation_degree == 90:
            image = [cv2.flip(channel.T, 0) for channel in cv2.split(image)]
            image = cv2.merge(image)
        elif orientation_degree == 180:
            image = cv2.flip(image, -1)
        elif orientation_degree == 270:
            image = [cv2.flip(channel.T, 1) for channel in cv2.split(image)]
            image = cv2.merge(image)

        return image

    def predict_orientation(self, pil_image):
        """
        Predict the orientation of the image and return the oriented image and class name.
        """
        # Preprocess the PIL image
        img = self.preprocess_image(pil_image)
        img_numpy = img.astype(np.float32)

        # Run inference
        outputs = self.ort_session.run(None, {'input': img_numpy})

        # Get predictions
        output_tensor = np.array(outputs[0])
        predicted_class = np.argmax(output_tensor, axis=1)[0]
        print(f"Predicted class: {predicted_class}")
        class_name = self.model_label2name[predicted_class]
        print(f"Predicted class name: {class_name}")
        orientation_degree = self.orient2degree[class_name]
        print(f"Predicted orientation degree: {orientation_degree}")
        # Rotate original image (convert PIL image to numpy)
        numpy_image = np.array(pil_image)
        oriented_image = self.get_oriented_image(numpy_image, orientation_degree)
        return oriented_image, class_name
