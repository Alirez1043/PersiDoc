import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
import requests
import math



app = Flask(__name__)
MODEL_URI='http://tf_serving:8501/v1/models/saved_model:predict'

model_label2name = {0:'down2up', 1:'up2down' ,2:'right2left',3:'left2right'}
orient2degree = {'right2left': 270, 'left2right': 90, 'up2down': 180, 'down2up': 0}



def get_oriented_image(image, orientation_degree):
    if orientation_degree == 90:
        # Rotate each channel separately
        image = [cv2.flip(channel.T, 0) for channel in cv2.split(image)]
        image = cv2.merge(image)
    elif orientation_degree == 180:
        image = cv2.flip(image, -1)
    elif orientation_degree == 270:
        image = [cv2.flip(channel.T, 1) for channel in cv2.split(image)]
        image = cv2.merge(image)

    return image


def affine_image(image ,degree) :
    
    height, width = image.shape[:2]
    heightNew = int(width * abs(math.sin(math.radians(degree))) + height * abs(math.cos(math.radians(degree))))
    widthNew = int(height * abs(math.sin(math.radians(degree))) + width * abs(math.cos(math.radians(degree))))

    M = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
    M[0, 2] += (widthNew - width) / 2
    M[1, 2] += (heightNew - height) / 2
    res = cv2.warpAffine(image, M, (widthNew, heightNew), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return res



def orient_image(preprocessed_image, predictions):
    class_name = model_label2name[np.argmax(predictions)]
    oriented_image = get_oriented_image(preprocessed_image, orient2degree[class_name])
        
    return oriented_image ,class_name

def load_and_preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (612, 612))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    return image




def determine_score(arr, angle):
    # order of 0 indicates that nearest-neighbor interpolation should be used
    data = inter.rotate(arr, angle, reshape=False, order=0)  # ensures that the array's shape remains unchanged (reshape=False).
    histogram = np.sum(data, axis=1, dtype=float) # do sum for each row because of axis = 1
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
    return histogram, score

def determine_score_wrapper(args):
    return determine_score(*args)


def deskewing_image(oriented_image ,method = 1) : # method 1 --> fast & accurate       |       method 2 --> accrate & fast | 
                                         # method 3 --> perfectly accurate but slow ( good for data labeling ) 
    
    if method == 3 :
        _, thresh = cv2.threshold(oriented_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

        limit = 20
        step = 0.2

        angles = np.arange(-limit, limit + step, step)
        with Pool() as pool: # using pool to optimize inf time
            scores = pool.map(determine_score_wrapper, [(thresh, angle) for angle in angles])

        histograms, actual_scores = zip(*scores)
        
        best_index = np.argmax(actual_scores)
        best_angle = angles[best_index]
        
        return best_angle

    elif method == 2 :
        
        blur = cv2.GaussianBlur(oriented_image, (5, 5), 0)
        org_height , org_width= oriented_image.shape[:2]
        height,width = oriented_image.shape[:2]
        org_center = (width//2, height//2)
        center = (width//2, height//2)
        # threshold image
        _, threshed = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # dilate image
        dilate = cv2.dilate(threshed, (35, 35), iterations=3)
        lines = cv2.HoughLinesP(dilate,1,np.pi/180,200,None,150,10)
        rotation_angle = None
        if lines is not None:
            lines_array = np.squeeze(np.array(lines))
            diff_x = lines_array[:, 2] - lines_array[:, 0]
            diff_y = lines_array[:, 3] - lines_array[:, 1]

            slopes = diff_y / (diff_x + 1e-10)
            angles = np.degrees(np.arctan(slopes))

            # Filter based on angles (horizontal filter)

            horizontal_mask = np.abs(angles) < 30

            # Extract horizontal lines based on angle threshold

            horizontal_lines = lines_array[horizontal_mask]
            filtered_angles = angles[horizontal_mask]

            Q1, Q3 = np.percentile(filtered_angles, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            non_outliers_mask = (filtered_angles >= lower_bound) & (filtered_angles <= upper_bound)

            filtered_angles = filtered_angles[non_outliers_mask]
            final_horizontal_lines = horizontal_lines[non_outliers_mask]

            if len(filtered_angles) != 0:
                rotation_angle = sum(filtered_angles) / len(filtered_angles)
        return rotation_angle
        
        

    else : # default --> method 1

        blank = np.zeros((oriented_image.shape[0], oriented_image.shape[1]), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        # Binarization
        _, thresh = cv2.threshold(oriented_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        erosion = cv2.erode(opening, kernel)
        edges = cv2.Canny(erosion, 80, 240, apertureSize=3)
        # Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=250, maxLineGap=70)
        lines1 = lines[:, 0, :]
        Theta = np.arctan2(lines1[:, 1] - lines1[:, 3], lines1[:, 2] - lines1[:, 0]) * 180/np.pi
        # Filter vertical lines 
        valid_indices = np.where((Theta >= -35) & (Theta <= 35))
        Theta = Theta[valid_indices]
        lines1 = lines1[valid_indices]
        # Rotate
        angle_i, angle = np.histogram(Theta, bins=90)
        angle_m = angle_i.argmax()
        degree = -angle[angle_m]
        
        return degree




@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    try:
        BASE_IMAGE_PATH = "/app/data/"
        BASE_OUTPUT_PATH = "/app/output/"
        data = request.get_json()

        image_name = data.get('image_name')
        method = int(data.get('method', 1))
        image_half = data.get('image_half', False)
        
        image_path = os.path.join(BASE_IMAGE_PATH,image_name)
        output_path = os.path.join(BASE_OUTPUT_PATH,'output_'+image_name)
        
        # Validate the image path (you can enhance this further)
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 400
        
        image = cv2.imread(image_path)
        # cv2.imwrite(output_path ,image)
        if image is None:
            return jsonify({"error": "Failed to read the image"}), 400
        
        preprocessed_image = load_and_preprocess_image(image)

        data = {
            "signature_name": "serving_default",
            "instances": [preprocessed_image.tolist()]
        }

        response = requests.post(MODEL_URI, json=data)
        prediction = response.json()["predictions"][0]
        
        
        
        oriented_image ,class_name = orient_image(image, prediction)
        
        org_width = oriented_image.shape[1]
        org_height = oriented_image.shape[0]
        org_oriented_image = oriented_image.copy()
        if image_half :
            scale_percent = 50
            resized_width = int(oriented_image.shape[1] * scale_percent / 100)
            resized_height = int(oriented_image.shape[0] * scale_percent / 100)
            dim = (resized_width, resized_height)
            oriented_image = cv2.resize(oriented_image, dim, interpolation=cv2.INTER_AREA)
            
        gray = cv2.cvtColor(oriented_image, cv2.COLOR_BGR2GRAY)
        degree = deskewing_image(gray ,method = 1) # set method to 1 or 2 or 3 based on your problem
        rotated_image = affine_image(org_oriented_image ,degree)
        cv2.imwrite(output_path ,rotated_image)

        return jsonify({"message": "Image processed succesfully .", "Image Orient": class_name ,'Image Degree':degree,'Processed image path':output_path})  
    
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080 ,debug=True)
