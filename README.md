<span align="center">
    <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/static/v1?label=TensorFlow&message=Official&color=FF6F00&logo=tensorflow"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/static/v1?label=PyTorch&message=Official&color=EE4C2C&logo=pytorch"></a>
    <a href="https://www.tensorflow.org/tfx/guide/serving"><img src="https://img.shields.io/static/v1?label=TensorFlow%20Serving&message=Official&color=FF6F00&logo=tensorflow"></a>
    <a href="https://www.docker.com/"><img src="https://img.shields.io/static/v1?label=Docker&message=Official&color=2496ED&logo=docker"></a>
    <a href="https://flask.palletsprojects.com/en/2.1.x/"><img src="https://img.shields.io/static/v1?label=Flask&message=Official&color=000000&logo=flask"></a>
</span>

# PersiDoc
In this project, we focus on accurately detecting the orientation of Persian documents and subsequently deskewing them. By leveraging Deep Neural Networks (DNN) for orientation detection and integrating computer vision techniques for deskewing, we've achieved both speed and precision in our solution.
<p align="center">
  <img src="https://github.com/Alirez1043/PersiDoc/blob/main/images/Screenshot%20from%202023-11-01%2000-49-44.png" alt="" width="600" height="400">
</p>



<p align="center">
  <img src="https://github.com/Alirez1043/PersiDoc/blob/main/images/Screenshot%20from%202023-11-01%2001-28-00.png" alt="" width="1000" height="400">
</p>


## How to use❓ 🚀

## Running app
1. Clone the repository: `git clone https://github.com/Alirez1043/PersiDoc.git`
2. Navigate to the project directory: `cd persiDoc-App`
3. Run docker.sh
   
   ```
   bash docker.sh
4. When docker.sh is running , asks you to define your absolute path of directory contains images for inference
   
    `Please enter the images directory absolute path: `
5. After container starts , Run inference.sh
   ```
   bash inference.sh
Inference bash script asks you for options to use :

1. image_name

2. select method :  (1 or 2 or 3) default = 1

 Methods for deskewing:
 
     1. Method 1: High speed and good accuracy  
     
     2. Method 2: High speed and good accuracy  
     
     3. Method 3: Low speed  and perfect accuracy (Good for data labeling)

3. half_image(y/n)

    if yes :   the image size will be resized to half (Inference Time Optimization Method , default = False)

Finally :
    Output preprocessed Image will be save at ./app_outputs (you do not need to make it  ,docker.sh handle it)
## Contact Us 🤝

Should you have any technical inquiries regarding the model, pretraining, code, or publications, please raise an issue in the GitHub repository. This ensures a swift response from our side.

## Citation ↩️

While we haven't published any papers on this work, if you find it valuable and use it in your research or projects, we appreciate your citation. Please reference us using the following bibtex entry:

```bibtex
@misc{PersiDoc,
  author          = {Tomari, Alireza},
  title           = {PersiDoc: Persian Document orientation detection and deskewing},
  year            = 2023,
  publisher       = {GitHub},
  howpublished    = {\url{https://github.com/Alirez1043/PersiDoc},
}
