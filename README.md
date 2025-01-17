<span align="center">
    <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/static/v1?label=TensorFlow&message=Official&color=FF6F00&logo=tensorflow"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/static/v1?label=PyTorch&message=Official&color=EE4C2C&logo=pytorch"></a>
    <a href="https://www.tensorflow.org/tfx/guide/serving"><img src="https://img.shields.io/static/v1?label=TensorFlow%20Serving&message=Official&color=FF6F00&logo=tensorflow"></a>
    <a href="https://www.docker.com/"><img src="https://img.shields.io/static/v1?label=Docker&message=Official&color=2496ED&logo=docker"></a>
    <a href="https://flask.palletsprojects.com/en/2.1.x/"><img src="https://img.shields.io/static/v1?label=Flask&message=Official&color=000000&logo=flask"></a>
</span>

# PersianDocAngleAdjust
In this project, we focus on accurately detecting the orientation of Persian documents and subsequently deskewing them. By leveraging Deep Neural Networks (DNN) for orientation detection and integrating computer vision techniques for deskewing, we've achieved both speed and precision in our solution.



## How to use❓ 🚀

## Running app
1. Clone the repository: `git clone https://github.com/Alirez1043/PersianDocAngleAdjust.git`
2. Navigate to the project directory: `cd PersianDocAngleAdjust/app`
3. Build the Docker image: `docker build -t persidoc .`
4. Run the Docker container: `docker run -p 8501:8501 persidoc`
5. Access the application : Open your web browser and go to `http://localhost:8501` .


## Contact Us 🤝

Should you have any technical inquiries regarding the model, pretraining, code, or publications, please raise an issue in the GitHub repository. This ensures a swift response from our side.

## Citation ↩️

While we haven't published any papers on this work, if you find it valuable and use it in your research or projects, we appreciate your citation. Please reference us using the following bibtex entry:

```bibtex
@misc{PersianDocAngleAdjust
,
  author          = {Tomari, Alireza},
  title           = {PersianDocAngleAdjust: Design and Implementation of a Document Angle Adjustment System to Enhance Text Extraction},
  year            = 2023,
  publisher       = {GitHub},
  howpublished    = {\url{https://github.com/Alirez1043/PersianDocAngleAdjust},
}
