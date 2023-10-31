FROM python:3.8-slim


WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8080

# ENV TF_SERVING_ENDPOINT=tf_serving:8501

CMD python3 app.py
