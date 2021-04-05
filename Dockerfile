
# use base image with chesslib pre-installed
FROM python:3.8-buster
# TODO: think of using an official TensorFlow image with GPU support like tensorflow/tensorflow:latest-gpu

ENV MODELS_ROOT=/app/models
ENV LOGS_ROOT=/app/logs

# install all Python dependencies from requirements.txt
RUN python -m pip install pip --upgrade
ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --upgrade

# copy source code
ADD ./src /app/src
WORKDIR /app/src


# TODO: configure entrypoint
