
# use base image with chesslib pre-installed
FROM python:3.8-buster

ENV MODELS_ROOT=/app/models
ENV LOGS_ROOT=/app/logs

# install all Python dependencies from requirements.txt
RUN python3 -m pip install pip --upgrade
ADD ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt --upgrade

# copy source code
ADD ./src /app/src
WORKDIR /app/src


# TODO: configure entrypoint
