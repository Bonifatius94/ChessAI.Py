
# use base image with chesslib pre-installed
FROM chesslib-python3:latest

# install all Python dependencies from requirements.txt
ADD ./requirements.txt /requirements.txt
WORKDIR /
RUN pip3 install -r requirements.txt

# copy source code
ADD ./src /home/ai/src
WORKDIR /home/ai/src

# configure entrypoint
#ADD ./entrypoint.sh /entrypoint.sh
#ENTRYPOINT ["/entrypoint.sh"]
