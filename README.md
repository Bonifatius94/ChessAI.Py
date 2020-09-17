# Experimental Python Chess AI

## About
This project offers experimental chess AI approaches meant for learning purposes.

## Main Idea
Evaluating a chess position accurately is very difficult. Just summarizing static values per chess piece on the board (like e.g. in [Shannon's approach 1949](http://archive.computerhistory.org/projects/chess/related_materials/text/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon.062303002.pdf)) is often not enough. The computation of such scores also heavily depends on an accurate estimator that takes positional metadata in consideration (e.g. how pieces cover each other or whether there are weaknesses like double peasant, etc.). Using a good heuristic estimator may also massively speed up the chess AI's best draw computation in comparison to computation-heavy game tree algorithms like minimax (several milliseconds vs. several minutes).

### Conceptional Approach No.1: Reinforcement Learning (WIP)
For training a heuristic chess score function there may be used a reinforcement learning technique. The training algorithm stores the 'best score estimator' that is mutated each training iteration. The actual training consists of the mutated score functions playing against the currently best score function for several times (training phase iterations). After a training iteration the best scoring functions are picked using the win rate. Those functions get transformed into the new "best score estimator' for the next training iteration. 

Inbetween the trainging phases there may also be an evaluation phase performed from time to time by playing against well-known strong chess engines like e.g. stockfish. This ensures that there is no specialization on a specific playstyle that may not be as relevant in real chess games.

Parameters of the scoring function are a set of tuples consisting of chess draws and their resulting chess positions after applying the draw. The tuples may be modeled as 1D numpy arrays of size 14 and datatype np.uint64 (112 bytes) in which the chess draw is just appended to the chess board (consisting of 13 uint64 values) as 14th value.

The scoring function itself uses several fully-connected foreward neuronal layers. The first approach would be 7 layers with 64 neurons. The resulting score is a normalized floating-point value between 0 and 1.

### Conceptional Approach No.2: Markow Chain
Markow Chain that backpropagates possible future game situations to evaluate a score, so the scoring function takes the game tree in consideration. (actual implementation still unclear, it's still just a vague idea)

## How to Build / Train
Pull the source code and set up the repository.
```sh
# install docker (e.g. on Ubuntu 18.04)
sudo apt-get update && sudo apt-get install -y git docker.io docker-compose
sudo usermod -aG docker $USER && reboot

# clone the github repo (including all submodules)
git clone --recurse-submodules https://github.com/Bonifatius94/ChessAI.Py
cd ChessAI.Py
```
Use Docker composition files to run your learning algorithms within a Docker environment that has the chesslib and all Python3 tools pre-installed (including Numpy and TensorFlow, further changes can be applied by overriding the *requirements.txt* file).

On composition startup the Dockerfile from the [Bonifatius94/ChessLib.Py](https://github.com/Bonifatius94/ChessLib.Py) submodule gets automatically built (if required). Afterwards the composition mounts the source code into the newly created Docker container and starts the entrypoint.sh script which eventually runs the Python AI script. So launching your AI basically comes down to a single **docker-compose -f compose-file.yml up** command and docker-compose takes care of everything else.
```sh
# start the chess deep reinforcement learning algorithm
docker-compose -f reinf-learning-compose.yml up
```
Here is a small TensorFlow Keras sample for demonstration purposes. This code trains a convolutional object classification algorithm on images to determine clothing pieces like shoes, t-shirts, etc.
```sh
# start the keras deep learning sample taken from https://www.tensorflow.org/tutorials/keras/classification
docker-compose -f keras-test-compose.yml up
```
