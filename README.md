# Experimental Python Chess AI

## About
This project offers experimental chess AI approaches meant for learning purposes.

## Disclaimer
There is still work in progess ...

## Main Idea
Evaluating a chess position accurately is very difficult. Just summarizing static values 
per chess piece on the board (like e.g. in [Shannon's approach 1949]
(http://archive.computerhistory.org/projects/chess/related_materials/text/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon.062303002.pdf))
is often not enough. The computation of such scores also heavily depends on an accurate 
estimator that takes positional metadata in consideration (e.g. how pieces cover each 
other or whether there are weaknesses like double peasant, etc.). Using a good heuristic 
estimator may also massively speed up the chess AI's best draw computation in comparison 
to computation-heavy game tree algorithms like minimax (several milliseconds vs. several minutes).

### Conceptional Approach No.1: Reinforcement Learning (WIP)
For training a heuristic chess score function there may be used a reinforcement learning technique. 
The training algorithm stores the 'best score estimator' that is mutated each training iteration. 
The actual training consists of the mutated score functions playing against the currently best score 
function for several times (training phase iterations). After a training iteration the best scoring 
functions are picked using the win rate. Those functions get transformed into the new "best score 
estimator" for the next training iteration. 

Inbetween the trainging phases there may also be an evaluation phase performed from time to time 
by playing against well-known strong chess engines like e.g. stockfish. This ensures that there 
is no specialization on a specific playstyle that may not be as relevant in real chess games.

Parameters of the scoring function are a set of tuples consisting of chess draws and their 
resulting chess positions after applying the draw. The tuples may be modeled as 1D numpy arrays 
of size 14 and datatype np.uint64 (112 bytes) in which the chess draw is just appended to the 
chess board (consisting of 13 uint64 values) as 14th value.

The scoring function itself uses several fully-connected foreward neuronal layers. The first 
approach would be 7 layers with 64 neurons. The resulting score is a normalized floating-point 
value between 0 and 1.

### Conceptional Approach No.2: Markow Chain
Markow Chain that backpropagates possible future game situations to evaluate a score, so the scoring 
function takes the game tree in consideration. (actual implementation still unclear, it's still just 
a vague idea)

## How to Build / Train
Install git and docker, pull the source code and set up the repository:

```sh
# install docker (e.g. on Ubuntu 18.04)
sudo apt-get update && sudo apt-get install -y git docker.io docker-compose
sudo usermod -aG docker $USER && reboot

# clone the github repo
git clone https://github.com/Bonifatius94/ChessAI.Py
cd ChessAI.Py
```

Build the training environment and run the training scripts using Docker:

```sh
# build the Dockerfile
docker build . -t "chessai-train"

# run pre-training script
docker run -v $PWD/../model_out:/home/ai/model \
           -e MODEL_OUT=/home/ai/model \
           chessai-train pyhton3 pretrain-chessai.py

# run reinforcement learning script
docker run -v $PWD/../model_out:/home/ai/model \
           -e MODEL_OUT=/home/ai/model \
           chessai-train pyhton3 reinf-chessai.py

# run gameplay test script
docker run chessai-train python3 gameplay-test.py

# run weights mutation test script
docker run chessai-train python3 mutate-test.py

# run keras test script (deep learning, classification task)
docker run chessai-train python3 keras-test.py
```

## Experiment Results
### Reinforcement Learning Approach:
#### Results summarization

- creation of TF Keras models and useful techniques with Keras
- creation of a custom weight update function
- implementation of an algorithm to make two Keras estimation functions play against each other
- gameplay training result evaulation (determinating when the game is over, which player won, 
  loop detection, win rate computation, etc.)
- achievement of first training results, players were not able to win in a reasonable time 
  because of random drawing, draw selection is deterministic -> less training progress due to 
  missing variation

#### Resume
The reinforcement learning approach was finally put to work and the model did somewhat train. But the 
given computational power was simply not enough to train a random player's model to grandmaster elo 
from scratch. So there need to be made some adjustments to the training and/or network model.

#### Further Steps / Adjustments / Additional Approaches

*1) Start with a pre-trained network*
- use the already existing win rate cache (SQLite) from the ChessAI.CS project in order to train a model 
  to predict the win rates of draws (supervised learning; data types should be compatible as ChessLib.Py 
  is a clone of ChessAI.CS's Chess.Lib project)
- info: the win rate cache consists of the results of ~ 360000 human grandmaster games, so it's 
  biased data
- after supervised learning, put that pre-trained win rate prediction model as initial model for 
  training and remove the human bias by reinforcement learning and lots of self-play (be careful 
  with the learning rate!!!)
- the training should be a lot more efficient by now as the AI is not playing randomly
- additionally implement ideas of 4) to add more variation at draw selection

*2) Make network model adjustments*
- test if the training results get better when using a smaller network (not 10 hidden layers anymore ...)

*3) Use a draw cache like in AlphaZero's chess approach*
- don't train a model that predicts the strength of chess draws
- cache the chess game tree of known 'best' draws and adjust the win rate of those draws during 
  reinforcement training (this approach can probably start from scratch)

*4) Improve the draw selection variation during training*
- maybe think of dropout layers
- use a random variation of e.g. 2% on draw selection
