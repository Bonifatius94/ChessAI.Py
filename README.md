# Experimental Python Chess AI

## About
This project offers experimental chess AI approaches meant for learning purposes.

## Disclaimer
There is still work in progess ...

## Main Idea
Evaluating a chess position accurately is very difficult. Just summarizing static values 
per chess piece on the board (like e.g. in 
[Shannon's approach 1949](http://archive.computerhistory.org/projects/chess/related_materials/text/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon.062303002.pdf))
is often not enough. The computation of such scores also heavily depends on an accurate 
estimator that takes positional metadata in consideration (e.g. how pieces cover each 
other or whether there are weaknesses like double peasant, etc.). Using a good heuristic 
estimator may also massively speed up the chess AI's best draw computation in comparison 
to computation-heavy game tree algorithms like minimax (several milliseconds vs. several minutes).

### Deep Learning: Imitate Elaborated Gameplay
Extract grandmaster gameplay data from common sources and learn the draws played by real high-elo players. The training may result into a good artificial chess player and should at least prepare a neuronal network for further refinement.

### Reinforcement Learning: Monte-Carlo Learning
Use the Monte-Carlo techniques to evaluate good and bad actions with lots of self-play. After each game played, compute estimated reward scores for each (state, action) tuple from the game's draw history.

### Reinforcement Learning: Deep Q Learning
Implement Deep Q Learning to determine the goodness of chess draws. This can be used as an alternative to Monte-Carlo results.

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
docker run -v $PWD/src:/home/ai/src \
           -v $PWD/../model_out:/home/ai/model \
           -e MODEL_OUT=/home/ai/model \
           chessai-train pyhton3 pretrain-chessai.py

# run reinforcement learning script
docker run -v $PWD/src:/home/ai/src \
           -v $PWD/../model_out:/home/ai/model \
           -e MODEL_OUT=/home/ai/model \
           chessai-train pyhton3 reinf-chessai.py

# run gameplay test script
docker run -v $PWD/src:/home/ai/src \
           chessai-train python3 gameplay-test.py

# run weights mutation test script
docker run -v $PWD/src:/home/ai/src \
           chessai-train python3 mutate-test.py

# run keras test script (deep learning, classification task)
docker run -v $PWD/src:/home/ai/src \
           chessai-train python3 keras-test.py
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
