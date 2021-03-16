# Experimental Python Chess AI

## About
This project offers experimental chess AI approaches meant for learning purposes.

## Disclaimer
There is still work in progess ... don't expect this to be ready-to-use yet.

## Main Idea
Evaluating a chess position accurately is very difficult. Just summarizing static values 
per chess piece on the board (like e.g. in 
[Shannon's approach 1949](http://archive.computerhistory.org/projects/chess/related_materials/text/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon.062303002.pdf))
is often not enough. The computation of such scores also heavily depends on an accurate 
estimator that takes positional metadata in consideration (e.g. how pieces cover each 
other or whether there are weaknesses like double peasant, etc.). Using a good heuristic 
estimator may massively speed up the chess AI's best draw computation in comparison 
to computation-heavy algorithmic approaches like minimax with alpha-beta prune.

### Deep Learning: Imitate Elaborated Gameplay
Extract grandmaster gameplay data from common sources and learn the draws played by real
high-elo players. The training may result into a good artificial chess player and should
at least prepare a neural network for further refinement.

### Reinforcement Learning: Learn by Trial-and-Error
Use several reinforcement learning techniques to evaluate good and bad actions with
lots of self-play. Potential approaches could be Deep-Q Learning, Monte-Carlo 
Tree-Search, Policy Gradient, ... Those training techniques can be used to refine the
initial training on grandmaster game data.

## How to Train
For launching the training, first install docker and git to your environment.

```sh
# install docker (e.g. on Ubuntu 18.04)
sudo apt-get update && sudo apt-get install -y git docker.io docker-compose
sudo usermod -aG docker $USER && reboot
```

After successfully installing the prerequisites, download the source code
and set up the repository.

```sh
# clone the github repo
git clone https://github.com/Bonifatius94/ChessAI.Py
cd ChessAI.Py
```

Now, go ahead and launch the training in a dockerized manner.

```sh
# run all configs sequentially (default behavior)
docker-compose up --build

# run only a specific config (here it's the 'pretrain' config)
docker-compose build && docker-compose run chessai pretrain
```

Currently valid configs are:
- all (=default)
- pretrain

## Learning Approaches

*1) Start with a pre-trained network*
- use the already existing win rate cache (SQLite) from the ChessAI.CS project in order to train a model 
  to predict the win rates of draws (supervised learning; data types should be compatible as ChessLib.Py 
  is a clone of ChessAI.CS's Chess.Lib project)
- info: the win rate cache consists of the results of ~ 360000 human grandmaster games, so it's biased data

- for pre-training, transform the chess board into 2D maps with channels for each piece type and color
  like in the bitboard representation, but don't use the bitboards directly. Instead use a shape like (batch_size, 8, 8, 13)
  and assign each bit on the bitboards as 1 or 0 (float32).
- put chess draws as 2D maps (1 channel for in-pos, 1 channel for out-pos) and append those to the chess boards
- those transformations should allow the usage of Conv2D layers to extract 2D chess position features
- add a rating component learning the win rates of common chess positions

- after supervised learning, put that pre-trained win rate prediction model as initial model for 
  training and remove the human bias by reinforcement learning and lots of self-play (be careful 
  with the learning rate!!!)
- the training should be a lot more efficient by now as the AI is not playing randomly

*2) Add reinforcement learning approaches like Deep Q, Monte Carlo, A2C, Policy Gradient, ...*
- think of the best ways and techinques that suit the chess game
- most likely re-use the pre-trained 2D feature extractor -> no need to learn the game from scratch
