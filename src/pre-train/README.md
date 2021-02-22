
# Chess AI Pre-Training

## About
This is a subproject of the chess AI focusing on training from scratch using captured
grandmaster games as experience.

## Dataset

### Raw Input Format
The dataset comes as SQLite database containing (in-state, action, out-state, win rate)
tuples where the win rate can be interpreted as an estimated action reward (= Q value).

### Data Preparation
The model input consists of multiple 2D feature maps encoding the game state as single
bits of the bitboards representation. Therefore create a dataset map function 
transforming bitboards into multiple 8x8 2D feature maps. 

Think of putting black and white pieces of the same piece type into the same feature maps 
and encoding white pieces with 1, black pieces with -1, empty fields with 0.

The Q values to be learned are the win rates, aggregated as n-step returns (this can be 
pre-computed).

## Model and Data Flow

### Model Architecture
The model consists of some CNN layers for 2D feature map extraction followed by some 
fully-connected dense layers doing the interpretation task.

The CNN model could look like:

| Layer Name | Layer Specification                        | Output Shape     |
| ---------- | ------------------------------------------ | ---------------- |
| Conv_Input | takes chess board 2D feature maps as input | (None, 8, 8, 7)  |
| Conv_1     | extract features with 8x8 convolution      | (None, ?, ?, ?)  |
| Conv_2     | extract features with 6x6 convolution      | (None, ?, ?, ?)  |
| Conv_3     | extract features with 5x5 convolution      | (None, ?, ?, ?)  |
| Conv_4     | extract features with 4x4 convolution      | (None, ?, ?, ?)  |
| Conv_5     | extract features with 3x3 convolution      | (None, ?, ?, ?)  |
| Conv_6     | extract features with 3x3 convolution      | (None, ?, ?, ?)  |
| Conv_Flat  | flatten the convoluted features            | (None, ?)        |
| Conv_Out   | dense the flattened, convoluted features   | (None, 512)      |

The rating model could look like:

| Layer Name | Layer Specification                        | Output Shape     |
| ---------- | ------------------------------------------ | ---------------- |
| Rate_Input | takes convoluted board features as input   | (None, 512)      |
| Rate_1     | hidden rating dense layer                  | (None, 512)      |
| Rate_2     | hidden rating dense layer                  | (None, 256)      |
| Rate_3     | hidden rating dense layer                  | (None, 128)      |
| Rate_4     | hidden rating dense layer                  | (None, 64)       |
| Rate_Out   | logits dense layer -> output rating        | (None, 1)        |

## Training
For training, supervised deep learning methods should be appropriate.
As a refinement, there could be off-policy deep reinforcement methods learning
on experience from the grandmaster games dataset.

### Step 1: Make the feature extraction understand how the chess pieces draw
Approach:
- use the convolution layers and flatten the convoluted output of it
- dense it to a (None, 512) shape (this is the layer to be trained !!!)
- add some LSTM cells being feeded the densed data
- generate state transitions of all possible draw outcomes and make the LSTM learn it

Basic Idea:
- it's not so important what the LSTM really predicts as applying draws is deterministic
- but this produces a dense layer extracting a useful meta-representation of game states
- -> use the dense layer as input for the rating task

### Step 2: Use the feature extraction (freezed) and rate the chess positions
Approach:
- add some further dense layers doing the rating task (maybe 1-3 hidden layers)
- dense the output to a single logit predicting the rating -> output shape (None, 1)
- now, start training using the win rates as Q value labels
- **important:** make the feature extraction untrainable

Basic Idea:
- create a rating model that learns the Q values

### Step 3: Unfreeze the feature extraction and refine the rating model
Approach:
- implement a model-based learning algorithm
- feeded the algorithm with experiences from the grandmaster chess games dataset
- unfreeze the feature extraction and make it trainable (with a low learning rate)

Basic Idea:
- model-based learning can be pulled off with stored gameplay experiences (no self-play)
- make the rating predictions even better

## Summary
After finalizing this training, the model should be good enough for further training.
Remove all human decision errors to achieve perfect gameplay.