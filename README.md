# Experimental Python Chess AI

## About
This project offers experimental chess AI approaches meant for learning purposes.

## Main Idea
Evaluating a chess position accurately is very difficult. Just summarizing static values per chess piece on the board (like e.g. in [Shannon's approach 1949](http://archive.computerhistory.org/projects/chess/related_materials/text/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon/2-0%20and%202-1.Programming_a_computer_for_playing_chess.shannon.062303002.pdf)) is often not enough. The computation of such scores also heavily depends on an accurate estimator that takes positional metadata in consideration (e.g. how pieces cover each other or whether there are weaknesses like double peasant, etc.). Using a good heuristic estimator may also massively speed up the chess AI's best draw computation in comparison to computation-heavy game tree algorithms like minimax (several milliseconds vs. several minutes).

## Conceptional Approach No.1: Reinforcement Learning
For training a heuristic chess score function there may be used a reinforcement learning technique. The training algorithm stores the 'best score estimator' that is mutated each training iteration. The actual training consists of the mutated score functions playing against the currently best score function for several times (training phase iterations). After a training iteration the best scoring functions are picked using the win rate. Those functions get transformed into the new "best score estimator' for the next training iteration. 

Inbetween the trainging phases there may also be an evaluation phase performed from time to time by playing against strong chess engines like e.g. stockfish. This ensures that there is no specialization on a specific playstyle that may not be as relevant in real chess games. Moreover it avoids overfitting caused by always playing against the same engines.

Parameters of the scoring function are a set of tuples consisting of chess draws and their resulting chess positions after applying the draw. The tuples may be modeled as 1D numpy arrays of size 14 and datatype np.uint64 (112 bytes) in which the chess draw is just appended to the chess board (consisting of 13 uint64 values) as 14th value.

The scoring function itself uses several fully-connected foreward neuronal layers. The first approach would be 7 layers with 64 neurons. The resulting score is a normalized floating-point value between 0 and 1.

## Conceptional Approach No.2: Markow Chain
Markow Chain that backpropagates possible future game situations to evaluate a score, so the scoring function takes the game tree in consideration. (actual implementation still unclear, it's still just a vague idea)
