#!/bin/bash

# ===========================================
#    CHESS AI TRAINING ENTRYPOINT SCRIPT
# ===========================================

# define args
export AI_SRC_ROOT=/home/ai/src
export MAIN_SCRIPT=$1

# print the tool versions installed
python3 --version

# start the training script
python3 $AI_SRC_ROOT/$MAIN_SCRIPT

# TODO: add further steps for exporting a trained model
