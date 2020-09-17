#!/bin/bash

# ===========================================
#    CHESS AI TRAINING ENTRYPOINT SCRIPT
# ===========================================

# define constants
export AI_SRC_ROOT=/home/ai/src

# read configurable script args
#   $1: the python script to be executed
export MAIN_SCRIPT=$1

# print python version and installed module versions
python3 --version
python3 -c $'import pip\nprint(sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()]))'

# start the training script
python3 $AI_SRC_ROOT/$MAIN_SCRIPT

# TODO: add further steps for exporting a trained model
