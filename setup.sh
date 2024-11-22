#!/bin/bash
echo "Starting Install"

## In case of working outside of Ubuntu, but with WSL installed
# Doesn't work because WSL overrides so cannot distinguish between a ubuntu terminal
# and a windows cmd

# echo "This script assumes miniconda specifically is installed as per the instructions. 
# If you are instead using anaconda, this should still work, but every instance of "miniconda3"
# should be exchanged for your anaconda equivalent."
# if [[ $(test -f ~/proc/sys/fs/binfmt_misc/WSLInterop) ]]; then
#     wsl="True"
# elif [[ $(grep -i Microsoft /proc/version) ]]; then
#     wsl="True"
# elif [[ $(uname -r | grep -i WSL) ]]; then
#     wsl="True"
# fi

# if [[ $wsl == "True" ]]; then
#     source ~/miniconda3/etc/profile.d/conda.sh
# fi

## Create and activate environment
# conda create -n "TinyML" python=3.10 -y
# conda activate TinyML

## If WSL, ensure that the correct python is used for pip
# if [[ $wsl == "True" ]]; then
#     export PATH="~/miniconda3/envs/TinyML/Scripts:$PATH"
# fi

## Fix GLIBCXX error
echo "Updating gcc"
conda install -c conda-forge gcc=12.1.0 -y

## Install packages
echo "Installing pre-requisites"
pip install -r requirements.txt

## Enable OpenCV window
# TO-DO: What do we do with Mac?
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Updating libgtk and installing libgl1"
    sudo apt-get update
    sudo apt-get install libgtk2.0-dev
    sudo apt-get install pkg-config
    sudo apt-get install libgl1
else
    echo "OS type not linux, found to be;"
    echo "$OSTYPE"
fi
