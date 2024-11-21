#!/bin/bash
echo "Starting Install"

## Fix GLIBCXX error
echo "Updating gcc"
conda install -c conda-forge gcc=12.1.0

## Install packages
echo "Installing pre-requisites"
pip install -r requirements.txt

## Enable OpenCV window
## TO-DO: Only run this if the person is on Linux/Ubuntu
echo "Updating libgtk and installing libgl1"
sudo apt-get update
sudo apt-get install libgtk2.0-dev
sudo apt-get install pkg-config
sudo apt-get install libgl1