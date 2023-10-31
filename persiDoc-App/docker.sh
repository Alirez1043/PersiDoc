#!/bin/bash

docker pull tensorflow/serving

# Create the output_images directory in the current directory if it doesn't exist
if [ ! -d "./app_outputs" ]; then
  sudo mkdir ./app_outputs
fi

# Prompt the user for the data path
echo "Please enter the images directory absolute path: "
read DATA_PATH

# Check if the provided directory exists
if [ ! -d "$DATA_PATH" ]; then
  echo "The provided directory does not exist. Exiting."
  exit 1
fi

export DATA_PATH
export OUTPUT_DIR=$(pwd)/app_outputs

docker-compose build 
docker-compose up
