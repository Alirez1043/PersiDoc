#!/bin/bash

# Function to get a yes/no answer from the user
prompt_yes_no() {
    while true; do
        read -p "$1 [y/n]: " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Prompt user for inputs
echo "Enter the image name (e.g., first_image.jpeg):"
read image_name


# Explain methods
echo "Methods for deskewing:"
echo "1. Method 1: High speed and good accuracy  ."
echo "2. Method 2: High speed and good accuracy  ."
echo "3. Method 3: Low speed  and perfect accuracy (Good for data labeling)"

echo "Enter the method you wanna use for deskewing (1 or 2 or 3):"
read method

if prompt_yes_no "Do you wanna use half of image for deskewing(this reduce accracy but improve inference time)?"; then
    image_half=true
else
    image_half=false
fi

# Make the curl request
curl -X POST -H "Content-Type: application/json" \
     -d "{\"image_name\": \"$image_name\", \"method\": $method, \"image_half\": $image_half}" \
     http://localhost:8080/preprocess

echo # Print a newline
