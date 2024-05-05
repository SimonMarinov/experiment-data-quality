#!/bin/bash

# Function to create folders and remove them if they exist
create_folders() {
    if [ -d "$1" ]; then
        rm -r "$1"
    fi
    mkdir -p "$1"
}

# Create data folder and subfolders train and test
create_folders "data/train"
create_folders "data/test"

# Run python scripts pol.py and gen.py
python3 data_generator.py
python3 polution.py
