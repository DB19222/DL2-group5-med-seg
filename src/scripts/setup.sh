#!/bin/bash

ROOT=$(pwd | sed 's/\/src/scripts$/')
cd "$ROOT"

# create virtual environment
venv="$ROOT/.venv"
if [ ! -d "$venv" ]; then
    python3 -m venv "$venv"
fi

# activate virtual environment
source "$venv/bin/activate"

# install dependencies
pip install -r "$ROOT/requirements.txt"

# update the peft submodule
git submodule update --init --recursive

echo "Setup complete. Run"
echo 
echo 'source $venv/bin/activate' 
echo 
echo "to activate the virtual environment."