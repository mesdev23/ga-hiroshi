#!/bin/sh

# start xvfb in background
# Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
# export DISPLAY=:99
# python3 manage.py makemigrations db --noinput
# python3 manage.py migrate db --noinput

# run python script
python3 gasolve.py