#!/bin/bash
set -e

# Install auto-shaping in editable mode
echo "Installing auto-shaping..."
cd /home/vtprl/auto-shaping && pip install -e .
cd /home

# Start an interactive bash shell
exec bash