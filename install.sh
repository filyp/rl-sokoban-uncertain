#!/usr/bin/env bash
git clone --recursive https://github.com/filyp/rl-starter-files.git
cd rl-starter-files
python -m venv venv
venv/bin/pip install -r requirements.txt