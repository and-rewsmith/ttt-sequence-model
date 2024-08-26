#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

pip install pytest

# TODO: replace dummy line below when ready
# python -m pytest -rP model/tests