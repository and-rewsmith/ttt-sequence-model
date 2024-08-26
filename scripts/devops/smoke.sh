#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

export WANDB_MODE=dryrun

# TODO: replace dummy line below when ready
# python -m model.tests.smoke.smoke_zenke_2a > ci_log.txt 2>&1 || python_exit_code=$?

if [ -n "$python_exit_code" ]; then
    cat ci_log.txt
    exit $python_exit_code
fi
