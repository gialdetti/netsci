#!/usr/bin/env bash

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo PROJECT_PATH: $PROJECT_PATH

PYTHONPATH=${PYTHONPATH}:$PROJECT_PATH jupyter notebook --notebook-dir=$PROJECT_PATH/examples
