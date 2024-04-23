#!/bin/bash
set -e

pip install -e .

exec "$@"
