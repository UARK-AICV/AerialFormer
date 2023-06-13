#!/bin/bash
set -e

CONFIG=$1
CHECKPOINT=$2
DATAPATH="${DATAPATH:-$PWD/data}"

# Check if the config file exists
if [ ! -f "$CONFIG" ]; then
  echo "[*] Error: Config file not found: $CONFIG"
  exit -1
fi

# Check if the checkpoint file exists
if [ ! -f "$CHECKPOINT" ]; then
  echo "[*] Error: Config file not found: $CHECKPOINT"
  exit -1
fi

singularity run -e --bind $PWD:$PWD \
 --bind $DATAPATH:$PWD/data \
 --nv docker://$REGISTRY_NAME/$IMAGE_NAME \
 python tools/test.py $CONFIG "${@:2}"