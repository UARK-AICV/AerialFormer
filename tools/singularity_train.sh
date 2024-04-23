#!/bin/bash
set -e

CONFIG=$1
DATAPATH="${DATAPATH:-$PWD/data}"
echo $DATAPATH
# Check if the config file exists
if [ ! -f "$CONFIG" ]; then
  echo "[*] Error: Config file not found: $CONFIG"
  exit -1
fi

if [ -z "$REGISTRY_NAME" ]; then
  echo "[*] Error: REGISTRY_NAME is not set"
fi

if [ -z "$IMAGE_NAME" ]; then
  echo "[*] Error: IMAGE_NAME is not set"
fi

singularity run -e --bind $PWD:$PWD \
 --bind $DATAPATH:$PWD/data \
 --nv docker://$REGISTRY_NAME/$IMAGE_NAME \
 python tools/train.py $CONFIG "${@:2}"
