#!/bin/bash
set -eu

CONFIG=$1
GPUS=$2
DATAPATH="${DATAPATH:-$PWD/data}"

# Check if the config file exists
if [ ! -f "$CONFIG" ]; then
  echo "[*] Error: Config file not found: $CONFIG"
  exit -1
fi

if [ -z "$REGISTRY_NAME" ]; then
  echo "[*] Error: REGISTRY_NAME is not set"
  exit -1
fi

if [ -z "$IMAGE_NAME" ]; then
  echo "[*] Error: IMAGE_NAME is not set"
  exit -1
fi

DATAPATH="${DATAPATH:-$PWD/data}"
singularity run -e --bind $PWD:$PWD \
 --bind $DATAPATH:$PWD/data \
 --nv docker://$REGISTRY_NAME/$IMAGE_NAME \
 bash tools/dist_train.sh $CONFIG $GPUS "${@:3}"
