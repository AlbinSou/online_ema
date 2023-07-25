#!/bin/sh

PROJECT_DIR="$(cd .. && pwd)"
SRC_DIR="$PROJECT_DIR/src"

echo $PROJECT_DIR
echo $SRC_DIR

for SEED in 0 1 2 3 4 5
do
    python $SRC_DIR/main_boundaries.py --config $SRC_DIR/config/imagenet/er.yml --seed $SEED
done
