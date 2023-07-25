#!/bin/sh

echo "Acc"
python final_avg_acc.py --paths "$1"
echo "AAA" 
python avg_acc.py --paths "$1" --key "AAA_Stream/eval_phase/valid_stream/Task000" --add_ema
echo "Wc-Acc"
python avg_acc.py --paths "$1" --add_ema
