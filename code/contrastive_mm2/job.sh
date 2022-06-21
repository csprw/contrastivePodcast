#!/bin/sh
echo "script 1:"
python evaluate.py --calc_acc --model_weights_path logs/run2-clip_loss_None_gru_768_False_2022-06-12_12-41-49/output/full_model_epoch_5_weights.pth

echo "Script 2:"
python evaluate.py --calc_acc --model_weights_path logs/run2-clip_loss_None_gru_768_False_2022-06-12_12-41-49/output/full_model_epoch_9_weights.pth

echo "Bash script done."