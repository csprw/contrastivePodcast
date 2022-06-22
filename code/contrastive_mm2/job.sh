#!/bin/sh
echo "script 3:"
python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_0_weights.pth

echo "script 4:"
python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_5_weights.pth

echo "Script 5:"
python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_9_weights.pth

echo "Bash script done."