#!/bin/sh
echo "script 6:"
#python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_0_weights.pt
# python evaluate.py --calc_acc --model_weights_path logs/run4-clip_loss_None_lstm_768_False_2022-06-17_19-59-06/output/full_model_epoch_0_weights.pt
python evaluate.py --calc_acc --model_weights_path logs/run5-clip_loss_None_sph_768_False_2022-06-21_07-38-58/output/full_model_epoch_0_weights.pt

echo "script 7:"
#python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_5_weights.pt
# python evaluate.py --calc_acc --model_weights_path logs/run4-clip_loss_None_lstm_768_False_2022-06-17_19-59-06/output/full_model_epoch_4_weights.pt
python evaluate.py --calc_acc --model_weights_path logs/run5-clip_loss_None_sph_768_False_2022-06-21_07-38-58/output/full_model_epoch_4_weights.pt

echo "Script 8:"
#python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_9_weights.pt
# python evaluate.py --calc_acc --model_weights_path logs/run4-clip_loss_None_lstm_768_False_2022-06-17_19-59-06/output/full_model_epoch_6_weights.pt
# TODO: python evaluate.py --calc_acc --model_weights_path logs/run5-clip_loss_None_sph_768_False_2022-06-21_07-38-58/output/full_model_epoch_0_weights.pt
echo "Bash script done."




