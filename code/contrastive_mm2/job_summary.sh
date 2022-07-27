#!/bin/sh
echo "script 10: "
#python evaluate.py --calc_acc --model_weights_path logs/run3-clip_loss_None_sph_768_False_2022-06-14_08-53-55/output/full_model_epoch_0_weights.pt
# python evaluate.py --calc_acc --model_weights_path logs/run4-clip_loss_None_lstm_768_False_2022-06-17_19-59-06/output/full_model_epoch_0_weights.pt
# python evaluate.py --calc_acc --model_weights_path logs/run5-clip_loss_None_sph_768_False_2022-06-21_07-38-58/output/full_model_epoch_0_weights.pt
# python train.py --train_dataset sp --val_dataset sp --test_dataset sp --weak_shuffle --pad_pack --text_pooling cls --loss_type clip_loss --audio_proj_head gru_v2 --save_model --num_epochs 10 --max_train_samples 5000000 


# python evaluate_v2.py --calc_acc --model_weights_path logs/30m-gru_v2_2022-07-06_07-25-51/output/full_model_weights.pt

# python evaluate_v2.py --calc_acc --model_weights_path logs/30m-mlp_2022-07-05_10-54-42/output/full_model_weights.pt

# python evaluate_v2.py --calc_acc --model_weights_path logs/15m-mlp_2022-07-05_10-52-12/output/full_model_weights.pt

# RUNNING:    15m gru_v2      windows  
# TODO:       5m gru_v2       17 juli
# TODO:       5m mlp          windows
# TODO:       5m sph          windows
# TODO:       15m sph         17 juli
# RUNNING:       25m sph         16 juli

echo "STARTING 1"
python evaluate_summary.py --calc_acc --model_weights_path logs/5m-gru_v2_2022-07-21_07-57-25/output/full_model_weights.pt


echo "STARTING 2"
python evaluate_summary.py --calc_acc --model_weights_path logs/5m-mlp_2022-07-21_07-57-47/output/full_model_weights.pt


echo "STARTING 3"
python evaluate_summary.py --calc_acc --model_weights_path logs/5m-sph_2022-07-20_07-33-19/output/full_model_weights.pt


echo "STARTING 11"
python evaluate_summary.py --calc_acc --model_weights_path logs/15m-gru_v2_2022-07-16_17-00-32/output/full_model_weights.pt


echo "STARTING 12"
python evaluate_summary.py --calc_acc --model_weights_path logs/15m-mlp_2022-07-05_10-52-12/output/full_model_weights.pt


echo "STARTING 14"
python evaluate_summary.py --calc_acc --model_weights_path logs/15m-sph_2022-07-19_18-10-51/output/full_model_weights.pt


echo "STARTING 21"
python evaluate_summary.py --calc_acc --model_weights_path logs/30m-gru_v2_2022-07-06_07-25-51/output/full_model_weights.pt

echo "STARTING 22"
python evaluate_summary.py --calc_acc --model_weights_path logs/30m-mlp_2022-07-05_10-54-42/output/full_model_weights.pt

echo "STARTING 23"
python evaluate_summary.py --calc_acc --model_weights_path logs/25m-sph_2022-07-16_19-44-36/output/full_model_weights.pt




