#!/bin/sh
echo "script 10: "

echo "STARTING 1"
# python evaluate_topic_sent.py --calc_acc --model_weights_path logs/5m-gru_v2_2022-07-21_07-57-25/output/full_model_weights.pt


echo "STARTING 2"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/5m-mlp_2022-07-21_07-57-47/output/full_model_weights.pt


echo "STARTING 3"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/5m-sph_2022-07-20_07-33-19/output/full_model_weights.pt


echo "STARTING 11"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/15m-gru_v2_2022-07-16_17-00-32/output/full_model_weights.pt


echo "STARTING 12"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/15m-mlp_2022-07-05_10-52-12/output/full_model_weights.pt


echo "STARTING 14"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/15m-sph_2022-07-19_18-10-51/output/full_model_weights.pt


echo "STARTING 21"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/30m-gru_v2_2022-07-06_07-25-51/output/full_model_weights.pt

echo "STARTING 22"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/30m-mlp_2022-07-05_10-54-42/output/full_model_weights.pt

echo "STARTING 23 (eerder al gedaan, skip)"
python evaluate_topic_sent.py --calc_acc --model_weights_path logs/25m-sph_2022-07-16_19-44-36/output/full_model_weights.pt




