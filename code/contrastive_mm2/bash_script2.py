#! /usr/bin/env python
import time
import sys
import subprocess


def process_args(string):
    cmd = string.split(" ")
    # cmd = [" "+ w for w in argus]

    # Remove empty strings
    cmd = list(filter(lambda w: w.strip(), cmd))
    cmd = [sys.executable] + cmd
    return cmd


job = "gru_test2.py --num_epochs 1 --audio_proj_head rnn"
cmd = process_args(job)
print("1---------------------------- Starting: ", cmd)
# subprocess.run(cmd, shell=False)


job = "gru_test2.py --num_epochs 1 --audio_proj_head rnn --pad_pack"
cmd = process_args(job)
print("2---------------------------- Starting: ", cmd)
time.sleep(1)
# subprocess.run(cmd, shell=False)


job = "gru_test2.py --num_epochs 1 --audio_proj_head rnn --pad_pack --loss_type clip_loss"
cmd = process_args(job)
print("3---------------------------- Starting: ", cmd)
time.sleep(1)
# subprocess.run(cmd, shell=False)

job = "gru_test2.py --num_epochs 1 --audio_proj_head gru --pad_pack --loss_type clip_loss"
cmd = process_args(job)
print("3---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run(cmd, shell=False)


job = "gru_test2.py --num_epochs 1 --audio_proj_head gru --pad_pack --loss_type clip_loss --normalize"
cmd = process_args(job)
print("4---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run(cmd, shell=False)

print("----------------- activation METHODS")
job = "gru_test2.py --num_epochs 1 --audio_activation gelu"
cmd = process_args(job)
print("---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run(cmd, shell=False)



print("----------------- POOLING METHODS")
job = "gru_test2.py --num_epochs 1 --text_pooling mean"
cmd = process_args(job)
print("---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run(cmd, shell=False)

job = "gru_test2.py --num_epochs 1 --text_pooling cls"
cmd = process_args(job)
print("---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run(cmd, shell=False)

# job = "ddd"
# cmd = process_args(job)
# print("---------------------------- Starting: ", cmd)
# time.sleep(1)
# # subprocess.run(cmd, shell=False)


# job = "ddd"
# cmd = process_args(job)
# print("---------------------------- Starting: ", cmd)
# time.sleep(1)
# # subprocess.run(cmd, shell=False)