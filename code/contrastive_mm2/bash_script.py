#! /usr/bin/env python
import time
import sys
import subprocess


def process_args(string):
    cmd = string.split(" ")
    # cmd = [" "+ w for w in argus]

    # Remove empty strings
    cmd = list(filter(lambda w: w.strip(), cmd))
    return cmd


job = "gru_test2.py --num_epochs 1"
cmd = process_args(job)
print("1---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)


job = "gru_test2.py --num_epochs 1 --pad_pack"
cmd = process_args(job)
print("2---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)


job = "gru_test2.py --num_epochs 1 --audio_proj_head simple_projection_head "
cmd = process_args(job)
print("3---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)

job = "mm3.py --num_epochs 1 --proj_head simple_projection_head"
cmd = process_args(job)
print("4---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)

job = "gru_test2.py --num_epochs 1 --loss_type clip_loss"
cmd = process_args(job)
print("5---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)


job = "gru_test2.py --num_epochs 1 --loss_type clip_loss_simple"
cmd = process_args(job)
print("6---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)


job = "gru_test2.py --num_epochs 1 --loss_type simcse_loss --scale_type learned"
cmd = process_args(job)
print("7---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)

job = "gru_test2.py --num_epochs 1 --loss_type clip_loss --scale_type learned"
cmd = process_args(job)
print("8---------------------------- Starting: ", cmd)
time.sleep(1)
subprocess.run([sys.executable, cmd], shell=False)

# job = "ddd"
# cmd = process_args(job)
# print("---------------------------- Starting: ", cmd)
# time.sleep(1)
# # subprocess.run([sys.executable, cmd], shell=False)