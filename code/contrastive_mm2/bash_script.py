#! /usr/bin/env python
import time
import sys
import subprocess

# for cmd in ['python', 'gru_test2.py', '--num_epochs 1'], ['python','gru_test2.py', '--num_epochs', '1', '--test_evalchange']]:
# cmd = ['python', ' gru_test2.py', ' --num_epochs 1']
# subprocess.run([sys.executable, cmd], shell=False)


print("second")
time.sleep(1)

cmd = ['gru_test2.py', ' --num_epochs 1']
subprocess.run([sys.executable, cmd], shell=False)

# print("third")
# time.sleep(1)


# subprocess.run(['gru_test2.py', ' --num_epochs 1'], shell=False)

print("DONE")