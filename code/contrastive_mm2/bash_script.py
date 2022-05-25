#! /usr/bin/env python

import sys
import subprocess

for cmd in [['gru_test2.py', '--num_epochs 1'], ['gru_test2.py', '--num_epochs', '1', '--test_evalchange']]:
    subprocess.run([sys.executable, cmd], shell=False)