#!/bin/bash

value="$(/home/kfuangsung/anaconda3/envs/pyenv/bin/python /home/kfuangsung/python_projects/algotrade/check_log_time.py)"

#if return 1 --> restart and run script
if [[ $value -eq 1 ]]; then
    echo script not running
    # sudo pkill python &&
fi