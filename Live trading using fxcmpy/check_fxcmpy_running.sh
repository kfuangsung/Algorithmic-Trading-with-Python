#!/bin/bash
if [[ ! $(pgrep -f dema_tema_trader.py) ]]; then
    echo 'start running'
    /home/azureuser/anaconda3/envs/fxcm/bin/python /home/azureuser/fxcmpy/dema_tema_trader.py
fi

