import fxcmpy
import csv
import os 
import time
import random
import pytz
from fxcmtoken import *
from datetime import datetime
from tqdm import tqdm


# Connection
print("Connecting...",end="")
con = fxcmpy.fxcmpy(access_token=DEMO_TOKEN, log_level='error', server='demo', log_file='log.txt')
print("Done")
print(f"Connection Status: {con.connection_status}")
print('-'*50)

# Get instruments
instruments = con.get_instruments_for_candles()
if '' in instruments: instruments.remove('')
print(f"Number of total symbols: {len(instruments)}")

# exlude individual stocks
suffix = set()
for ins in instruments:
    if "." in ins:
        suf = ins.split(".")[-1]
        if suf.islower():
            suffix.add(suf)
suffix = list(suffix)
print('Excluding individual stocks.')
print(suffix)

exclude_symbols = []
for ins in instruments:
    for suf in suffix:
        if suf in ins:
            exclude_symbols.append(ins)
print(f"Number of excluded symbols: {len(exclude_symbols)}")
print(f"Number of target symbols: {len(instruments) - len(exclude_symbols)}")

include_instruments = list(set(instruments) - set(exclude_symbols))
print(f"Number of included symbols {len(include_instruments)}")

# Get margin requirements
margin_req = []
for symbol in tqdm(include_instruments):
    if random.random() > 0.5:
        while True:
            try:
                order = con.create_market_buy_order(symbol, amount=1)
                break
            except:
                continue
    else:
        while True:
            try:
                order = con.create_market_sell_order(symbol, amount=1)
                break
            except:
                continue
    if order == 0: # market is closed --> for stock baskets
        print(f"Market is closed, cannot trade <{symbol}>")
    else:
        time.sleep(5) # need time to update 'usdMr'
        margin = {}
        margin['symbol'] = symbol
        while True:
            try:
                margin['margin'] = con.get_accounts()['usdMr'][0]
                break
            except:
                continue
        margin_req.append(margin)
        while True:
            try:
                con.close_all()
                break
            except:
                continue

while True:
    if con.get_accounts()['usdMr'][0] == 0:
        break
    else:
        print('Closing leftover positions.')
        while True:
            try:
                con.close_all() 
                break
            except:
                continue
	
# Save to .csv
filename = 'fxcm_margin_' + datetime.now(pytz.UTC).strftime("%Y_%m_%d") + '.csv'
header = ['symbol', 'margin']
with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(margin_req)

if os.path.exists(filename):
    print(f'***** Success, <{filename}> is created. *****')
print('-'*50)
# Disconnection
print("Disconnecting...", end="")
con.close()
print("Done")
print(f"Connection Status: {con.connection_status}")

