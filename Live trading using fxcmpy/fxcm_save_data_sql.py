import os
import sys
import time
import timeout_decorator
import sqlite3
from datetime import datetime
from tqdm import tqdm
import fxcmpy
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path)
from fxcmtoken import *


def connect_fxcmpy():
    print("-"*50)
    print("Connecting...",end="")
    global con
    con = fxcmpy.fxcmpy(access_token=DEMO_TOKEN2, log_level='error', server='demo', log_file='log.txt')
    print("Done")
    print(f"Connection Status: {con.connection_status}")
    print("-"*50)
    global instruments
    instruments = con.get_instruments_for_candles()


@timeout_decorator.timeout(60)
def get_candles(ticker, time_period):
    return con.get_candles(ticker, period=time_period, number=10000)


def get_price_data(time_period, tickers, main_folder='PriceData', max_attempt=5):
    # tickers --> 'all' or 'my_assets'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.join(dir_path, main_folder)
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    name_db = os.path.join(main_path, f"PriceData_{time_period}.db")
    # print(f"|{datetime.now()}|{name_db}|")
    sql_con = sqlite3.connect(name_db)
    fail_tickers = []
    if tickers == 'all':
        assets = instruments
    elif tickers == 'my_assets':
        assets = my_assets
        
    for ticker in tqdm(assets, leave=False, desc=time_period):
        if len(ticker) < 1: # empty name ''
            pass
        else:
            try:
                data = get_candles(ticker, time_period)
                time.sleep(0.5)
                
                if not data.empty: # save only if dataframe is not empty
                    data.to_sql(ticker, sql_con, if_exists='replace', index=True)
                    tqdm.write(f"{datetime.now()}|{ticker} is saved.")
                    
                else: # if empty save for retry later
                    fail_tickers.append(ticker)
                    tqdm.write(f"{datetime.now()}|{ticker} is empty.")
            except Exception as e:
                fail_tickers.append(ticker)
                tqdm.write(f"{datetime.now()}|{ticker} is failed.")
                print(e)
    
    # retry for fail tickers
    attempt = 0
    # set max attempt , stop if there's no fail_tickers
    while attempt < max_attempt and len(fail_tickers) > 0:
        time.sleep(10)
        tqdm.write(f'|Retrying Fail Tickers|Attempt:{attempt+1}|tickers:{len(fail_tickers)}|')
        fail_tickers_copy = fail_tickers.copy()
        for ticker in tqdm(fail_tickers_copy, leave=False):
            try:
                data = get_candles(ticker, time_period)
                time.sleep(0.5)
                
                if not data.empty: # save only if dataframe is not empty
                    data.to_sql(ticker, sql_con, if_exists='replace', index=True)
                    fail_tickers.remove(ticker)
                    tqdm.write(f"{datetime.now()}|{ticker} is saved.")
                
                else:
                    tqdm.write(f"{datetime.now()}|{ticker} is empty.")
                
            except Exception as e:
                tqdm.write(f"{datetime.now()}|{ticker} is failed.")
                print(e)
                           
        attempt += 1
    if len(fail_tickers) == 0:
        tqdm.write(f"|{datetime.now()}|{name_db}|All data is saved|")
    else:
        tqdm.write(f"|{datetime.now()}|{name_db}|NOT saved:{len(fail_tickers)}|")
        print(fail_tickers)
    tqdm.write("-"*50)
        
    sql_con.commit()
    sql_con.close()
        
def disconnect_fxcmpy():    
    print("-"*50)
    print("Disconnecting...", end="")
    con.close()
    print("Done")
    print(f"Connection Status: {con.connection_status}")
    print("-"*50)
    
if __name__ == "__main__":
    try:
        time_period = sys.argv[1]
        tickers = sys.argv[2]
    except:
        print('Usage: python fxcm_save_data_sql.py {time period or "all"} {"all" or "my_assets"}')
        sys.exit()
        
    if not time_period in time_frame+['all']:
        raise ValueError(f"Incorrect time period '{time_period}'")

    if not tickers in ['all', 'my_assets']:
        raise ValueError(f"Incorrect assets '{tickers}'")
    
    if time_period == 'all':
        connect_fxcmpy()
        for t in tqdm(time_frame, desc='Total'):
            get_price_data(t, tickers)
        disconnect_fxcmpy()
    else:    
        if time_period in time_frame:
            connect_fxcmpy()
            get_price_data(time_period, tickers)
            disconnect_fxcmpy()
        else:
            print('Usage: python fxcm_save_data_sql.py {time period or "all"} {"all" or "my_assets"}')
            sys.exit()
