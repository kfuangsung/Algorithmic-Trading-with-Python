import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import fxcmpy
import sqlite3
from fxcmtoken import *
from datetime import datetime, timezone
from tqdm import tqdm

def connect_fxcmpy():
    print("-"*50)
    print("Connecting...",end="")
    global con
    con = fxcmpy.fxcmpy(access_token=DEMO_TOKEN, log_level='error', server='demo', log_file='log.txt')
    print("Done")
    print(f"Connection Status: {con.connection_status}")
    print("-"*50)
    global instruments
    instruments = con.get_instruments_for_candles()


def get_price_data(time, tickers, main_folder='PriceData', max_attempt=5):
    # tickers --> 'all' or 'majorforex'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.join(dir_path, main_folder)
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    name_db = os.path.join(main_path, f"PriceData_{time}.db")
    print(f"|{datetime.now()}|{name_db}|")
    sql_con = sqlite3.connect(name_db)
    fail_tickers = []
    if tickers == 'all':
        assets = instruments
    elif tickers == 'majorforex':
        assets = major_forex_pairs
        
    for ticker in tqdm(assets):
        if len(ticker) < 1: # empty name ''
            pass
        else:
            try:
                data = con.get_candles(ticker, period=time, number=10000)
                if not data.empty: # save only if dataframe is not empty
                    data.to_sql(ticker, sql_con, if_exists='replace', index=True)
                else: # if empty save for retry later
                    fail_tickers.append(ticker)
            except:
                fail_tickers.append(ticker)
    
    # retry for fail tickers
    attempt = 0
    # set max attempt , stop if there's no fail_tickers
    while attempt < max_attempt and len(fail_tickers) > 0:
        print(f'|Retrying Fail Tickers|Attempt:{attempt+1}|tickers:{len(fail_tickers)}|')
        for ticker in fail_tickers:
            try:
                data = con.get_candles(ticker, period=time, number=10000)
                if not data.empty: # save only if dataframe is not empty
                    data.to_sql(ticker, sql_con, if_exists='replace', index=True)
                    fail_tickers.remove(ticker)
            except:
                pass
        attempt += 1
    if len(fail_tickers) == 0:
        print(f"|{datetime.now()}|{name_db}|All data is saved|")
    else:
        print(f"|{datetime.now()}|{name_db}|NOT saved:{len(fail_tickers)}|")
        print(fail_tickers)
    print("-"*50)
        
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
        time = sys.argv[1]
        tickers = sys.argv[2]
    except:
        print('Usage: python fxcm_save_data_sql.py {time frame or "all"} {"all" or "majorforex"}')
        sys.exit()
        
    if not time in time_frame+['all']:
        raise ValueError(f"Incorrect time frame '{time}'")

    if not tickers in ['all', 'majorforex']:
        raise ValueError(f"Incorrect assets '{tickers}'")
    
    if time == 'all':
        connect_fxcmpy()
        for t in time_frame:
            get_price_data(t, tickers)
        disconnect_fxcmpy()
    else:    
        if time in time_frame:
            connect_fxcmpy()
            get_price_data(time, tickers)
            disconnect_fxcmpy()
        else:
            print('Usage: python fxcm_save_data_sql.py {time frame or "all"} {"all" or "majorforex"}')
            sys.exit()
