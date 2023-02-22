import sys 
import os  
import datetime as dt 
import pandas as pd 
import numpy as np 
import talib
import sqlite3 as sql
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm import tqdm
from p_tqdm import p_umap
sys.path.append('../')
from backtester import * 
from fxcmtoken import my_assets


def get_data(ticker, freq):
    conn = sql.connect(f'../PriceData/PriceData_{freq}.db')
    print(ticker)
    data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
    for col in ['open', 'close', 'high', 'low']:
        data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
    conn.close()
    data_sql.tail()
    
    return data_sql
    

def get_parameters():
    params_list = []

    for mama_period in np.arange(0.1, 1, 0.1):
        for slope_period in range(50, 501, 50):
            for tsf_period in range(50, 501, 50):
                params_list.append((mama_period, slope_period, tsf_period))
    
    return params_list


def get_performance(data, split_date, params, freq):
    # params --> (mama, slope, tsf)
      
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data['mama'], data['fama'] = talib.MAMA(data.ht, fastlimit=params[0], slowlimit=params[0]/10)
    data['slope'] = talib.LINEARREG_SLOPE(data.close, timeperiod=params[1])
    data['tsf'] = talib.TSF(data.ht, params[2])
    data.dropna(axis=0, inplace=True)

    # signals
    data['mama_signals'] = np.where(data.mama > data.fama, 1, -1) 
    data['slope_signals'] = data.slope.apply(np.sign)
    data['tsf_signals'] = np.where(data.ht > data.tsf, 1, -1)
    signal_cols = ['mama_signals', 'slope_signals', 'tsf_signals']
    data['agg_signals'] = data[signal_cols].mode(axis=1)
    
    # train/test split
    train, test = (data.loc[:pd.Timestamp(split_date)-dt.timedelta(days=1)] , 
                   data.loc[pd.Timestamp(split_date):])
    
    # backtest train
    train_date_range = train.index[-1]-train.index[0]
    train_backtest = IterativeBacktester(data=train, signals=train.agg_signals, freq=freq)
    train_backtest.backtest(progress_bar=False)
    
    train_ret = train_backtest.return_df.loc['TotalReturn', 'Portfolio']
    train_signal_counts = train_backtest.signals.value_counts()
    train_signal_changes = train_backtest.signals.diff(1).dropna().apply(np.abs).value_counts()
    
    train_total_days = train_date_range.total_seconds() / (60*60*24)
    train_pos_short = train_signal_counts[-1]
    train_pos_long = train_signal_counts[1]
    train_pos_changes = (train_signal_changes.index * train_signal_changes).sum()
    
    # backtest test
    test_date_range = test.index[-1]-test.index[0]
    test_backtest = IterativeBacktester(data=test, signals=test.agg_signals, freq=freq)
    test_backtest.backtest(progress_bar=False)
    
    test_ret = test_backtest.return_df.loc['TotalReturn', 'Portfolio']
    test_signal_counts = test_backtest.signals.value_counts()
    test_signal_changes = test_backtest.signals.diff(1).dropna().apply(np.abs).value_counts()
    
    test_total_days = test_date_range.total_seconds() / (60*60*24)
    test_pos_short = test_signal_counts[-1]
    test_pos_long = test_signal_counts[1]
    test_pos_changes = (test_signal_changes.index * test_signal_changes).sum()
    
    return (params, 
            train_ret, train_total_days, train_pos_short, train_pos_long, train_pos_changes,
            test_ret, test_total_days, test_pos_short, test_pos_long, test_pos_changes)


def write_to_excel(res_df, ticker, file_name):
    
    if os.path.exists(file_name):
        mode = 'a'
    else:
        mode = 'w'

    engine = "openpyxl"
    with pd.ExcelWriter(file_name, engine=engine, mode=mode) as writer:  
        workBook = writer.book
        try:
            workBook.remove(workBook[ticker.replace('/','_')])
        except:
            print("worksheet doesn't exist")
        finally:
            res_df.to_excel(writer, sheet_name=ticker.replace('/','_'), engine=engine)
            print(f"wrote {ticker} results to {file_name}")
        writer.save()
        

if __name__ == "__main__":
    
    freq = 'm5'
    split_date = '2022-07-17'
    num_cpus = 6
    file_name = 'performances_train_test.xlsx'
    
    for ticker in tqdm(my_assets, desc='my_assets'):
        print("***** OPTIMIZATION *****")
        print(f'Ticker: {ticker}')
        print(f'Time frame: {freq}')
        print(f'Split date: {split_date}')
        print(f'Number of cpus: {num_cpus}')
        print('-'*50)
            
        data_sql = get_data(ticker, freq)
        params_list = get_parameters()
        results = p_umap(get_performance, 
                        repeat(data_sql), 
                        repeat(split_date), 
                        params_list, 
                        repeat(freq),
                        **{"num_cpus": num_cpus})
        res_df = pd.DataFrame(results)
        res_df.columns = ['params', 
                        'train_returns', 'train_n_days', 'train_n_short', 'train_n_long', 'train_n_changes',
                        'test_returns', 'test_n_days', 'test_n_short', 'test_n_long', 'test_n_changes']
        res_df.set_index('params', inplace=True)
        res_df.sort_values(by='train_returns', ascending=False, inplace=True)
        write_to_excel(res_df, ticker, file_name)