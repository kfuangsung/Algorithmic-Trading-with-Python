import sys 
import os  
import json
import datetime as dt 
import pandas as pd 
import numpy as np 
import talib
import sqlite3 as sql
from tqdm import tqdm
from itertools import repeat
from p_tqdm import p_umap
sys.path.append('../')
from backtester import * 
from fxcmtoken import my_assets



def get_mama_return(n_period, data, freq, split_date):
    
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    data['mama'], data['fama'] = talib.MAMA(data.ht, fastlimit=n_period, slowlimit=n_period/10)
    data.dropna(axis=0, inplace=True)
    data['signals'] = np.where(data.mama > data.fama, 1, -1) 
    
    train, test = (data.loc[pd.Timestamp(split_date)-dt.timedelta(days=7):pd.Timestamp(split_date)-dt.timedelta(days=1)] , 
                   data.loc[pd.Timestamp(split_date):])
    
    train_backtest = IterativeBacktester(data=train, signals=train.signals, freq=freq)
    train_backtest.backtest(progress_bar=False)
    
    test_backtest = IterativeBacktester(data=test, signals=test.signals, freq=freq)
    test_backtest.backtest(progress_bar=False)

    return n_period, train_backtest.return_df.loc['TotalReturn', 'Portfolio'], test_backtest.return_df.loc['TotalReturn', 'Portfolio']

# ------------------------------------------------------------------

def get_tsf_return(n_period, data, freq, split_date):
    
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data['tsf'] = talib.TSF(data.ht, n_period)
    data.dropna(axis=0, inplace=True)
    data['signals'] = np.where(data.ht > data.tsf, 1, -1)
    
    train, test = (data.loc[pd.Timestamp(split_date)-dt.timedelta(days=7):pd.Timestamp(split_date)-dt.timedelta(days=1)] , 
                   data.loc[pd.Timestamp(split_date):])
    
    train_backtest = IterativeBacktester(data=train, signals=train.signals, freq=freq)
    train_backtest.backtest(progress_bar=False)
    
    test_backtest = IterativeBacktester(data=test, signals=test.signals, freq=freq)
    test_backtest.backtest(progress_bar=False)

    return n_period, train_backtest.return_df.loc['TotalReturn', 'Portfolio'], test_backtest.return_df.loc['TotalReturn', 'Portfolio']

# ----------------------------------------------------------------

def get_slope_return(n_period, data, freq, split_date):
    
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    data['signals'] = talib.LINEARREG_SLOPE(data.ht, timeperiod=n_period).apply(np.sign)
    data.dropna(axis=0, inplace=True)
    
    train, test = (data.loc[pd.Timestamp(split_date)-dt.timedelta(days=7):pd.Timestamp(split_date)-dt.timedelta(days=1)] , 
                   data.loc[pd.Timestamp(split_date):])
    
    train_backtest = IterativeBacktester(data=train, signals=train.signals, freq=freq)
    train_backtest.backtest(progress_bar=False)
    
    test_backtest = IterativeBacktester(data=test, signals=test.signals, freq=freq)
    test_backtest.backtest(progress_bar=False)

    return n_period, train_backtest.return_df.loc['TotalReturn', 'Portfolio'], test_backtest.return_df.loc['TotalReturn', 'Portfolio']

# -----------------------------------------------------------------------------

def get_performance(data, split_date, params, freq):
    # params --> (mama, slope, tsf)
      
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data['mama'], data['fama'] = talib.MAMA(data.ht, fastlimit=params[0], slowlimit=params[0]/10)
    data['slope'] = talib.LINEARREG_SLOPE(data.ht, timeperiod=params[1])
    data['tsf'] = talib.TSF(data.ht, params[2])
    data.dropna(axis=0, inplace=True)

    # signals
    data['mama_signals'] = np.where(data.mama > data.fama, 1, -1) 
    data['slope_signals'] = data.slope.apply(np.sign)
    data['tsf_signals'] = np.where(data.ht > data.tsf, 1, -1)
    signal_cols = ['mama_signals', 'slope_signals', 'tsf_signals']
    data['agg_signals'] = data[signal_cols].mode(axis=1)
    
    # train/test split
    train, test = (data.loc[pd.Timestamp(split_date)-dt.timedelta(days=7):pd.Timestamp(split_date)-dt.timedelta(days=1)] , 
                   data.loc[pd.Timestamp(split_date):])
    
    # backtest train
    train_date_range = train.index[-1]-train.index[0]
    train_backtest = IterativeBacktester(data=train, signals=train.agg_signals, freq=freq)
    train_backtest.backtest(progress_bar=False)
    
    train_ret = train_backtest.return_df.loc['TotalReturn', 'Portfolio']
    train_signal_counts = train_backtest.signals.value_counts()
    train_signal_changes = train_backtest.signals.diff(1).dropna().apply(np.abs).value_counts()
    
    train_total_days = train_date_range.total_seconds() / (60*60*24)
    try:
        train_pos_short = train_signal_counts[-1]
    except:
        train_pos_short = 0
    try:
        train_pos_long = train_signal_counts[1]
    except:
        train_pos_long = 0
    train_pos_changes = (train_signal_changes.index * train_signal_changes).sum()
    
    # backtest test
    test_date_range = test.index[-1]-test.index[0]
    test_backtest = IterativeBacktester(data=test, signals=test.agg_signals, freq=freq)
    test_backtest.backtest(progress_bar=False)
    
    test_ret = test_backtest.return_df.loc['TotalReturn', 'Portfolio']
    test_signal_counts = test_backtest.signals.value_counts()
    test_signal_changes = test_backtest.signals.diff(1).dropna().apply(np.abs).value_counts()
    
    test_total_days = test_date_range.total_seconds() / (60*60*24)
    try:
        test_pos_short = test_signal_counts[-1]
    except:
        test_pos_short = 0
    try:
        test_pos_long = test_signal_counts[1]
    except:
        test_pos_long = 0
    test_pos_changes = (test_signal_changes.index * test_signal_changes).sum()
    
    # (train , test)
    # returns, #days, #short, #long, #posChanges
#     performances[params] = (train_ret, train_total_days, train_pos_short, train_pos_long, train_pos_changes,
#                             test_ret, test_total_days, test_pos_short, test_pos_long, test_pos_changes)
    return (params, 
            train_ret, train_total_days, train_pos_short, train_pos_long, train_pos_changes,
            test_ret, test_total_days, test_pos_short, test_pos_long, test_pos_changes)
    
    
def main():
    freq = 'H1'
    conn = sql.connect(f'../PriceData/PriceData_{freq}.db')
    split_date = '2022-07-24'
    train_returns = []
    test_returns = []
    optim_parameters = {}
    folder = 'optimize_train_test'
    os.makedirs(folder, exist_ok=True)
    fname = f'optimize_separate_train_test_split{split_date}.txt'
    
    print("***** OPTIMZATION *****")
    print(f"Frequency: {freq}")
    print(f"Split date: {split_date}")
    
    params_ranges = {
        'mama': (0.1, 1, 0.1),
        'slope': (100, 1001, 10),
        'tsf': (100, 1001, 10)
    }

    print("Parameter ranges")
    for k, v in params_ranges.items():
        print(f"\t{k.ljust(8)} --- {v}")

    for ticker in tqdm(my_assets, desc='Tickers', leave=True):    
        data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
        for col in ['open', 'close', 'high', 'low']:
            data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
        
        
        mama_range = params_ranges['mama']
        mama_results = p_umap(get_mama_return, 
                            np.arange(mama_range[0], mama_range[1], mama_range[2]),
                            repeat(data_sql.copy(deep=True)), 
                            repeat(freq),
                            repeat(split_date), 
                            leave=False)
        best_mama_period = pd.DataFrame(mama_results).sort_values(by=1, ascending=False).iloc[0, 0]

        
        slope_range = params_ranges['slope']
        slope_results = p_umap(get_slope_return, 
                            np.arange(slope_range[0], slope_range[1], slope_range[2]),
                            repeat(data_sql.copy(deep=True)), 
                            repeat(freq),
                            repeat(split_date),
                            leave=False)
        best_slope_period = pd.DataFrame(slope_results).sort_values(by=1, ascending=False).iloc[0, 0]

        
        tsf_range = params_ranges['tsf']
        tsf_results = p_umap(get_tsf_return, 
                            np.arange(tsf_range[0], tsf_range[1], tsf_range[2]),
                            repeat(data_sql.copy(deep=True)), 
                            repeat(freq),
                            repeat(split_date),
                            leave=False)
        best_tsf_period = pd.DataFrame(tsf_results).sort_values(by=1, ascending=False).iloc[0, 0]
    
    
        params = (best_mama_period, best_slope_period, best_tsf_period)
        optim_parameters[ticker] = {}
        optim_parameters[ticker]['mama'] = float(best_mama_period)
        optim_parameters[ticker]['slope'] = int(best_slope_period)
        optim_parameters[ticker]['tsf'] = int(best_tsf_period)
        
        perf = get_performance(data_sql.copy(deep=True), split_date, params, freq)
        train_ret = perf[1]
        test_ret = perf[6]
        train_returns.append(train_ret)
        test_returns.append(test_ret)
        
        lines = [f"{ticker} | {freq}", 
                f'{params}\nTrainReturn: {train_ret*100:.2f} %\nTestReturn: {test_ret*100:.2f} %',
                '-'*50]
        # for l in lines: print(l)
            
        with open(os.path.join(folder, fname), 'a') as f:
            f.writelines('\n'.join(lines))
            f.write('\n')
        
    conn.close()

    lines = ["*** Average returns ***", 
            f"Train {np.mean(train_returns)*100:.2f} %",
            f"Test {np.mean(test_returns)*100:.2f} %"]

    # for l in lines: print(l)

    with open(os.path.join(folder, fname), 'a') as f:
            f.writelines('\n'.join(lines))
            f.write('\n\n\n')
            
    # with open('my_parameters.json', 'w') as f:
    #     json.dump(optim_parameters, f, indent=4)
        

if __name__ == '__main__':
    main()
