import sys 
import json
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


def get_mama_return_lastweek(n_period, data, freq):
    
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    data['mama'], data['fama'] = talib.MAMA(data.ht, fastlimit=n_period, slowlimit=n_period/10)
    data.dropna(axis=0, inplace=True)
    data['signals'] = np.where(data.mama > data.fama, 1, -1) 
    
    sat_index = data.resample('W-Sat').last().index
    data = data.loc[sat_index[-2]:sat_index[-1], :]
    
    backtester = IterativeBacktester(data=data, signals=data.signals, freq=freq)
    backtester.backtest(progress_bar=False)

    return n_period, backtester.return_df.loc['TotalReturn', 'Portfolio']


def get_tsf_return_lastweek(n_period, data, freq):
    
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data['tsf'] = talib.TSF(data.ht, n_period)
    data.dropna(axis=0, inplace=True)
    data['signals'] = np.where(data.ht > data.tsf, 1, -1)
    
    sat_index = data.resample('W-Sat').last().index
    data = data.loc[sat_index[-2]:sat_index[-1], :]
    
    backtester = IterativeBacktester(data=data, signals=data.signals, freq=freq)
    backtester.backtest(progress_bar=False)

    return n_period, backtester.return_df.loc['TotalReturn', 'Portfolio']


def get_slope_return_lastweek(n_period, data, freq):
    
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    data['signals'] = talib.LINEARREG_SLOPE(data.ht, timeperiod=n_period).apply(np.sign)
    data.dropna(axis=0, inplace=True)
    
    sat_index = data.resample('W-Sat').last().index
    data = data.loc[sat_index[-2]:sat_index[-1], :]
    
    backtester = IterativeBacktester(data=data, signals=data.signals, freq=freq)
    backtester.backtest(progress_bar=False)

    return n_period, backtester.return_df.loc['TotalReturn', 'Portfolio']


def main():
    freq = 'H1'
    sql_path = f'/home/kachain/python_projects/algorithmic_trading/fxcmpy_trader/PriceData_{freq}.db'
    conn = sql.connect(sql_path)
    optim_parameters = {}
    
    params_ranges = {
        'mama': (0.1, 1, 0.01),
        'slope': (10, 1001, 10),
        'tsf': (10, 1001, 10)
    }

    print("Parameter ranges")
    for k, v in params_ranges.items():
        print(f"\t{k.ljust(8)} --- {v}")

    for ticker in tqdm(my_assets, desc='Tickers', leave=True):    
        data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
        for col in ['open', 'close', 'high', 'low']:
            data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
        sat_index = data_sql.resample('W-Sat').last().index
        
        tqdm.write("***** PARAMETERS OPTIMZATION *****")
        tqdm.write(f"Ticker: {ticker}")
        tqdm.write(f"Frequency: {freq}")
        tqdm.write(f"Dates: {sat_index[-2].strftime('%Y-%m-%d')} to {sat_index[-1].strftime('%Y-%m-%d')}")
        tqdm.write('-'*50)
        
        mama_range = params_ranges['mama']
        mama_results = p_umap(
            get_mama_return_lastweek, 
            np.arange(mama_range[0], mama_range[1], mama_range[2]),
            repeat(data_sql.copy(deep=True)), 
            repeat(freq),
            leave=False
            )
        best_mama_period = pd.DataFrame(mama_results).sort_values(by=1, ascending=False).iloc[0, 0]
                
        slope_range = params_ranges['slope']
        slope_results = p_umap(
            get_slope_return_lastweek, 
            np.arange(slope_range[0], slope_range[1], slope_range[2]),
            repeat(data_sql.copy(deep=True)), 
            repeat(freq),
            leave=False
            )
        best_slope_period = pd.DataFrame(slope_results).sort_values(by=1, ascending=False).iloc[0, 0]

        tsf_range = params_ranges['tsf']
        tsf_results = p_umap(
            get_tsf_return_lastweek, 
            np.arange(tsf_range[0], tsf_range[1], tsf_range[2]),
            repeat(data_sql.copy(deep=True)), repeat(freq),
            leave=False
            )
        best_tsf_period = pd.DataFrame(tsf_results).sort_values(by=1, ascending=False).iloc[0, 0]
    
        optim_parameters[ticker] = {}
        optim_parameters[ticker]['mama'] = float(best_mama_period)
        optim_parameters[ticker]['slope'] = int(best_slope_period)
        optim_parameters[ticker]['tsf'] = int(best_tsf_period)
        
    conn.close()
           
    with open('my_parameters.json', 'w') as f:
        json.dump(optim_parameters, f, indent=4)
        

if __name__ == '__main__':
    main()
