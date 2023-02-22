import sys 
import json
sys.path.append('../')
from itertools import repeat
import pandas as pd 
import numpy as np 
from scipy.stats import mode
import talib
import sqlite3 as sql
from p_tqdm import p_umap, t_map
from tqdm import tqdm
from backtester import * 
from fxcmtoken import my_assets


def get_mama_signals(data, param, start, end):
    data = data.copy(deep=True)
    data["mama"], data["fama"] = talib.MAMA(data.close, fastlimit=param[0], slowlimit=param[1])
    data.dropna(axis=0, inplace=True)
    data['signal'] = np.where(data["mama"] > data["fama"], 1,
                               np.where(data["mama"] < data["fama"], -1, np.nan)
                              )
    data['signal'] = data['signal'].ffill()
    data = data.loc[start:end]
    if len(data) == 0:
        return None
    else:
        return data


def get_strategy_return(data, param, freq, start, end):    
    data = get_mama_signals(data, param, start, end)
    if data is None: return 
    if len(data) == 0: return
    backtester = IterativeBacktester(data=data, signals=data.signal, freq=freq)
    backtester.backtest(progress_bar=False)

    return param, backtester.return_df.loc['TotalReturn', 'Portfolio']


def get_optim_param(data, params, freq, start, end): 
    try:
        res = t_map(get_strategy_return, repeat(data), params, repeat(freq), 
                     repeat(start), repeat(end), leave=False)

        res = pd.DataFrame(res, columns=['params', 'returns'])
        res.sort_values(by='returns', ascending=False, inplace=True)
        best = res.iloc[0]
    except:
        return
    
    return start, end, best['params'], best['returns'] 


def main():
    params = []
    for i in np.arange(0.1, 1, 0.1):
        for j in np.arange(0.01, 0.11, 0.01):
            params.append((i,j))
    
    freq = 'H1'
    n_roll = 5
    aggregate_method = {'EUR/USD': 'median', 'USD/JPY': 'mean', 'GBP/USD': 'mode',
                        'USD/CHF': 'mean', 'AUD/USD': 'mean', 'USD/CAD': 'mode',
                        'NZD/USD': 'median', 'Bund': 'median', 'AUS200': 'mean',
                        'ESP35': 'mean', 'EUSTX50': 'mean', 'FRA40': 'mean',
                        'GER30': 'mean', 'HKG33': 'mean', 'JPN225': 'median',
                        'NAS100': 'mode', 'SPX500': 'mean', 'UK100': 'mean',
                        'US30': 'median', 'Copper': 'mode', 'NGAS': 'mean',
                        'UKOil': 'median', 'USOil': 'mode','XAU/USD': 'mode',
                        'XAG/USD': 'mode'}
    mama_parameters = {}
    
    print("***** OPTIMIZE MAMA parameters *****")
    print(f"TimeFrame: {freq}")
    print(f"RollingWindow: {n_roll}")
    for i, (k, v) in enumerate(aggregate_method.items()):
        print(f"{i}| {k.ljust(10)}{v}")
    
    for i in tqdm(range(len(my_assets))):
        # sql_path = f'/home/kachain/python_projects/algorithmic_trading/PriceData/PriceData_{freq}.db'
        sql_path = f'PriceData_{freq}.db'
        conn = sql.connect(sql_path)
        ticker = my_assets[i]
        data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
        conn.close()
        for col in ['open', 'close', 'high', 'low']:
            data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
            
        tqdm.write(ticker)
        
        sat_index = data_sql.resample('W-Sat').last().index
        start_dates = []
        end_dates = []
        for i in range(1, len(sat_index)-1):
            start_dates.append(sat_index[i])
            end_dates.append(sat_index[i+1])
        optim_params = p_umap(get_optim_param, repeat(data_sql), repeat(params), 
                              repeat(freq), start_dates, end_dates, leave=False)
        opt_df = pd.DataFrame(optim_params, columns=['starts', 'ends', 'params', 'returns'])
        opt_df.sort_values(by='starts', ascending=True, inplace=True)
        opt_df['fast'] = opt_df.params.apply(lambda x: x[0] if x is not None else None)
        opt_df['slow'] = opt_df.params.apply(lambda x: x[1] if x is not None else None)
        
        agg_method = aggregate_method[ticker]
        if agg_method == 'median':
            fast_roll_median = opt_df.fast.rolling(n_roll).median().dropna()
            slow_roll_median = opt_df.slow.rolling(n_roll).median().dropna()
            slow_roll_median = pd.Series(np.where(slow_roll_median < 0.01, 0.01, slow_roll_median))
            p = (fast_roll_median.iloc[-1], slow_roll_median.iloc[-1])
        
        elif agg_method == 'mean':
            fast_roll_mean = opt_df.fast.rolling(n_roll).mean().dropna()
            slow_roll_mean = opt_df.slow.rolling(n_roll).mean().dropna()
            slow_roll_mean = pd.Series(np.where(slow_roll_mean < 0.01, 0.01, slow_roll_mean))
            p = (fast_roll_mean.iloc[-1], slow_roll_mean.iloc[-1])
        
        elif agg_method == 'mode':
            fast_roll_mode = opt_df.fast.rolling(n_roll).apply(lambda x: mode(x)[0]).dropna()
            slow_roll_mode = opt_df.slow.rolling(n_roll).apply(lambda x: mode(x)[0]).dropna()
            slow_roll_mode = pd.Series(np.where(slow_roll_mode < 0.01, 0.01, slow_roll_mode))
            p = (fast_roll_mode.iloc[-1], slow_roll_mode.iloc[-1])
            
        else:
            raise ValueError(f"invalid aggregate method: {agg_method}")
  
        mama_parameters[ticker] = p
        
    with open("mama_parameters.json", "w") as f:
        json.dump(mama_parameters, f, indent=4)
    print("mama_parameters.json is saved.")
  
  
if __name__ == '__main__':
    main()