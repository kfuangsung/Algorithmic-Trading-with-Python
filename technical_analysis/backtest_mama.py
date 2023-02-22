import sys 
sys.path.append('../')
from itertools import repeat
import pandas as pd 
import numpy as np 
from scipy.stats import mode
import talib
import sqlite3 as sql
from p_tqdm import p_map, p_umap, t_map
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
        for j in np.arange(0.01, 0.1, 0.01):
            params.append((i,j))
            
    n_roll = 10
            
    res_df = pd.DataFrame(columns=[
        'optim_winrate', 'optim_ret',
        'median_winrate', 'median_ret',
        'mean_winrate', 'mean_ret',
        'mode_winrate', 'mode_ret'
        ])
    
    for i in tqdm(range(len(my_assets))):
        freq = 'H1'
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
        optim_winrate = opt_df.returns.apply(np.sign).value_counts(normalize=True)[1]
        optim_ret = opt_df.returns.mean()
        
        opt_df['fast'] = opt_df.params.apply(lambda x: x[0] if x is not None else None)
        opt_df['slow'] = opt_df.params.apply(lambda x: x[1] if x is not None else None)
        
        fast_roll_median = opt_df.fast.iloc[:-1].rolling(n_roll).median()
        fast_roll_mean = opt_df.fast.iloc[:-1].rolling(n_roll).mean()
        fast_roll_mode = opt_df.fast.iloc[:-1].rolling(n_roll).apply(lambda x: mode(x)[0])
        
        slow_roll_median = opt_df.slow.iloc[:-1].rolling(n_roll).median()
        slow_roll_mean = opt_df.slow.iloc[:-1].rolling(n_roll).mean()
        slow_roll_mode = opt_df.slow.iloc[:-1].rolling(n_roll).apply(lambda x: mode(x)[0])
        
        # median
        fast_p = fast_roll_median.dropna().values
        slow_p = slow_roll_median.dropna().values
        slow_p = np.where(slow_p < 0.01, 0.01, slow_p)
        params = [*zip(fast_p, slow_p)]
        test_returns =  p_umap(get_strategy_return, repeat(data_sql), params, repeat(freq), 
                               start_dates[-len(params):], end_dates[-len(params):], leave=False)
        test_returns = pd.DataFrame(test_returns, columns=['params', 'returns'])
        median_winrate = test_returns.returns.apply(np.sign).value_counts(normalize=True)[1]
        median_ret = test_returns.returns.mean()
        
        # mean
        fast_p = fast_roll_mean.dropna().values
        slow_p = slow_roll_mean.dropna().values
        slow_p = np.where(slow_p < 0.01, 0.01, slow_p)
        params = [*zip(fast_p, slow_p)]
        test_returns =  p_umap(get_strategy_return, repeat(data_sql), params, repeat(freq), 
                               start_dates[-len(params):], end_dates[-len(params):], leave=False)
        test_returns = pd.DataFrame(test_returns, columns=['params', 'returns'])
        mean_winrate = test_returns.returns.apply(np.sign).value_counts(normalize=True)[1]
        mean_ret = test_returns.returns.mean()
        
        # mode
        fast_p = fast_roll_mode.dropna().values
        slow_p = slow_roll_mode.dropna().values
        slow_p = np.where(slow_p < 0.01, 0.01, slow_p)
        params = [*zip(fast_p, slow_p)]
        test_returns =  p_umap(get_strategy_return, repeat(data_sql), params, repeat(freq), 
                               start_dates[-len(params):], end_dates[-len(params):], leave=False)
        
        test_returns = pd.DataFrame(test_returns, columns=['params', 'returns'])
        mode_winrate = test_returns.returns.apply(np.sign).value_counts(normalize=True)[1]
        mode_ret = test_returns.returns.mean()
        
        res_df.loc[ticker, 'optim_winrate'] = optim_winrate
        res_df.loc[ticker, 'optim_ret'] = optim_ret
        res_df.loc[ticker, 'median_winrate'] = median_winrate
        res_df.loc[ticker, 'median_ret'] = median_ret
        res_df.loc[ticker, 'mean_winrate'] = mean_winrate
        res_df.loc[ticker, 'mean_ret'] = mean_ret
        res_df.loc[ticker, 'mode_winrate'] = mode_winrate
        res_df.loc[ticker, 'mode_ret'] = mode_ret
        
    res_df.to_csv('MAMA_backtest_results.csv')
    print('MAMA_backtest_results.csv is saved.')
    

if __name__ == '__main__':
    main()