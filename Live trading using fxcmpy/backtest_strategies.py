import sys 
sys.path.append('../')
from itertools import repeat
import pandas as pd 
import numpy as np 
import bottleneck as bn
from scipy.stats import mode
# import talib
import tulipy as ti
import sqlite3 as sql
from p_tqdm import p_map, p_umap, t_map
from tqdm import tqdm
from backtester import * 
from fxcmtoken import my_assets


def ti_pad(values, func, period):
        a = np.full_like(values, fill_value=np.nan)
        b = func(values, period)
        a[-len(b):] = b
        return a
    

def get_signals(data, param, start, end, strategy):
    if param[0] in [None, np.nan]: return np.nan
    if param[1] in [None, np.nan]: return np.nan
    
    data = data.copy(deep=True)
    
    try:
        param1_period = int(param[0])
        param2_period = int(param[1])
        
        # 1
        if strategy == 'SMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.sma, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.sma, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 2
        elif strategy == 'EMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.ema, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.ema, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 3
        elif strategy == 'KAMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.kama, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.kama, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 4
        elif strategy == 'DEMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.dema, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.dema, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 5
        elif strategy == 'TEMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.tema, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.tema, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 6
        elif strategy == 'TRIMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.trima, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.trima, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 7
        elif strategy == 'HMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.hma, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.hma, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 8
        elif strategy == 'WMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.wma, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.wma, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 9
        elif strategy == 'ZLEMA':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.zlema, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.zlema, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 10
        elif strategy == 'TSF':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.tsf, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.tsf, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 11    
        elif strategy == 'WILDERS':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.wilders, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.wilders, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 12
        elif strategy == 'SLOPE':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.linregslope, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.linregslope, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
        
        # 13
        elif strategy == 'TRIX':
            data = data.assign(
                param1=lambda x: ti_pad(x.close.values, ti.trix, param1_period),
                param2=lambda x: ti_pad(x.close.values, ti.trix, param2_period),
                signal=lambda x: bn.push(np.where(x.param1 > x.param2, 1, np.where(x.param1 < x.param2, -1, np.nan)), axis=0)
            )
            
    except:
        return np.nan
    
    data.dropna(axis=0, inplace=True)
    data = data.loc[start:end]
    if len(data) == 0:
        return np.nan
    else:
        return data


def get_strategy_return(data, param, freq, start, end, strategy):    
    data = get_signals(data, param, start, end, strategy)
    if not isinstance(data, pd.DataFrame): return (param, np.nan)
    if len(data) == 0: return (param, np.nan)
    backtester = IterativeBacktester(data=data, signals=data.signal, freq=freq)
    backtester.backtest(progress_bar=False)

    return param, backtester.return_df.loc['TotalReturn', 'Portfolio']


def get_optim_param(data, params, freq, start, end, strategy): 
    try:
        res = t_map(get_strategy_return, repeat(data), params, repeat(freq), 
                    repeat(start), repeat(end), repeat(strategy), leave=False)
        res = pd.DataFrame(res, columns=['params', 'returns'])
        res.sort_values(by='returns', ascending=False, inplace=True)
        res.reset_index(drop=True, inplace=True)
        best = res.iloc[0]
    except:
        return
    
    return start, end, best['params'], best['returns'] 


def get_optimized_args(ticker, strategy, parameters):

    # load data
    conn = sql.connect(sql_path)
    data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
    conn.close()
    for col in ['open', 'close', 'high', 'low']:
        data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
        
    # get weekly dates
    sat_index = data_sql.resample('W-Sat').last().index
    start_dates = []
    end_dates = []
    for i in range(1, len(sat_index)-1):
        start_dates.append(sat_index[i])
        end_dates.append(sat_index[i+1])
        
    # get optimized params
    optim_params = p_umap(get_optim_param, 
                          repeat(data_sql), 
                          repeat(parameters), 
                          repeat(freq), 
                          start_dates, 
                          end_dates, 
                          repeat(strategy), 
                          leave=False,
                          desc=f'{strategy}-optimizing')
    try:
        opt_df = pd.DataFrame(optim_params, columns=['starts', 'ends', 'params', 'returns'])
    except:
        return (strategy, None, np.nan, np.nan, np.nan)
    opt_df.sort_values(by='starts', ascending=True, inplace=True)
    opt_df.reset_index(inplace=True, drop=True)
    opt_df['param1'] = opt_df.params.apply(lambda x: x[0] if x is not None else None)
    opt_df['param2'] = opt_df.params.apply(lambda x: x[1] if x is not None else None)
    
    n_periods = [i for i in range(1, int(len(start_dates)*0.7), 1)]
    results = []
    for n in tqdm(n_periods, leave=False, desc=f'{strategy}-rolling periods'):
        param1_roll_median = opt_df.param1.iloc[:-1].rolling(n).median()
        param1_roll_mean = opt_df.param1.iloc[:-1].rolling(n).mean()
        param1_roll_mode = opt_df.param1.iloc[:-1].rolling(n).apply(lambda x: mode(x)[0])

        param2_roll_median = opt_df.param2.iloc[:-1].rolling(n).median()
        param2_roll_mean = opt_df.param2.iloc[:-1].rolling(n).mean()
        param2_roll_mode = opt_df.param2.iloc[:-1].rolling(n).apply(lambda x: mode(x)[0])

        # median
        param1_p = param1_roll_median.values
        param2_p = param2_roll_median.values
        params = [*zip(param1_p, param2_p)]
        test_returns =  p_map(get_strategy_return, repeat(data_sql), params, 
                              repeat(freq), start_dates[-len(params):], 
                              end_dates[-len(params):], repeat(strategy),
                              leave=False, desc='median')
        test_returns = pd.DataFrame(test_returns, columns=['params', 'returns'])
        median_winrate = test_returns.returns.apply(np.sign).value_counts(normalize=True)[1]
        median_ret = test_returns.returns.mean()
        results.append(('median', n, median_winrate, median_ret))

        # mean
        param1_p = param1_roll_mean.values
        param2_p = param2_roll_mean.values
        params = [*zip(param1_p, param2_p)]
        test_returns =  p_map(get_strategy_return, repeat(data_sql), params, 
                              repeat(freq), start_dates[-len(params):],  
                              end_dates[-len(params):], repeat(strategy),
                              leave=False, desc='mean')
        test_returns = pd.DataFrame(test_returns, columns=['params', 'returns'])
        mean_winrate = test_returns.returns.apply(np.sign).value_counts(normalize=True)[1]
        mean_ret = test_returns.returns.mean()
        results.append(('mean', n, mean_winrate, mean_ret))

        # mode
        param1_p = param1_roll_mode.values
        param2_p = param2_roll_mode.values
        params = [*zip(param1_p, param2_p)]
        test_returns =  p_map(get_strategy_return, repeat(data_sql), params, 
                              repeat(freq), start_dates[-len(params):], 
                              end_dates[-len(params):], repeat(strategy),
                              leave=False, desc='mode')
        test_returns = pd.DataFrame(test_returns, columns=['params', 'returns'])
        mode_winrate = test_returns.returns.apply(np.sign).value_counts(normalize=True)[1]
        mode_ret = test_returns.returns.mean()
        results.append(('mode', n, mode_winrate, mode_ret))
        
    res_df = pd.DataFrame(results, columns = ['agg_methods', 'n_periods', 'win_rates', 'returns'])
    res_df.sort_values(by='returns', ascending=False, inplace=True)
    res_df.reset_index(inplace=True, drop=True)
    
    pos_ret = res_df.query('returns > 0')
    if len(pos_ret) > 0:
        best_agg = pos_ret.loc[0].agg_methods
        best_n = pos_ret.loc[0].n_periods
        winrate = pos_ret.loc[0].win_rates
        ret = pos_ret.loc[0].returns
        return (strategy, best_agg, best_n, winrate, ret)
    else:
        return (strategy, None, np.nan, np.nan, np.nan)


def main():
    for i, v in enumerate(my_assets):
        print(i, '|', v)

    global freq, sql_path
    freq = 'H1'
    sql_path = f'PriceData_{freq}.db'
    strategies = [
        'SMA', 'EMA', 'KAMA', 'DEMA', 'TEMA', 
        'TRIMA', 'HMA', 'WMA', 'ZLEMA', 'TSF',
        'WILDERS', 'SLOPE', 'TRIX'
        ]
    parameters = []
    param_ranges = range(50, 501, 50)
    for i in param_ranges:
        for j in param_ranges:
            if i < j: parameters.append((i,j))
    print(f'Parameters ranges: {param_ranges}')
    print(f'Parameters length: {len(parameters)}')
    
    ticker_args = {}
    for ticker in tqdm(my_assets, desc='tickers'):
        res = t_map(get_optimized_args, repeat(ticker), strategies, repeat(parameters), desc=ticker, leave=False)
        res_df = pd.DataFrame(res, columns=['strategy', 'agg', 'n_periods', 'winrates', 'rets'])
        res_df.sort_values(by='rets', ascending=False, inplace=True)
        res_df.reset_index(drop=True, inplace=True)
        ticker_args[ticker] = res_df.copy(deep=True)

        fname = 'best_args_500_50.xlsx'
        with pd.ExcelWriter(fname, mode='w') as writer:  
            for k, v in ticker_args.items():
                v.to_excel(writer, sheet_name=k.replace('/','_'), index=False)
        tqdm.write(f'{fname} is saved.')


if __name__ == '__main__': 
    main()