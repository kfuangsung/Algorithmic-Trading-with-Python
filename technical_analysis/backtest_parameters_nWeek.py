import os
import sys 
import gc
import pandas as pd 
import numpy as np 
import talib
import sqlite3 as sql
from itertools import repeat
from p_tqdm import p_umap, t_map
sys.path.append('../')
from backtester import * 
from fxcmtoken import my_assets


def get_strategy_return(data, signal, param, freq):    
    if len(data) == 0: return param, np.nan
    
    backtester = IterativeBacktester(data=data, signals=signal, freq=freq)
    backtester.backtest(progress_bar=False)

    return param, backtester.return_df.loc['TotalReturn', 'Portfolio']


def get_optim_param(data, signal, param, freq, start_date, end_date): 
    res = p_umap(get_strategy_return, repeat(data.copy(deep=True)), signal, param, repeat(freq), leave=False)
    optim_param = pd.DataFrame(res).sort_values(by=1, ascending=False).iloc[0,0]
    
    return (start_date, end_date), optim_param


def get_mama_optim(input_data, freq, param_range):

    mama_periods = np.arange(param_range[0], param_range[1], param_range[2])

    data = input_data.copy(deep=True)
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    for n in mama_periods:
        data[f"mama_{n:.1f}"], data[f"fama_{n:.1f}"] = talib.MAMA(data.ht, fastlimit=n, slowlimit=n/10)
        data[f'signals_{n:.1f}'] = np.where(data[f"mama_{n:.1f}"] > data[f"fama_{n:.1f}"], 1, -1) 
    data.dropna(axis=0, inplace=True)

    sat_index = data.resample('W-Sat').last().index
    datasets = []
    start_dates = []
    end_dates = []
    params = []
    signals = []

    for i in range(n_week, len(sat_index)-1):
        start_dates.append(sat_index[i])
        end_dates.append(sat_index[i+1])
        data_ = data.loc[sat_index[i-n_week]:sat_index[i],:]
        datasets.append(data_.copy(deep=True))
        sub_params = []
        sub_signals = []
        for n in mama_periods:
            sub_params.append(n)
            sub_signals.append(data_.loc[:,f"signals_{n:.1f}"])
        params.append(sub_params)
        signals.append(sub_signals)
        
    mama_optim = t_map(get_optim_param, datasets, signals, params, repeat(freq), start_dates, end_dates, leave=False)
    mama_optim = pd.DataFrame(mama_optim)
    mama_optim.columns = ['dates', 'mama_param']
    mama_optim.set_index('dates', inplace=True)

    gc.collect()
    
    return mama_optim


def get_slope_optim(input_data, freq, param_range):

    slope_periods = np.arange(param_range[0], param_range[1], param_range[2])

    data = input_data.copy(deep=True)
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    for n in slope_periods:
        data[f'signals_{n:.0f}'] = talib.LINEARREG_SLOPE(data.ht, timeperiod=n).apply(np.sign)
    data.dropna(axis=0, inplace=True)

    sat_index = data.resample('W-Sat').last().index
    datasets = []
    start_dates = []
    end_dates = []
    params = []
    signals = []

    for i in range(n_week, len(sat_index)-1):
        start_dates.append(sat_index[i])
        end_dates.append(sat_index[i+1])
        data_ = data.loc[sat_index[i-n_week]:sat_index[i],:]
        datasets.append(data_.copy(deep=True))
        sub_params = []
        sub_signals = []
        for n in slope_periods:
            sub_params.append(n)
            sub_signals.append(data_.loc[:,f"signals_{n:.0f}"])
        params.append(sub_params)
        signals.append(sub_signals)
        
    slope_optim = t_map(get_optim_param, datasets, signals, params, repeat(freq), start_dates, end_dates, leave=False)
    slope_optim = pd.DataFrame(slope_optim)
    slope_optim.columns = ['dates', 'slope_param']
    slope_optim.set_index('dates', inplace=True)

    gc.collect()
    
    return slope_optim


def get_tsf_optim(input_data, freq, param_range):
    
    tsf_periods = np.arange(param_range[0], param_range[1], param_range[2])

    data = input_data.copy(deep=True)
    data['ht'] = talib.HT_TRENDLINE(data.close)
    data.dropna(axis=0, inplace=True)
    for n in tsf_periods:
        data[f'tsf_{n:.0f}'] = talib.TSF(data.ht, n)
        data[f'signals_{n:.0f}'] = np.where(data.ht > data[f'tsf_{n:.0f}'], 1, -1)
    data.dropna(axis=0, inplace=True)

    sat_index = data.resample('W-Sat').last().index
    datasets = []
    start_dates = []
    end_dates = []
    params = []
    signals = []

    for i in range(n_week, len(sat_index)-1):
        start_dates.append(sat_index[i])
        end_dates.append(sat_index[i+1])
        data_ = data.loc[sat_index[i-n_week]:sat_index[i],:]
        datasets.append(data_.copy(deep=True))
        sub_params = []
        sub_signals = []
        for n in tsf_periods:
            sub_params.append(n)
            sub_signals.append(data_.loc[:,f"signals_{n:.0f}"])
        params.append(sub_params)
        signals.append(sub_signals)
        
    tsf_optim = t_map(get_optim_param, datasets, signals, params, repeat(freq), start_dates, end_dates, leave=False)
    tsf_optim = pd.DataFrame(tsf_optim)
    tsf_optim.columns = ['dates', 'tsf_param']
    tsf_optim.set_index('dates', inplace=True)

    gc.collect()
    
    return tsf_optim


def get_aggregate_return(data, freq, ds, params):
    data_ = data.loc[:ds[1],:].copy(deep=True)
    if len(data_) == 0:  return ds[0], np.nan
    
    data_['ht'] = talib.HT_TRENDLINE(data_.close)

    data_["mama"], data_["fama"] = talib.MAMA(data_.ht, fastlimit=params[0], slowlimit=params[0]/10)
    data_['signals_mama'] = np.where(data_[f"mama"] > data_["fama"], 1, -1) 

    data_['signals_slope'] = talib.LINEARREG_SLOPE(data_.ht, timeperiod=params[1]).apply(np.sign)

    data_['tsf'] = talib.TSF(data_.ht, params[2])
    data_['signals_tsf'] = np.where(data_.ht > data_['tsf'], 1, -1)
    
    data_.dropna(axis=0, inplace=True)
    data_ = data_.loc[ds[0]:ds[1],:]
    if len(data_) == 0:  return ds[0], np.nan
    
    data_['signals_aggregate'] = data_[['signals_mama', 'signals_slope', 'signals_tsf']].mode(axis=1)
    
    backtester = IterativeBacktester(data=data_, signals=data_.signals_aggregate, freq=freq)
    backtester.backtest(progress_bar=False)
    
    return ds[0], backtester.return_df.loc['TotalReturn', 'Portfolio']


def get_backtest_returns(ticker, freq, params_ranges):
    freq = 'H1'
    sql_path = f'/home/kachain/python_projects/algorithmic_trading/fxcmpy_trader/PriceData_{freq}.db'
    conn = sql.connect(sql_path)
    data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
    for col in ['open', 'close', 'high', 'low']:
        data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
    conn.close()
        
    mama_optim = get_mama_optim(data_sql, freq, params_ranges['mama'])
    slope_optim = get_slope_optim(data_sql, freq, params_ranges['slope'])
    tsf_optim = get_tsf_optim(data_sql, freq, params_ranges['tsf'])
    
    best_parameters = pd.DataFrame()
    best_parameters = best_parameters.join(mama_optim, how='outer')
    best_parameters = best_parameters.join(slope_optim, how='outer')
    best_parameters = best_parameters.join(tsf_optim, how='outer')
    
    best_parameters.dropna(axis=0, inplace=True)
    best_parameters['params'] = best_parameters.values.tolist()
    dates = best_parameters.index.values
    parameters = best_parameters.params.values
    
    res_ = p_umap(get_aggregate_return, repeat(data_sql), repeat(freq), dates, parameters, leave=False)
    res_df = pd.DataFrame(res_)
    res_df.columns = ['dates', ticker]
    res_df.set_index('dates', inplace=True)
    
    gc.collect()
    
    return res_df


def main():
    args = sys.argv
    if len(args) < 2: raise ValueError("Usage python backtest_parameters_nWeek.py [number of week]")
    
    if not isinstance(int(args[1]), (int, float)): raise ValueError("[number of week] must be a number") 
    
    global n_week
    n_week = int(args[1])
    
    freq = 'H1'
    params_ranges = {
        'mama': (0.1, 1, 0.1),
        'slope': (50, 501, 50),
        'tsf': (50, 501, 50)
    }
    
    print("*** Parameter ranges ***")
    for k, v in params_ranges.items():
        print(f"\t{k.ljust(8)} --- {v}")
        
    print(f"Number of weeks: {n_week}")
        
    ret_ = t_map(get_backtest_returns, my_assets, repeat(freq), repeat(params_ranges))
    
    backtest_results = pd.DataFrame()
    for frame in ret_:
        backtest_results = backtest_results.join(frame, how='outer')    
    backtest_results.sort_index(ascending=True, inplace=True)
    
    save_folder = 'backtest_parameters'
    fname = f'backtest_params_results_{int(n_week)}week.csv'
    os.makedirs(save_folder, exist_ok=True)
    backtest_results.to_csv(os.path.join(save_folder, fname))
    
    print(f"{os.path.join(save_folder, fname)} is saved.")
    
if __name__ == '__main__':
    main()