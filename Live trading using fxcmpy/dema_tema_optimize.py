import sys 
import json
import empyrical as emp 
import pandas as pd 
import numpy as np 
import bottleneck as bn 
import talib 
import sqlite3 as sql
from itertools import repeat
from p_tqdm import p_umap, t_map
sys.path.append('../')
from backtester import * 
from fxcmtoken import my_assets

def get_strategy_return(data_sql, n_period, freq):    
  
    data_sql = data_sql.assign(
        medprice=lambda x: talib.MEDPRICE(x.high, x.low),
        dema=lambda x: talib.DEMA(x.medprice, n_period),
        tema=lambda x: talib.TEMA(x.medprice, n_period),
        signal=lambda x: bn.push(np.where(x.tema > x.dema, 1, np.where(x.tema < x.dema, -1, np.nan)), axis=0)
    )
    data_sql.dropna(axis=0, inplace=True)
    data = data_sql[['bidopen', 'bidclose', 'bidhigh', 'bidlow',
                     'askopen', 'askclose', 'askhigh', 'asklow',
                     'tickqty']]
    signal = data_sql.signal
    backtester = IterativeBacktester(data=data, signals=signal, freq=freq)
    backtester.backtest(progress_bar=False)
    
    month_index = data_sql.resample('M').last().index
    num_period_in_month = []
    for i in range(len(month_index)-2):
        num_period_in_month.append(data_sql.loc[month_index[i]:month_index[i+1]].shape[0])
    num_annual = int(np.mean(num_period_in_month) * 12)
    
    simple_ret = emp.simple_returns(backtester.portfolio_df.Portfolio)
    factor_ret = emp.simple_returns(backtester.portfolio_df.Benchmark)
    
    final_ret = emp.cum_returns_final(simple_ret, starting_value=0)
    annual_ret = emp.annual_return(simple_ret, annualization=num_annual)
    sharpe_ratio = emp.sharpe_ratio(simple_ret, risk_free=0, annualization=num_annual)
    sortino_ratio = emp.sortino_ratio(simple_ret, required_return=0, annualization=num_annual)
    omega_ratio = emp.omega_ratio(simple_ret, risk_free=0, required_return=0, annualization=num_annual)
    calmar_ratio = emp.calmar_ratio(simple_ret, annualization=num_annual)
    maxdd = emp.max_drawdown(simple_ret)
    alpha = emp.alpha(simple_ret, factor_returns=factor_ret, risk_free=0, annualization=num_annual)

    return n_period, final_ret, annual_ret, sharpe_ratio, sortino_ratio, omega_ratio, calmar_ratio, maxdd, alpha


def get_optim_param(ticker, freq, params): 
    # sql_path = f'PriceData_{freq}.db'
    sql_path = f'/home/kachain/python_projects/fxcmpy/PriceData/PriceData_{freq}.db'
    conn = sql.connect(sql_path)
    data_sql = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
    for col in ['open', 'close', 'high', 'low']:
        data_sql[col] = data_sql[['bid'+col, 'ask'+col]].mean(axis=1)
    conn.close()
    
    res = p_umap(get_strategy_return, repeat(data_sql.copy(deep=True)), params, repeat(freq), desc=ticker)
    res_df = pd.DataFrame(res, columns=['n_periods', 'final_returns', 'annual_returns',
                                        'sharpe_ratios', 'sortino_ratios', 'omega_ratios',
                                        'calmar_ratios', 'maxdds', 'alphas'])
    return res_df
    
    
def get_best_parameter(ticker, freq, params):
    res = get_optim_param(ticker, freq, params)
    res = res.rolling(int(len(params)*0.1)).mean().dropna(axis=0)
    res.sort_values(by='omega_ratios', ascending=False, inplace=True)
    return ticker, int(res.n_periods.values[0])


def main():
    freq = 'H1'
    max_param = 500
    param_range = range(2, max_param+1, 1)
    params = [i for i in param_range]
    print("OPTIMIZATION --- DEMA TEMA crossover")
    print(f"Freq: {freq}")
    print(f"Range: {param_range}")
    optim_params = t_map(get_best_parameter, my_assets, repeat(freq), repeat(params))
    optim_params_dict = {k:int(v) for k, v in optim_params}
    
    fname = "my_parameters.json"
    with open(fname, "w") as f:
        json.dump(optim_params_dict, f, indent=4)
    print(f"{fname} is saved.")
    
    
if __name__ == '__main__':
    main()
