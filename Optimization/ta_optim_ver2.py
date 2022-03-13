import os 
import sys
import warnings
import sqlite3
import concurrent
import json
from backtester import *
from tqdm import tqdm
from fxcmtoken import major_forex_pairs, time_frame
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

class BaseClass:
        
    @staticmethod
    def load_data(ticker, freq, split_data, high_param, data_path='PriceData', train_length=120, test_size=0.012):
        db_path = os.path.join(data_path, f'PriceData_{freq}.db')
        conn = sqlite3.connect(db_path)
        data = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
        conn.close()
        
        def split_data(data, test_size):
            split_len = int(len(data) * test_size)
            test_len = split_len + high_param
            test_data = data[-test_len:]
            train_data = data[-(train_length+split_len+high_param):-split_len]

            return train_data, test_data
        
        if split_data:
            data, _ = split_data(data, test_size)
        else:
            data = data.iloc[-(train_length+high_param):] # slice data
            
        return data
    
    @staticmethod
    def get_result(ticker, freq, data, params, backtester):
        backtest = backtester(data, freq=freq, params=params, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, params, ret, std, sharpe, dd)
    
    @staticmethod
    def sort_results(results):
        tickers = []
        params = []
        returns = []
        stdevs = []
        sharpes = []
        maxdds = []
        for res in results:
            tickers.append(res[0])
            params.append(res[1])
            returns.append(res[2])
            stdevs.append(res[3])
            sharpes.append(res[4])
            maxdds.append(res[5])
        return  pd.DataFrame(data={ 'tickers':tickers,
                                    'params':params,  
                                    'return':returns, 
                                    'stdev':stdevs, 
                                    'sharpe':sharpes, 
                                    'maxdd':maxdds
                                    })
    
    @staticmethod
    def save_txt(results_df, ticker, freq, name, inverse):
        save_folder = 'ta_optimize'
        txt_folder ='text'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(os.path.join(save_folder,txt_folder)):
            os.mkdir(os.path.join(save_folder, txt_folder))
        with open(os.path.join(save_folder,txt_folder,f'{name}_optimize.txt'), 'a') as f:
            f.write('-'*80 + '\n')
            f.write(f'***|{ticker}|{freq}|{datetime.now()}|***\n')
            f.write(results_df.sort_values(by='return', ascending=inverse).set_index('params').iloc[:5].to_markdown() + '\n')
            f.write('-'*80 + '\n')
    
    @staticmethod
    def get_best_params(results_df, inverse):
        return results_df.sort_values(by='return', ascending=inverse).iloc[0]['params']
    
    @staticmethod
    def save_json(params_dict, file_name):
        save_folder = 'ta_optimize'
        json_folder = 'json'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(os.path.join(save_folder,json_folder)):
            os.mkdir(os.path.join(save_folder, json_folder))
        with open(os.path.join(save_folder,json_folder,f"{file_name}.json"), 'w') as f:
            json.dump(params_dict, f, indent=4, sort_keys=True)
    
class Optimizer:
    
    @staticmethod
    def optim(args):
        ticker, freq, params = args
        high_param = max(params)
        data = BaseClass.load_data(ticker, freq, split_data, high_param)
        results = BaseClass.get_result(ticker, freq, data, params, backtester)
        return results

# ---------------------------------------------------------------------------------------------------------------------------------------------

class MAOptimizers:
    
    @staticmethod
    def get_args(ticker, freq, params_range):
        args_list = []
        for i in params_range:
            for j in params_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
    
# -------------------------------------------------------------------------------------------------------------------------------------------

class ThreeIndsOptimizer:
    
    @ staticmethod
    def get_args(ticker, freq, params_range):
        # (ema, rsi, bb)
        args_list = []
        for i in params_range:
            for j in params_range:
                for k in params_range:
                    p = (i, int(j/params_max*100), k)
                    args_list.append((ticker, freq, p))
        return args_list

# ------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    usage = "Usage:python forex_ta_optim.py {Ticker(s)} {TimeFrame} {Indicator} {TestSplit?} {ParamsMax} {ParamsStep} {Inverse?}"
    try:
        ticker = sys.argv[1]
        freq = sys.argv[2]
        indicator = sys.argv[3]
        test_split = sys.argv[4]
        params_max = int(sys.argv[5])
        params_step = int(sys.argv[6])
        inverse = sys.argv[7]
    except:
        print(usage)
        sys.exit()
        
    if freq not in time_frame:
        print(f"'{freq}' is NOT available.")
        print(f"Available time frames: {time_frame}")
        print(usage)
        sys.exit()
        
    indicators_dict = {
        'SMA' : {'optim' : MAOptimizers, 'backtest' : SMABacktester},
        'EMA' : {'optim' : MAOptimizers, 'backtest' : EMABacktester},
        'KAMA' : {'optim' : MAOptimizers, 'backtest' : KAMABacktester},
        'MIDPOINT' : {'optim' : MAOptimizers, 'backtest' : MIDPOINTBacktester},
        'MIDPRICE' : {'optim' : MAOptimizers, 'backtest' : MIDPRICEBacktester},
        'TRIMA' : {'optim' : MAOptimizers, 'backtest' : TRIMABacktester},
        'WMA' : {'optim' : MAOptimizers, 'backtest' : WMABacktester},
        'MultiMA' : {'optim' : MAOptimizers, 'backtest' : MultiMABacktester},
        'ThreeInds' : {'optim' : ThreeIndsOptimizer, 'backtest': ThreeIndicatorsBacktester}
                      }
    
    print('-'*50)
    print(f"{ticker} | {freq}")
    if ticker == 'majorforex':
        print(major_forex_pairs)
        
    if indicator not in list(indicators_dict.keys())+['all']:
        print(f"'{indicator}' is NOT available.")
        print(f"Available indicators: {list(indicators_dict.keys())}")
        print(usage)
        sys.exit()
    print(f"Indicator: {indicator}")
    if indicator == 'all':
        print(indicators_dict.keys())
    
    if test_split == 'yes'.lower():
        split_data = True
        print("Split data into Train/Test.")
    elif test_split == 'no'.lower():
        split_data = False
        print("No data split.")
    else:
        print("Incorrect response. Please type 'yes' or 'no'.")
        sys.exit()
    
    if inverse == 'yes'.lower():
        inverse = True
        print("Sorting by Ascending order.")
    elif inverse == 'no'.lower():
        inverse = False
        print("Sorting by Descending order.")
    else:
        print("Incorrect response. Please type 'yes' or 'no'.")
        sys.exit()
        
    num_workers = 6
    if params_step == 1:
        params_start = 2
    else:
        params_start = params_step
        
    params_range = range(params_start, params_max + 1, params_step)
    print(f"Parameters Range: {params_range}")
    print('-'*50)
    
    if ticker == 'majorforex':
        if indicator == 'all':
            params_dict = {}
            for name in tqdm(indicators_dict.keys(), leave=True, desc='All Indicators'):
                indicator_optim = indicators_dict[name]['optim']
                backtester = indicators_dict[name]['backtest']
                params_dict[name] = {}
                for ticker in tqdm(major_forex_pairs, leave=True, desc=name):
                    args_list = indicator_optim.get_args(ticker, freq, params_range)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                        results = list(tqdm(ex.map(Optimizer.optim, args_list), total=len(args_list), leave=False, desc=ticker))
                    results_df = BaseClass.sort_results(results)
                    BaseClass.save_txt(results_df, ticker, freq, name, inverse)
                    results_params = BaseClass.get_best_params(results_df, inverse)
                    params_dict[name][ticker] = results_params
                    # print(params_dict)
                if inverse == True:
                    file_name=f"OPTIMIZE_params_{freq}_{params_max}_{params_step}_negative"
                else:
                    file_name=f"OPTIMIZE_params_{freq}_{params_max}_{params_step}"
                BaseClass.save_json(params_dict, file_name)
        else:
            name = indicator
            indicator_optim = indicators_dict[name]['optim']
            backtester = indicators_dict[name]['backtest']
            params_dict = {}
            params_dict[name] = {}
            for ticker in tqdm(major_forex_pairs, leave=True, desc=name):
                args_list = indicator_optim.get_args(ticker, freq, params_range)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                    results = list(tqdm(ex.map(Optimizer.optim, args_list), total=len(args_list), leave=False, desc=ticker))
                # print(results)
                results_df = BaseClass.sort_results(results)
                BaseClass.save_txt(results_df, ticker, freq, name, inverse)
                results_params = BaseClass.get_best_params(results_df, inverse)
                params_dict[name][ticker] = results_params
                # print(params_dict)
            if inverse == True:
                file_name=f"OPTIMIZE_params_{freq}_{params_max}_{params_step}_negative"
            else:
                file_name=f"OPTIMIZE_params_{freq}_{params_max}_{params_step}"
            BaseClass.save_json(params_dict, file_name)
                    
    else:
        if indicator == 'all':
            best_params = {}
            for name in tqdm(indicators_dict.keys(), leave=True, desc='All Indicators'):
                indicator_optim = indicators_dict[name]['optim']
                backtester = indicators_dict[name]['backtest']
                best_params[name] = {}
                args_list = indicator_optim.get_args(ticker, freq, params_range)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                        results = list(tqdm(ex.map(Optimizer.optim, args_list), total=len(args_list), leave=False, desc=ticker))
                results_df = BaseClass.sort_results(results)
                BaseClass.save_txt(results_df, ticker, freq, name, inverse)
                results_params = BaseClass.get_best_params(results_df, inverse)
                best_params[name][ticker] = results_params
                if inverse == True:
                    file_name=f"OPTIMIZE_params_{freq}_{params_max}_{params_step}_negative"
                else:
                    file_name=f"OPTIMIZE_params_{freq}_{params_max}_{params_step}"
                BaseClass.save_json(best_params, file_name)
                
        else:
            name = indicator
            indicator_optim = indicators_dict[name]['optim']
            backtester = indicators_dict[name]['backtest']
            args_list = indicator_optim.get_args(ticker, freq, params_range)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                        results = list(tqdm(ex.map(Optimizer.optim, args_list), total=len(args_list), leave=False, desc=ticker))
            results_df = BaseClass.sort_results(results)
            BaseClass.save_txt(results_df, ticker, freq, name, inverse)