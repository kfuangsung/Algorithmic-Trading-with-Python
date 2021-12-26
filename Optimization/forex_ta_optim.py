import os 
import sqlite3
import sys
import concurrent
import json
from backtester import *
from tqdm import tqdm
from fxcmtoken import major_forex_pairs, time_frame
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_data(ticker, freq, split, main_path='PriceData'):
    db_path = os.path.join(main_path, f'PriceData_{freq}.db')
    conn = sqlite3.connect(db_path)
    data = pd.read_sql(f"SELECT * FROM '{ticker}'", conn, parse_dates=['date'], index_col=['date'])
    conn.close()
    if split is True:
        train_data, test_data = split_data(data)
        return train_data, test_data
    else:
        return data, None

def split_data(data, test_size=0.048):
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    # slice date for faster training --> too long data is not helping anyway(some data is too old)
    train_data = train_data.iloc[-1000:]
    return train_data, test_data

def save_db(df, ticker, freq, indicator):
    save_folder = 'ta_optimize'
    db_folder = 'database'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(os.path.join(save_folder,db_folder)):
        os.mkdir(os.path.join(save_folder, db_folder))
    conn = sqlite3.connect(os.path.join(save_folder, db_folder, f'{indicator}_optimize_{freq}.db'))
    name = ticker
    df.to_sql(name=f"'{name}'", con=conn, index=True, if_exists='replace')
    conn.close()
    
def save_txt(df, ticker, freq, indicator):
    save_folder = 'ta_optimize'
    txt_folder ='text'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(os.path.join(save_folder,txt_folder)):
        os.mkdir(os.path.join(save_folder, txt_folder))
    with open(os.path.join(save_folder,txt_folder,f'{indicator}_optimize.txt'), 'a') as f:
        f.write('-'*80 + '\n')
        f.write(f'***|{ticker}|{freq}|{datetime.now()}|***\n')
        f.write(df.sort_values(by='return', ascending=sort_ascend).set_index('params').iloc[:5].to_markdown() + '\n')
        f.write('-'*80 + '\n')
        
def save_json(params_dict, name):
    save_folder = 'ta_optimize'
    json_folder = 'json'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(os.path.join(save_folder,json_folder)):
        os.mkdir(os.path.join(save_folder, json_folder))
    with open(os.path.join(save_folder,json_folder,f"{name}.json"), 'w') as f:
        json.dump(params_dict, f, indent=4, sort_keys=True)
        
# ------------------------------------------------------------------------------------------------------
class MACD_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = MACDBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'MACD')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------------
class RSI_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, window  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = RSIBacktester(data=data, freq=freq, window=window, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, window, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        windows_range = range(5, 100, 5)
        for i in windows_range:
            args_list.append((ticker, freq, (i)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'RSI')
        # return best params
        return int(results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------------
class SO_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = StochasticOscillatorBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'SO')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class BB_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = BollingerBandBacktester(data=data, freq=freq, window_dev=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        dev_range = range(1, 6, 1)
        for i in windows_range:
            for j in dev_range:
                args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'BB')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class AROON_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = AroonBacktester(data=data, freq=freq, window_threshold=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        threshold_range = [70]
        for i in windows_range:
            for j in threshold_range:
                args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'AROON')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class CCI_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = CCIBacktester(data=data, freq=freq, window=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            args_list.append((ticker, freq, i))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'CCI')
        return int(results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------
class ADX_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = ADXBacktester(data=data, freq=freq, window_threshold=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        threshold_range = [10]
        for i in windows_range:
            for j in threshold_range:
                args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ADX')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class AO_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = AwesomeOscillatorBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j:
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'AO')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class CMF_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = CMFBacktester(data=data, freq=freq, window_threshold=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        threshold_range = [0.2]
        for i in windows_range:
            for j in threshold_range:
                args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'CMF')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class DONCHIAN_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = DonchianChannelBacktester(data=data, freq=freq, window=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            args_list.append((ticker, freq, i))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'DONCHIAN')
        return int(results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------
class KELTNER_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = KeltnerChannelBacktester(data=data, freq=freq, windows_threshold=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        threshold_range = [0.6]
        for i in windows_range:
            for j in windows_range:
                for k in threshold_range:
                    args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'KELTNER')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class MFI_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = MFIBacktester(data=data, freq=freq, window=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            args_list.append((ticker, freq, i))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'MFI')
        return int(results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------
class STOCHRSI_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = StochRSIBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        smooth_range = range(5, 21, 5)
        for i in windows_range:
            for j in smooth_range:
                for k in smooth_range:
                    args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'STOCHRSI')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class TRIX_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = TRIXBacktester(data=data, freq=freq, window=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            args_list.append((ticker, freq, i))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'TRIX')
        return int(results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------
class TSI_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = TSIBacktester(data=data, freq=freq, windows_threshold=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        threshold_range = [20]
        for i in windows_range:
            for j in windows_range:
                for k in threshold_range:
                    if i < j:
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'TSI')
        return results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params']
# ------------------------------------------------------------------------------------------------
class WILLIAMR_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = WilliamsRBacktester(data=data, freq=freq, window=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            args_list.append((ticker, freq, i))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'WILLIAMR')
        return int(results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------
class ICHIMOKU_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = IchimokuBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j<k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
    
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ICHIMOKU')
        return (results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]['params'])
# ------------------------------------------------------------------------------------------------
class SMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = SMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'SMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class EMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = EMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'EMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class DEMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = DEMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'DEMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class KAMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = KAMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'KAMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class MIDPOINT_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = MIDPOINTBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'MIDPOINT')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class MIDPRICE_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = MIDPRICEBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'MIDPRICE')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class TEMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = TEMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'TEMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class TRIMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = TRIMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'TRIMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class WMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        # print(data.index[-1])
        backtest = WMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'WMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class LINREG_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = LinearRegressionBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'LINREG')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class TSF_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows  = args[0], args[1], args[2]
        data, _ = load_data(ticker, freq, split=split)
        backtest = TSFBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                if i < j: 
                    args_list.append((ticker, freq, (i, j)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params, 'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'TSF')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeSMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeSMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeSMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeEMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeEMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeEMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeKAMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeKAMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeKAMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeMIDPOINT_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeMIDPOINTBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeMIDPOINT')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeMIDPRICE_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeMIDPRICEBacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeMIDPRICE')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeTRIMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeTRIMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeTRIMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------
class ThreeWMA_optimize:
    @staticmethod
    def get_results(args):
        ticker, freq, windows= args[0], args[1], args[2],
        data, _ = load_data(ticker, freq, split=split)
        backtest = ThreeWMABacktester(data=data, freq=freq, windows=windows, risk_free_rate=0)
        backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
        ret = backtest.ratio_df.loc['Portfolio', 'ExpectedReturn(%)']
        std = backtest.ratio_df.loc['Portfolio', 'StandardDeviation(%)']
        sharpe = backtest.ratio_df.loc['Portfolio', 'SharpeRatio']
        dd = backtest.ratio_df.loc['Portfolio', 'MaxDrawdown(%)']
        return (ticker, windows, ret, std, sharpe, dd)
    
    @staticmethod
    def get_args(ticker, freq):
        args_list = []
        # windows_range = range(10, 301, 10)
        for i in windows_range:
            for j in windows_range:
                for k in windows_range:
                    if i < j and j < k: 
                        args_list.append((ticker, freq, (i, j, k)))
        return args_list
            
    @staticmethod            
    def save_results(results, ticker, freq):
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
        results_df = pd.DataFrame(data={'tickers':tickers,'params':params,  'return':returns, 
                                        'stdev':stdevs, 'sharpe':sharpes, 'maxdd':maxdds})
        save_txt(results_df, ticker, freq, 'ThreeWMA')
        sorted_df = results_df.sort_values(by='return', ascending=sort_ascend).iloc[0]
        return sorted_df['params']
# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        ticker = sys.argv[1]
        freq = sys.argv[2]
        indicator = sys.argv[3]
        test_split = sys.argv[4]
        max_range = int(sys.argv[5])
        step = int(sys.argv[6])
        inverse = sys.argv[7]
    except:
        print("Usage: forex_ta_optim.py {ticker(s)} {time frame} {indicator} {test split?} {max range} {step} {inverse?}")
        sys.exit()
        
    indicator_func = {
                    #   'MA':MA_optimize,
                    #   'MACD':MACD_optimize,
                    #   'RSI':RSI_optimize,
                    #   'SO':SO_optimize,
                    #   'BB':BB_optimize,
                    #   'AROON':AROON_optimize,
                    #   'CCI':CCI_optimize,
                    #   'ADX':ADX_optimize,
                    #   'AO':AO_optimize,
                    #   'CMF':CMF_optimize,
                    #   'DONCHIAN':DONCHIAN_optimize,
                    #   'KELTNER':KELTNER_optimize,
                    #   'MFI':MFI_optimize,
                    #   'STOCHRSI':STOCHRSI_optimize,
                    #   'TRIX':TRIX_optimize,
                    #   'TSI':TSI_optimize,
                    #   'WILLIAMR':WILLIAMR_optimize,
                    #    'ICHIMOKU':ICHIMOKU_optimize,
                    'SMA' : SMA_optimize,
                    'EMA' : EMA_optimize,
                    # 'DEMA' : DEMA_optimize,
                    'KAMA' : KAMA_optimize,
                    'MIDPOINT' : MIDPOINT_optimize,
                    'MIDPRICE' : MIDPRICE_optimize,
                    # 'TEMA' : TEMA_optimize,
                    'TRIMA' : TRIMA_optimize,
                    'WMA' : WMA_optimize,
                    # 'LINREG' : LINREG_optimize,
                    # 'TSF' : TSF_optimize,
                    # 'ThreeSMA' : ThreeSMA_optimize,
                    # 'ThreeEMA' : ThreeEMA_optimize,
                    # 'ThreeKAMA' : ThreeKAMA_optimize,
                    # 'ThreeMIDPOINT' : ThreeMIDPOINT_optimize,
                    # 'ThreeMIDPRICE' : ThreeMIDPRICE_optimize,
                    # 'ThreeTRIMA' : ThreeTRIMA_optimize,
                    # 'ThreeWMA' : ThreeWMA_optimize,
                     }
    
    if freq not in time_frame:
        print(f"'{freq}' is NOT available.")
        print(f"Available time frames: {time_frame}")
        print("Usage: forex_ta_optim.py {ticker(s)} {time frame} {indicator} {test split?} {max range} {step} {inverse?}")
        sys.exit()
    
    if indicator not in list(indicator_func.keys())+['all']:
        print(f"'{indicator}' is NOT available.")
        print(f"Available indicators: {list(indicator_func.keys())}")
        print("Usage: forex_ta_optim.py {ticker(s)} {time frame} {indicator} {test split?} {max range} {step} {inverse?}")
        sys.exit()
        
    if test_split == 'yes'.lower():
        split = True
    elif test_split == 'no'.lower():
        split= False
    else:
        print("Incorrect test split. Please type 'yes' or 'no'.")
        sys.exit()
    
    if inverse == 'yes'.lower():
        sort_ascend = True
    elif inverse == 'no'.lower():
        sort_ascend = False
    else:
        print("Incorrect inverse. Please type 'yes' or 'no'.")
        sys.exit()
    
    num_workers = 6
    windows_range = range(step, max_range+1, step)
    
    if ticker == 'majorforex':
        if indicator == 'all':
            best_params = {}
            for ind in tqdm(indicator_func.keys(), leave=True, desc='All Indicators'):
                best_params[ind] = {}
                for ticker in tqdm(major_forex_pairs, leave=True, desc=ind):
                    args_list = indicator_func[ind].get_args(ticker, freq)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                        results = list(tqdm(ex.map(indicator_func[ind].get_results, args_list), total=len(args_list), leave=False, desc=ticker))
                    params = indicator_func[ind].save_results(results, ticker, freq)
                    best_params[ind][ticker] = params
                if inverse == 'no'.lower():
                    save_json(best_params, name=f"OPTIMIZE_params_{freq}_{max_range}_{step}")
                elif inverse == 'yes'.lower():
                    save_json(best_params, name=f"OPTIMIZE_params_{freq}_{max_range}_{step}_negative")
        else:
            best_params = {}
            for ticker in tqdm(major_forex_pairs, leave=True, desc=indicator):
                args_list = indicator_func[indicator].get_args(ticker, freq)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                    results = list(tqdm(ex.map(indicator_func[indicator].get_results, args_list), total=len(args_list), leave=False, desc=ticker))
                params = indicator_func[indicator].save_results(results, ticker, freq)
                best_params[ticker] = params     
            if inverse == 'no'.lower():
                save_json(best_params, name=f"OPTIMIZE_params_{freq}_{max_range}_{step}")
            elif inverse == 'yes'.lower():
                save_json(best_params, name=f"OPTIMIZE_params_{freq}_{max_range}_{step}_negative")
            
    else:
        if indicator == 'all':
            for ind in tqdm(indicator_func.keys(), leave=True, desc='All Indicators'):
                args_list = indicator_func[ind].get_args(ticker, freq)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                    results = list(tqdm(ex.map(indicator_func[ind].get_results, args_list), total=len(args_list), leave=False, desc=ticker))
                indicator_func[ind].save_results(results, ticker, freq)  
        else:
            args_list = indicator_func[indicator].get_args(ticker, freq)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                results = list(tqdm(ex.map(indicator_func[indicator].get_results, args_list), total=len(args_list), leave=False, desc=ticker))
            indicator_func[indicator].save_results(results, ticker, freq)