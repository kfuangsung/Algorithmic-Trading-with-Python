import numpy as np
import pandas as pd 
import ta
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from tabulate import tabulate
plt.style.use('seaborn-whitegrid')
pd.options.mode.chained_assignment = None

#--------------------------------------------------------------------------------------------------#

class IterativeBacktester():
    def __init__(self, data, signals, freq, risk_free_rate=0.01, stop_loss=None):
        self.data = data
        self.signals = signals
        self.freq = freq
            # minutes: m1, m5, m15 and m30,
            # hours: H1, H2, H3, H4, H6 and H8,
            # one day: D1,
            # one week: W1,
            # one month: M1.
        self.init_money = 1
        self.current_money = self.init_money
        self.portfolio_values = [self.init_money]
        self.current_pos = 0
        self.unit_cost = None
        self.holding_amount = None
        self.passive_values = [self.init_money]
        self.passive_amount = None
        self.passive_cost = None
        self.portfolio_df = None
        self.return_df = None
        self.std_df = None
        self.total_timedelta = None
        self.total_days = None
        self.risk_free_rate = risk_free_rate
        self.ratio_df = None
        self.multiplier = None
        self.drawdown = None
        self.stop_loss = stop_loss
        self.trade_info = {}
        self.is_in_stop_loss = False
        self.last_stop_signal = None
        
    def compute_amount(self, money, price):
        return money/price
    
    def compute_profit(self, cost, amount, bid, ask, is_long):
        if is_long is True:
            return amount * (bid - cost)
        elif is_long is False:
            return amount * (cost - ask)
    def init_passive(self, bid, ask):
        self.passive_cost = ask
        self.passive_amount = self.compute_amount(self.init_money, self.passive_cost)
    
    def update_passive(self, bid, ask):
        passive_profit = self.compute_profit(self.passive_cost, self.passive_amount, bid, ask, is_long=True)
        self.passive_values.append(self.init_money + passive_profit)
    
    def update_portfolio(self, bid, ask, is_long):
        profit = self.compute_profit(self.unit_cost, self.holding_amount, bid, ask, is_long)
        self.portfolio_values.append(self.current_money + profit)
        
    def open_long(self, bid, ask):
        self.unit_cost = ask
        self.holding_amount = self.compute_amount(self.current_money, self.unit_cost)
        self.update_portfolio(bid, ask, is_long=True)
        self.current_pos = 1
        
    def close_long(self, bid, ask):
        profit = self.compute_profit(self.unit_cost, self.holding_amount, bid, ask, is_long=True)
        self.current_money += profit
        self.unit_cost = None
        self.holding_amount = None
        self.current_pos = 0
        
    def open_short(self, bid, ask):
        self.unit_cost = bid
        self.holding_amount = self.compute_amount(self.current_money, self.unit_cost)
        self.update_portfolio(bid, ask, is_long=False)
        self.current_pos = -1
    
    def close_short(self, bid, ask):
        profit = self.compute_profit(self.unit_cost, self.holding_amount, bid, ask, is_long=False)
        self.current_money += profit
        self.unit_cost = None
        self.holding_amount = None
        self.current_pos = 0
        
    def check_stop_loss(self, signal, bid, ask):
       
        if self.stop_loss is None: 
            return False
        
         # afer stop loss was hit --> not open new position unless signal is changed
        if self.is_in_stop_loss and (self.last_stop_signal == signal):
            return True
        else:
            self.is_in_stop_loss = False
            self.last_stop_signal = None
        
        if self.trade_info == {}:
            self.trade_info['signal'] = signal
            self.trade_info['enter'] = {}
            self.trade_info['enter']['bid'] = bid
            self.trade_info['enter']['ask'] = ask
            self.trade_info['current'] = {}
            self.trade_info['current']['bid'] = bid
            self.trade_info['current']['ask'] = ask
            return False
        
        self.trade_info['current']['bid'] = bid
        self.trade_info['current']['ask'] = ask
        
        if self.trade_info['signal'] != signal:
            return False
        
        if signal == 1:
            self.trade_info['pnl'] = (self.trade_info['current']['bid'] / self.trade_info['enter']['ask']) - 1
        elif signal == -1:
            self.trade_info['pnl'] = -1 * ((self.trade_info['current']['ask'] / self.trade_info['enter']['bid']) - 1)
        else:
            return False
        
        if (self.trade_info['pnl']*100) <= -self.stop_loss:
            self.is_in_stop_loss = True
            self.last_stop_signal = self.trade_info['signal']
            self.trade_info = {}
            return True
        
    def backtest(self, progress_bar=True, is_notebook=True, leave=True):
        if is_notebook is True:
            iterate = tqdm_notebook(range(len(self.data)-1), disable=(not progress_bar), leave=leave)
        else:
            iterate = tqdm(range(len(self.data)-1), disable=(not progress_bar), leave=leave)
        for i in iterate:
            # buy at askopen in next row
            # sell at bidopen in next row
            signal = self.signals.iloc[i]
            next_row = self.data.iloc[i+1]
            bid = next_row['bidopen']
            ask = next_row['askopen']
            
            if i == 0: # get buy&hold portfolio as benchmark
                self.init_passive(bid, ask)
            self.update_passive(bid, ask)
            
            if self.check_stop_loss(signal, bid, ask):
                signal = 0
            
            if signal == 1:
                if self.current_pos == 0:
                    self.open_long(bid, ask)
                elif self.current_pos == -1:
                    # close current short first
                    self.close_short(bid, ask)
                    # then open long
                    self.open_long(bid, ask)
                elif self.current_pos == 1:
                    self.update_portfolio(bid, ask, is_long=True)
        
            if signal == -1:
                if self.current_pos == 0:
                    self.open_short(bid, ask)
                elif self.current_pos == 1:
                    # close long first
                    self.close_long(bid, ask)
                    # then open short
                    self.open_short(bid, ask)
                elif self.current_pos == -1:
                    self.update_portfolio(bid, ask, is_long=False)
    
            if signal == 0:
                if self.current_pos == 1:
                    # close long
                    self.close_long(bid, ask)
                elif self.current_pos == -1:
                    # close short
                    self.close_short(bid, ask)
                elif self.current_pos == 0:
                    pass
                self.portfolio_values.append(self.current_money)
        
        # compute returns/stdevs/drawdown
        self.get_total_time()
        self.get_multiplier()
        self.get_portfolio_df()
        self.get_return_df()
        self.get_std_df()
        self.get_drawdown()
        self.get_ratio_df()
        
    def get_total_time(self):
        # number of days
        self.total_timedelta = self.data.index[-1] - self.data.index[0]
        self.total_days = self.total_timedelta.total_seconds() / (3600*24)
        
    def get_portfolio_df(self):
        self.portfolio_df = pd.DataFrame(data={'Portfolio':self.portfolio_values, 'Benchmark':self.passive_values},
                                         index=self.data.index)
        self.portfolio_df['PortfolioReturns'] = np.log(self.portfolio_df['Portfolio']/self.portfolio_df['Portfolio'].shift(1))
        self.portfolio_df['BenchmarkReturns'] = np.log(self.portfolio_df['Benchmark']/self.portfolio_df['Benchmark'].shift(1))
        
    def get_return_df(self):
        ret_mean = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].mean(axis=0)
        ret_mean.rename(index={'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'}, inplace=True)
        return_df = pd.DataFrame(data={'TotalReturn':None, 'MonthlyReturn':None, 'AnnualReturn':None}, index=['Portfolio', 'Benchmark'])
        return_df.loc[:, 'TotalReturn'] = ret_mean * self.multiplier['total']
        return_df.loc[:, 'AnnualReturn'] = ret_mean * self.multiplier['annual']
        return_df.loc[:, 'MonthlyReturn'] = ret_mean * self.multiplier['monthly']
        return_df = return_df.T
        return_df['Alpha'] = return_df['Portfolio'] - return_df['Benchmark']
        self.return_df = return_df
        
    def get_std_df(self):
        ret_std = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].std()
        ret_std.rename(index={'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'}, inplace=True)
        std_df = pd.DataFrame(data={'TotalStdev':None, 'MonthlyStdev':None, 'AnnualStdev':None}, index=['Portfolio', 'Benchmark'])
        std_df.loc[:, 'TotalStdev'] = ret_std * np.sqrt(self.multiplier['total'])
        std_df.loc[:, 'AnnualStdev'] = ret_std * np.sqrt(self.multiplier['annual']) 
        std_df.loc[:, 'MonthlyStdev'] = ret_std * np.sqrt(self.multiplier['monthly'])
        std_df = std_df.T
        self.std_df = std_df
        
    def get_drawdown(self):
        cummax = self.portfolio_df[['Portfolio', 'Benchmark']].cummax()
        self.drawdown =  (self.portfolio_df[['Portfolio', 'Benchmark']] - cummax) / cummax
    
    def get_multiplier(self):
        self.multiplier = {'total':len(self.data),
                           'monthly':len(self.data)*(30/self.total_days),
                           'annual':len(self.data)*(365/self.total_days)} 
    
    def get_ratio_df(self):
        ratio_df = pd.DataFrame(index=['Portfolio', 'Benchmark'])
        # ratios are computed on returns 
        
        # Sharpe ratio 
        ann_mean = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].mean(axis=0) * self.multiplier['annual']
        ratio_df['ExpectedReturn(%)'] = (ann_mean*100).rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'})
        ann_std = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].std() * np.sqrt(self.multiplier['annual'])
        ratio_df['StandardDeviation(%)'] = (ann_std*100).rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'})
        sharpe = ((ann_mean - self.risk_free_rate)/ann_std).rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'})
        ratio_df['SharpeRatio'] = sharpe
        
        # Sortino ratio
        rf = self.risk_free_rate / self.multiplier['annual']
        downside_ret = (self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].dropna() - rf).apply(lambda x: np.where(x<0, x, 0))
        ann_dd = np.sqrt(np.mean(downside_ret**2)) * np.sqrt(self.multiplier['annual']) 
        sortino = ((ann_mean - self.risk_free_rate)/ann_dd).rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'})
        ratio_df['SortinoRatio'] = sortino
        
        # Maximun drawdown
        ratio_df['MaxDrawdown(%)'] = self.drawdown.min()*100
        
        # CAGR
        cagr = (self.portfolio_df.iloc[-1][['Portfolio', 'Benchmark']]**(365/self.total_days))-1
        ratio_df['CAGR(%)'] = cagr*100
        
        # Calmar ratio
        ratio_df['CalmarRatio'] = cagr/(self.drawdown.min().apply(np.abs))
        
        # Kelly Criterion
        simple_ret = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].dropna().apply(np.exp).subtract(1)
        kelly = (simple_ret - rf).mean(axis=0) / simple_ret.var()
        kelly.rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'}, inplace=True)
        ratio_df['KellyCriterion'] = kelly
        
        self.ratio_df = ratio_df

    def plot_portfolio(self):
        plt.figure(figsize=(15,15))
        ax1 = plt.subplot(3,1,1)
        self.portfolio_df[['Portfolio', 'Benchmark']].plot(color=['tab:blue', 'tab:orange'], ax=ax1, xlabel='')
        ax1.hlines(self.init_money, xmin=self.portfolio_df.index[0], xmax=self.portfolio_df.index[-1], color='k', linestyle='--')
        plt.fill_between(x=self.portfolio_df.index, y1=self.portfolio_df['Portfolio'], y2=self.init_money, color='tab:blue', alpha=0.1)
        plt.fill_between(x=self.portfolio_df.index, y1=self.portfolio_df['Benchmark'], y2=self.init_money, color='tab:orange', alpha=0.1)
        plt.text(self.portfolio_df.index[-1], self.portfolio_df['Portfolio'][-1], f"Portfolio\nReturn({self.portfolio_df['PortfolioReturns'].sum()*100:.2f}%)", color="tab:blue", fontsize=15, fontweight="heavy")
        plt.text(self.portfolio_df.index[-1], self.portfolio_df['Benchmark'][-1], f"Benchmark\nReturn({self.portfolio_df['BenchmarkReturns'].sum()*100:.2f}%)", color="tab:orange", fontsize=15, fontweight="heavy")
        plt.ylabel('Portfolio value', fontsize=18, fontweight='bold')
        
        ax2 = plt.subplot(3,1,2)
        self.drawdown.plot(color=['tab:blue', 'tab:orange'], ax=ax2, xlabel='')
        plt.scatter(self.drawdown.idxmin()['Portfolio'], self.drawdown.min()['Portfolio'], marker="X", color='k', s=100, label='MaxDrawdown')
        plt.scatter(self.drawdown.idxmin()['Benchmark'], self.drawdown.min()['Benchmark'], marker="X", color='k', s=100)
        plt.ylabel('Drawdown', fontsize=18, fontweight='bold')
        plt.text(self.drawdown.index[-1], self.drawdown['Portfolio'][-1], f"Portfolio\nMaxDD({self.drawdown.min()['Portfolio']*100:.2f}%)", color="tab:blue", fontsize=15, fontweight="heavy")
        plt.text(self.drawdown.index[-1], self.drawdown['Benchmark'][-1], f"Benchmark\nMaxDD({self.drawdown.min()['Benchmark']*100:.2f}%)", color="tab:orange", fontsize=15, fontweight="heavy")
        plt.legend()
        
        ax3 = plt.subplot(3,2,5)
        self.return_df.mul(100).plot(kind='bar', color=['tab:blue', 'tab:orange', 'tab:green'], rot=0, ax=ax3)
        plt.ylabel('Return(%)', fontsize=18, fontweight='bold')

        ax4 = plt.subplot(3,2,6)
        self.std_df.mul(100).plot(kind='bar', color=['tab:blue', 'tab:orange'], rot=0, ax=ax4)
        plt.ylabel('Standard Deviation(%)', fontsize=18, fontweight='bold')
          
        sns.despine()
        plt.show()
        
    def print_return(self):
        print('***** Portfolio Returns in percentage(%) *****')
        print(tabulate(self.return_df.mul(100), headers='keys', tablefmt='fancy_grid',  floatfmt=('.3f')))

    def print_std(self):
        print('***** Portfolio Standard Deviation in percentage(%) *****')
        print(tabulate(self.std_df.mul(100), headers='keys', tablefmt='fancy_grid', floatfmt=('.3f')))
        
    def print_ratio(self):
        print('***** Performance Ratio *****')
        print(f'Annualized with risk-free rate = {self.risk_free_rate*100:.2f} %')
        print(tabulate(self.ratio_df.T, headers='keys', tablefmt='fancy_grid', floatfmt=('.3f')))
        
    def results(self):
        self.plot_portfolio()
        print("-"*70)
        print(f"Data length: |{self.total_timedelta}|")
        print(f"Data range from |{self.data.index[0]}| to |{self.data.index[-1]}|")
        print("-"*70)
        self.print_return()
        self.print_std()
        print("-"*70)
        self.print_ratio()
        print("-"*70)
        
#--------------------------------------------------------------------------------------------------#

class TwoMovingAverageBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), kind='SMA', risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        # kind(str) ===> 'SMA' / 'EMA' / 'KAMA' / 'WMA'
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.kind = kind
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        if self.kind == 'SMA':
            self.data['ma_short'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.params[0])
            self.data['ma_long'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.params[1])
        elif self.kind == 'EMA':
            self.data['ma_short'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.params[0])
            self.data['ma_long'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.params[1])
        elif self.kind == 'KAMA':
            self.data['ma_short'] = ta.momentum.kama(close=self.data['midclose'], window=self.params[0])
            self.data['ma_long'] = ta.momentum.kama(close=self.data['midclose'], window=self.params[1])
        elif self.kind == 'WMA':
            self.data['ma_short'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.params[0])
            self.data['ma_long'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.params[1])
            
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['ma_short']>self.data['ma_long'], 1, np.where(self.data['ma_short']<self.data['ma_long'], -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------#

class ThreeMovingAverageBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), kind='SMA', risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, mid period, long period)
        # kind(str) ===> 'SMA' / 'EMA' / 'KAMA' / 'WMA'
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.kind = kind
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        if self.kind == 'SMA':
            self.data['ma_short'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.params[0])
            self.data['ma_mid'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.params[1])
            self.data['ma_long'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.params[2])
        elif self.kind == 'EMA':
            self.data['ma_short'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.params[0])
            self.data['ma_mid'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.params[1])
            self.data['ma_long'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.params[2])
        elif self.kind == 'KAMA':
            self.data['ma_short'] = ta.momentum.kama(close=self.data['midclose'], window=self.params[0])
            self.data['ma_mid'] = ta.momentum.kama(close=self.data['midclose'], window=self.params[1])
            self.data['ma_long'] = ta.momentum.kama(close=self.data['midclose'], window=self.params[2])
        elif self.kind == 'WMA':
            self.data['ma_short'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.params[0])
            self.data['ma_mid'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.params[1])
            self.data['ma_long'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.params[2])
        
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['ma_short'] > self.data['ma_mid'],
                                      self.data['ma_mid'] > self.data['ma_long']], axis=0), 1,
                              np.where(np.all([self.data['ma_short'] < self.data['ma_mid'],
                                               self.data['ma_mid'] < self.data['ma_long']], axis=0), -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------#

class MACDBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(12, 26, 9), risk_free_rate=0.01, stop_loss=None):
        # params --> (fast, slow, sign)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        # MACD signal line cross zero
        macd = ta.trend.MACD(self.data['midclose'], 
                             window_fast=self.params[0], 
                             window_slow=self.params[1],
                             window_sign=self.params[2], 
                             fillna=False)
        self.data['macd'] = macd.macd()
        # self.data['signal_line'] = macd.macd_signal()
        self.data['macd_hist'] = macd.macd_diff()
        self.data.dropna(inplace=True)
        # macd cross above signal_line from below --> long
        # macd cross below signal_line from above --> short
        self.data['signal'] = np.where(np.all([self.data['macd_hist'] > 0, self.data['macd'] < 0], axis=0), 1, 
                                        np.where(np.all([self.data['macd_hist'] < 0, self.data['macd'] > 0] , axis=0), -1, 
                                        np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------#

class RSIBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=14, risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params
        self.get_signals()
    
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        self.data['rsi'] = ta.momentum.rsi(self.data['midclose'], window=self.window, fillna=False)
        # RSI >= 70 --> short
        # RSI <= 30 --> long
        # else NA --> ffill() --> fillna(0)
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['rsi'] > 70, -1, np.where(self.data['rsi'] < 30, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------#

class StochasticOscillatorBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(14,3), risk_free_rate=0.01, stop_loss=None):
        # params --> (window, smooth_window)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        columns = ['open', 'high', 'low', 'close']
        for col in columns: 
            self.data['mid'+col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        stoch = ta.momentum.StochasticOscillator(high=self.data['midhigh'], low=self.data['midlow'], close=self.data['midclose'],
                                                 window=self.params[0], smooth_window=self.params[1], fillna=False)
        self.data['%K'] = stoch.stoch()
        self.data['%D'] = stoch.stoch_signal()
        # %D > 80 --> long 
        # %D < 20 --> short
        # else NA --> ffill NA
        # fillna(0) for other NA
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['%D'] > 80, -1,
                                       np.where(self.data['%D'] < 20, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------#

class BollingerBandBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(20, 2), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params[0]
        self.dev = params[1]
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        bb = ta.volatility.BollingerBands(close=self.data['midclose'], window=self.window, window_dev=self.dev, fillna=False)
        # self.data['bb_perc'] = bb.bollinger_pband()
        self.data['bb_high'] = bb.bollinger_hband()
        self.data['bb_avg'] = bb.bollinger_mavg()
        self.data['bb_low'] = bb.bollinger_lband()
        # price < bb_low --> long until price reach bb_avg --> neutral
        # price > bb_high --> short until price reach bb_avg --> neutral
        self.data.dropna(inplace=True)
        self.data['distance'] = self.data['midclose'] - self.data['bb_avg']
        self.data['signal'] = np.where(self.data['midclose'] < self.data['bb_low'], 1,
                                       np.where(self.data['midclose'] > self.data['bb_high'], -1, 
                                                np.where(self.data['distance']*self.data['distance'].shift(1)<0, 0, np.nan)))
        # self.data['signal'] = np.where(self.data['bb_perc'] > 1+self.threshold, -1,
        #                                np.where(self.data['bb_perc'] < 0-self.threshold, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#----------------------------------------------------------------------------------------------------#

class AroonBacktester(IterativeBacktester):
    def __init__(self, data, freq, window_threshold=(25, 50), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = window_threshold[0]
        self.threshold = window_threshold[1]
        self.get_signals()
    
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        aroon = ta.trend.AroonIndicator(close=self.data['midclose'], window=self.window, fillna=False)
        self.data['aroon_ind'] = aroon.aroon_indicator()
        # aroon indicator = aroon up - aroon down
        # aroon indicator > 0 --> long
        # aroon indicator < 0 --> short
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['aroon_ind'] > self.threshold, 1,
                                       np.where(self.data['aroon_ind'] < -1*self.threshold, -1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#----------------------------------------------------------------------------------------------------#

class CCIBacktester(IterativeBacktester):
    def __init__(self, data, freq, params, risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        self.data['midhigh'] = self.data[['bidhigh', 'askhigh']].mean(axis=1)
        self.data['midlow'] = self.data[['bidlow', 'asklow']].mean(axis=1)
        self.data['cci'] = ta.trend.cci(high=self.data['midhigh'], low=self.data['midlow'], close=self.data['midclose'], 
                                        window=self.window, constant=0.015, fillna=False)
        self.data.dropna(inplace=True)
        # strategy --> overbought/sold
        # cci > 200 --> sell
        # cci < -200 --> buy
        # else NA then ffill
        self.data['signal'] = np.where(self.data['cci'] > 200, -1,
                                    np.where(self.data['cci'] < -200, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class ADXBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(20, 30), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params[0]
        self.threshold = params[1]
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        adx = ta.trend.ADXIndicator(high=self.data['high'], low=self.data['low'], close=self.data['close'], 
                                    window=self.window, fillna=False)
        self.data['adx_pos'] = adx.adx_pos()
        self.data['adx_neg'] = adx.adx_neg()
        self.data = self.data.iloc[self.window+1:, :]
        self.data['adx_diff'] = self.data['adx_pos'] - self.data['adx_neg']
        self.data['signal'] = np.where(self.data['adx_diff'] > self.threshold, -1,
                                       np.where(self.data['adx_diff'] < -self.threshold, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']        
        
#--------------------------------------------------------------------------------------------------------#

class AwesomeOscillatorBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(5, 34), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window1 = params[0]
        self.window2 = params[1]
        # self.threshold = params[2]
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['ao'] = ta.momentum.awesome_oscillator(high=self.data['high'], low=self.data['low'], 
                                                         window1=self.window1, window2=self.window2, fillna=False)
        self.data.dropna(axis=0, inplace=True)
        # self.data['signal'] = np.where(self.data['ao'] > self.threshold, -1, 
        #                                np.where(self.data['ao'] < -self.threshold, 1, np.nan))
        # self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.data['signal'] = np.where(self.data['ao'] > 0, 1,
                                       np.where(self.data['ao'] < 0, -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class CMFBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(20, 0.2), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params[0]
        self.threshold = params[1]
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['cmf'] = ta.volume.chaikin_money_flow(high=self.data['high'], low=self.data['low'], close=self.data['close'],
                                                        volume=self.data['tickqty'], window=self.window, fillna = False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['cmf'] > self.threshold, -1,
                                       np.where(self.data['cmf'] < -self.threshold, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class DonchianChannelBacktester(IterativeBacktester):
    def __init__(self, data, freq, window=20, risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = window
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['donchian_perc'] =  ta.volatility.donchian_channel_pband(high=self.data['high'], low=self.data['low'], close=self.data['close'], 
                                                                           window=self.window,fillna=False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['donchian_perc'] > 0.8, -1,
                                       np.where(self.data['donchian_perc'] < 0.2, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class KeltnerChannelBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(20,10, 0.2), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params[0]
        self.atr = params[1]
        self.threshold = params[2]
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['keltner_perc'] =  ta.volatility.keltner_channel_pband(high=self.data['high'], low=self.data['low'], close=self.data['close'],
                                                                         window=self.window, window_atr=self.atr, 
                                                                         fillna=False, original_version=False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['keltner_perc'] > 1+self.threshold, -1,
                                       np.where(self.data['keltner_perc'] < 0-self.threshold, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class MFIBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=14, risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['mfi'] = ta.volume.money_flow_index(high=self.data['high'], low=self.data['low'], close=self.data['close'], 
                                                      volume=self.data['tickqty'], window=self.window, fillna=False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['mfi'] > 80, -1,
                                       np.where(self.data['mfi'] < 20, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class StochRSIBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(14,3,3), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params[0]
        self.smooth1 = params[1]
        self.smooth2 = params[2]
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['stochrsi'] = ta.momentum.stochrsi_d(close=self.data['close'], 
                                                       window=self.window, 
                                                       smooth1=self.smooth1,
                                                       smooth2=self.smooth2,
                                                       fillna=False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['stochrsi'] > 0.8, -1,
                                       np.where(self.data['stochrsi'] < 0.2, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class TRIXBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=14, risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['trix'] = ta.trend.trix(close=self.data['close'], window=self.window)
        self.data.dropna(axis=0, inplace=True) 
        self.data['signal'] = np.where(self.data['trix'] > 0, 1,
                                       np.where(self.data['trix'] < 0, -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class TSIBacktester(IterativeBacktester):
    def __init__(self, data, freq, windows_threshold=(13,25,30), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window_fast = windows_threshold[0]
        self.window_slow = windows_threshold[1]
        self.threshold = windows_threshold[2]
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['tsi'] = ta.momentum.tsi(close=self.data['close'], 
                                           window_slow=self.window_slow, 
                                           window_fast=self.window_fast, 
                                           fillna=False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['tsi'] > self.threshold, -1,
                                       np.where(self.data['tsi'] < -self.threshold, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class WilliamsRBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=14, risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.window = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['williamsr'] = ta.momentum.williams_r(high=self.data['high'], low=self.data['low'], close=self.data['close'], 
                                                       lbp=self.window, fillna=False)
        self.data.dropna(axis=0, inplace=True)
        self.data['signal'] = np.where(self.data['williamsr'] > -20, -1,
                                       np.where(self.data['williamsr'] < -80, 1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class IchimokuBacktester(IterativeBacktester):
    def __init__(self, data, freq, params=(9, 26, 52), risk_free_rate=0.01, stop_loss=None):
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        ichimoku = ta.trend.IchimokuIndicator(high=self.data['high'], 
                                              low=self.data['low'], 
                                              visual=True,
                                              window1=self.params[0], 
                                              window2=self.params[1], 
                                              window3=self.params[2])
        self.data['span_a'] = ichimoku.ichimoku_a()
        self.data['span_b'] = ichimoku.ichimoku_b()
        self.data['base'] = ichimoku.ichimoku_base_line()
        self.data['conv'] = ichimoku.ichimoku_conversion_line()
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['close'] > self.data[['span_a', 'span_b']].max(axis=1), 
                                       self.data['conv'] > self.data['base']], axis=0), 1,
                                       np.where(np.all([self.data['close'] < self.data[['span_a', 'span_b']].min(axis=1), 
                                                        self.data['conv'] < self.data['base']], axis=0), -1, np.nan))
        self.data['signal'] = self.data['signal'].ffill().fillna(0)
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class SMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['sma_short'] = talib.SMA(self.data['close'], self.params[0]) 
        self.data['sma_long'] = talib.SMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['sma_short']>self.data['sma_long'], 1, 
                              np.where(self.data['sma_short']<self.data['sma_long'], -1, 0))
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class EMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['ema_short'] = talib.EMA(self.data['close'], self.params[0]) 
        self.data['ema_long'] = talib.EMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['ema_short']>self.data['ema_long'], 1, 
                              np.where(self.data['ema_short']<self.data['ema_long'], -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class DEMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['dema_short'] = talib.DEMA(self.data['close'], self.params[0]) 
        self.data['dema_long'] = talib.DEMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['dema_short']>self.data['dema_long'], 1, 
                              np.where(self.data['dema_short']<self.data['dema_long'], -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class KAMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['kama_short'] = talib.KAMA(self.data['close'], self.params[0]) 
        self.data['kama_long'] = talib.KAMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['kama_short']>self.data['kama_long'], 1, 
                              np.where(self.data['kama_short']<self.data['kama_long'], -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class MIDPOINTBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['midpoint_short'] = talib.MIDPOINT(self.data['close'], self.params[0]) 
        self.data['midpoint_long'] = talib.MIDPOINT(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['midpoint_short']>self.data['midpoint_long'], 1, 
                              np.where(self.data['midpoint_short']<self.data['midpoint_long'], -1, 0))
        self.signals = self.data['signal']        
        
#--------------------------------------------------------------------------------------------------------#

class MIDPRICEBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['midprice_short'] = talib.MIDPRICE(self.data['high'], self.data['low'], self.params[0]) 
        self.data['midprice_long'] = talib.MIDPRICE(self.data['high'], self.data['low'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['midprice_short']>self.data['midprice_long'], 1, 
                              np.where(self.data['midprice_short']<self.data['midprice_long'], -1, 0))
        self.signals = self.data['signal']            
        
#--------------------------------------------------------------------------------------------------------#

class TEMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['tema_short'] = talib.TEMA(self.data['close'], self.params[0]) 
        self.data['tema_long'] = talib.TEMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['tema_short']>self.data['tema_long'], 1, 
                              np.where(self.data['tema_short']<self.data['tema_long'], -1, 0))
        self.signals = self.data['signal']
        
#--------------------------------------------------------------------------------------------------------#

class TRIMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['trima_short'] = talib.TRIMA(self.data['close'], self.params[0]) 
        self.data['trima_long'] = talib.TRIMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['trima_short']>self.data['trima_long'], 1, 
                              np.where(self.data['trima_short']<self.data['trima_long'], -1, 0))
        self.signals = self.data['signal']        
        
#--------------------------------------------------------------------------------------------------------#

class WMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['wma_short'] = talib.WMA(self.data['close'], self.params[0]) 
        self.data['wma_long'] = talib.WMA(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['wma_short']>self.data['wma_long'], 1, 
                              np.where(self.data['wma_short']<self.data['wma_long'], -1, 0))
        self.signals = self.data['signal']  
        
#--------------------------------------------------------------------------------------------------------#

class LinearRegressionBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['linreg_short'] = talib.LINEARREG(self.data['close'], self.params[0]) 
        self.data['linreg_long'] = talib.LINEARREG(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['linreg_short']>self.data['linreg_long'], 1, 
                              np.where(self.data['linreg_short']<self.data['linreg_long'], -1, 0))
        self.signals = self.data['signal']  
        
#--------------------------------------------------------------------------------------------------------#

class TSFBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20,100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['tsf_short'] = talib.TSF(self.data['close'], self.params[0]) 
        self.data['tsf_long'] = talib.TSF(self.data['close'], self.params[1]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(self.data['tsf_short']>self.data['tsf_long'], 1, 
                              np.where(self.data['tsf_short']<self.data['tsf_long'], -1, 0))
        self.signals = self.data['signal']  
        
#--------------------------------------------------------------------------------------------------------#

class ThreeSMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.SMA(self.data['close'], self.params[0]) 
        self.data['mid'] = talib.SMA(self.data['close'], self.params[1])
        self.data['long'] = talib.SMA(self.data['close'], self.params[2]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class ThreeEMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.EMA(self.data['close'], self.params[0]) 
        self.data['mid'] = talib.EMA(self.data['close'], self.params[1])
        self.data['long'] = talib.EMA(self.data['close'], self.params[2]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class ThreeKAMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.KAMA(self.data['close'], self.params[0]) 
        self.data['mid'] = talib.KAMA(self.data['close'], self.params[1])
        self.data['long'] = talib.KAMA(self.data['close'], self.params[2]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class ThreeMIDPOINTBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.MIDPOINT(self.data['close'], self.params[0]) 
        self.data['mid'] = talib.MIDPOINT(self.data['close'], self.params[1])
        self.data['long'] = talib.MIDPOINT(self.data['close'], self.params[2]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class ThreeMIDPRICEBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.MIDPRICE(self.data['high'], self.data['low'], self.params[0])  
        self.data['mid'] = talib.MIDPRICE(self.data['high'], self.data['low'], self.params[1]) 
        self.data['long'] = talib.MIDPRICE(self.data['high'], self.data['low'], self.params[2])  
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class ThreeTRIMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.TRIMA(self.data['close'], self.params[0]) 
        self.data['mid'] = talib.TRIMA(self.data['close'], self.params[1])
        self.data['long'] = talib.TRIMA(self.data['close'], self.params[2]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class ThreeWMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 50, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
        self.data['short'] = talib.WMA(self.data['close'], self.params[0]) 
        self.data['mid'] = talib.WMA(self.data['close'], self.params[1])
        self.data['long'] = talib.WMA(self.data['close'], self.params[2]) 
        self.data.dropna(inplace=True)
        self.data['signal'] = np.where(np.all([self.data['short'] > self.data['mid'], self.data['mid'] > self.data['long']], axis=0), 1,
                              np.where(np.all([self.data['short'] < self.data['mid'], self.data['mid'] < self.data['long']], axis=0), -1, 0)) 
        self.signals = self.data['signal']

#--------------------------------------------------------------------------------------------------------#

class MultiMABacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (20, 100), risk_free_rate=0.01, stop_loss=None):
        # params(tuple) ===> (short period, long period)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
            
        indicators = {
            'SMA': talib.SMA,
            'EMA': talib.EMA,
            'KAMA': talib.KAMA,
            'MIDPOINT': talib.MIDPOINT,
            'MIDPRICE': talib.MIDPRICE,
            'TRIMA': talib.TRIMA,
            'WMA': talib.WMA
        }
        
        
        for name, func in indicators.items(): # compute all indicators
            for i in range(len(self.params)):
                if name == 'MIDPRICE':
                    self.data[f"{name}_{i+1}"] = func(self.data['high'],self.data['low'], self.params[i])
                else:
                    self.data[f"{name}_{i+1}"] = func(self.data['close'], self.params[i])
        
        signal_col = [] # getting all signals
        for name in indicators.keys():
            self.data[f"{name}_signal"] = np.where(self.data[f"{name}_1"] > self.data[f"{name}_2"], 1,
                                          np.where(self.data[f"{name}_1"] < self.data[f"{name}_2"], -1, np.nan))
            self.data[f"{name}_signal"] = self.data[f"{name}_signal"].ffill().fillna(0)
            signal_col.append(f"{name}_signal")
        
        self.signals = self.data[signal_col].mode(axis=1).iloc[:, 0] # get signals mode (i.e., majority vote)

# ------------------------------------------------------------------------------------------------------------

class ThreeIndicatorsBacktester(IterativeBacktester):
    def __init__(self, data, freq, params = (50, 30, 100), risk_free_rate=0.01, stop_loss=None):
        # EMA / RSI / BollingerBands
        # params(tuple) ===> (ema_window, rsi_window, bb_window)
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate, stop_loss=stop_loss)
        self.params = params
        self.get_signals()
        
    def get_signals(self):
        for col in ['open', 'close', 'high', 'low']:
            self.data[col] = self.data[['bid'+col, 'ask'+col]].mean(axis=1)
            
        # get indicators
        self.data['ema'] = talib.EMA(self.data.close, self.params[0])
        self.data['rsi'] = talib.RSI(self.data.close, self.params[1])
        self.data['bb_upper'], self.data['bb_mid'], self.data['bb_lower'] = talib.BBANDS(self.data.close, self.params[2])
        self.data.dropna(axis=0, inplace=True)
        
        # get signals
        self.data['ema_signal'] = np.where(self.data.close > self.data.ema, 1, -1)
        
        self.data['rsi_signal'] = np.where(self.data.rsi > 70, -1, np.where(self.data.rsi < 30, 1, np.nan))
        self.data['rsi_less_50'] = np.where(self.data.rsi < 50, 1, -1)
        self.data.loc[(self.data.rsi_less_50 * self.data.rsi_less_50.shift(1) < 0), 'rsi_signal'] = 0
        self.data.rsi_signal.ffill(inplace=True)
        self.data.rsi_signal.fillna(0, inplace=True)
        
        self.data['bb_signal'] = np.where(self.data.close > self.data.bb_upper, -1, 
                                    np.where(self.data.close < self.data.bb_lower, 1, np.nan))
        self.data['bb_less_mid'] = np.where(self.data.close < self.data.bb_mid, 1, -1)
        self.data.loc[(self.data.bb_less_mid * self.data.bb_less_mid.shift(1) < 0), 'bb_signal'] = 0
        self.data.bb_signal.ffill(inplace=True)
        self.data.bb_signal.fillna(0, inplace=True)
    
        signal_cols = ['ema_signal', 'rsi_signal', 'bb_signal']
        self.signals = self.data.loc[:,signal_cols].mode(axis=1).iloc[:, 0]