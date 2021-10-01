import numpy as np
import pandas as pd 
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from tabulate import tabulate
plt.style.use('seaborn-whitegrid')

class IterativeBacktester():
    def __init__(self, data, signals, freq, risk_free_rate=0.01):
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
        total_ret = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].sum()
        total_ret.rename(index={'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'}, inplace=True)
        return_df = pd.DataFrame(data={'TotalReturn':None, 'MonthlyReturn':None, 'AnnualReturn':None}, index=['Portfolio', 'Benchmark'])
        return_df.loc[:, 'TotalReturn'] = total_ret
        return_df.loc[:, 'AnnualReturn'] = total_ret * (self.multiplier/len(self.data)) 
        return_df.loc[:, 'MonthlyReturn'] = total_ret * ((self.multiplier/len(self.data))/12) 
        return_df = return_df.T
        return_df['Alpha'] = return_df['Portfolio'] - return_df['Benchmark']
        self.return_df = return_df
        
    def get_std_df(self):
        total_std = self.portfolio_df[['Portfolio', 'Benchmark']].std()
        std_df = pd.DataFrame(data={'TotalStdev':None, 'MonthlyStdev':None, 'AnnualStdev':None}, index=['Portfolio', 'Benchmark'])
        std_df.loc[:, 'TotalStdev'] = total_std
        std_df.loc[:, 'AnnualStdev'] = total_std * np.sqrt(self.multiplier/len(self.data)) 
        std_df.loc[:, 'MonthlyStdev'] = total_std * np.sqrt((self.multiplier/len(self.data))/12)
        std_df = std_df.T
        self.std_df = std_df
        
    def get_drawdown(self):
        cummax = self.portfolio_df[['Portfolio', 'Benchmark']].cummax()
        self.drawdown =  (self.portfolio_df[['Portfolio', 'Benchmark']] - cummax) / cummax
    
    def get_multiplier(self):
        self.multiplier = len(self.data)*(365/self.total_days) 
    
    def get_ratio_df(self):
        ratio_df = pd.DataFrame(index=['Portfolio', 'Benchmark'])
        # ratios are computed on returns 
        
        # Sharpe ratio 
        ann_mean = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].mean() * self.multiplier
        ann_std = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].std() * np.sqrt(self.multiplier)
        sharpe = ((ann_mean - self.risk_free_rate)/ann_std).rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'})
        ratio_df['SharpeRatio'] = sharpe
        
        # Sortino ratio
        rf = self.risk_free_rate / self.multiplier
        downside_ret = (self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].dropna() - rf).apply(lambda x: np.where(x<0, x, 0))
        ann_dd = np.sqrt(np.mean(downside_ret**2)) * np.sqrt(self.multiplier) 
        sortino = ((ann_mean - self.risk_free_rate)/ann_dd).rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'})
        ratio_df['SortinoRatio'] = sortino
        
        # Maximun drawdown
        ratio_df['MaxDrawdown(%)'] = self.drawdown.min()*100
        
        # Calmar ratio
        cagr = (self.portfolio_df.iloc[-1][['Portfolio', 'Benchmark']]**(365/self.total_days))-1
        ratio_df['CalmarRatio'] = cagr/(self.drawdown.min().apply(np.abs))
        
        # Kelly Criterion
        simple_ret = self.portfolio_df[['PortfolioReturns', 'BenchmarkReturns']].dropna().apply(np.exp).subtract(1)
        kelly = (simple_ret - rf).mean() / simple_ret.var()
        kelly.rename({'PortfolioReturns':'Portfolio', 'BenchmarkReturns':'Benchmark'}, inplace=True)
        ratio_df['KellyCriterion'] = kelly
        
        self.ratio_df = ratio_df

    def plot_portfolio(self):
        plt.figure(figsize=(15,15))
        ax1 = plt.subplot(3,1,1)
        self.portfolio_df[['Portfolio', 'Benchmark']].plot(color=['tab:blue', 'tab:orange'], ax=ax1, xlabel='')
        ax1.plot(self.portfolio_df.index ,[self.init_money]*len(self.portfolio_df),'--k')
        plt.fill_between(x=self.portfolio_df.index, y1=self.portfolio_df['Portfolio'], y2=self.init_money, color='tab:blue', alpha=0.1)
        plt.fill_between(x=self.portfolio_df.index, y1=self.portfolio_df['Benchmark'], y2=self.init_money, color='tab:orange', alpha=0.1)
        plt.text(self.portfolio_df.index[-1], self.portfolio_df['Portfolio'][-1], f"Portfolio\nReturn({self.portfolio_df['PortfolioReturns'].sum()*100:.2f}%)", color="tab:blue", fontsize=15, fontweight="heavy")
        plt.text(self.portfolio_df.index[-1], self.portfolio_df['Benchmark'][-1], f"Benchmark\nReturn({self.portfolio_df['BenchmarkReturns'].sum()*100:.2f}%)", color="tab:orange", fontsize=15, fontweight="heavy")
        plt.ylabel('Portfolio value', fontsize=18, fontweight='bold')
        
        ax2 = plt.subplot(3,1,2)
        self.drawdown.plot(color=['tab:blue', 'tab:orange'], ax=ax2, xlabel='')
        plt.plot(self.drawdown.idxmin()['Portfolio'], self.drawdown.min()['Portfolio'], marker="X", color='k', markersize=12, label='MaxDrawdown')
        plt.plot(self.drawdown.idxmin()['Benchmark'], self.drawdown.min()['Benchmark'], marker="X", color='k', markersize=12)
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
        print("-"*50)
        print(f"Data length: |{self.total_timedelta}|")
        print(f"Data range from |{self.data.index[0]}| to |{self.data.index[-1]}|")
        print("-"*50)
        self.print_return()
        self.print_std()
        print("-"*50)
        self.print_ratio()
        
#-----------------------------------------------------------------------------------------------------------------------------

class TwoMovingAverageBacktester(IterativeBacktester):
    def __init__(self, data, freq, periods = (20,100), kind='SMA', risk_free_rate=0.01):
        # periods(tuple) ===> (short period, long period)
        # kind(str) ===> 'SMA' / 'EMA' / 'KAMA' / 'WMA'
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate)
        self.periods = periods
        self.kind = kind
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        if self.kind == 'SMA':
            self.data['ma_short'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_long'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.periods[1])
        elif self.kind == 'EMA':
            self.data['ma_short'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_long'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.periods[1])
        elif self.kind == 'KAMA':
            self.data['ma_short'] = ta.momentum.kama(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_long'] = ta.momentum.kama(close=self.data['midclose'], window=self.periods[1])
        elif self.kind == 'WMA':
            self.data['ma_short'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_long'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.periods[1])
            
        self.data['signal'] = np.where(self.data['ma_short']>self.data['ma_long'], 1, np.where(self.data['ma_short']<self.data['ma_long'], -1, 0))
        self.data.dropna(inplace=True)
        self.signals = self.data['signal']
        
#---------------------------------------------------------------------------------------------------------------------------

class ThreeMovingAverageBacktester(IterativeBacktester):
    def __init__(self, data, freq, periods = (20, 50, 100), kind='SMA', risk_free_rate=0.01):
        # periods(tuple) ===> (short period, mid period, long period)
        # kind(str) ===> 'SMA' / 'EMA' / 'KAMA' / 'WMA'
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, risk_free_rate=risk_free_rate)
        self.periods = periods
        self.kind = kind
        self.get_signals()
        
    def get_signals(self):
        self.data['midclose'] = self.data[['bidclose', 'askclose']].mean(axis=1)
        if self.kind == 'SMA':
            self.data['ma_short'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_mid'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.periods[1])
            self.data['ma_long'] = ta.trend.sma_indicator(close=self.data['midclose'], window=self.periods[2])
        elif self.kind == 'EMA':
            self.data['ma_short'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_mid'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.periods[1])
            self.data['ma_long'] = ta.trend.ema_indicator(close=self.data['midclose'], window=self.periods[2])
        elif self.kind == 'KAMA':
            self.data['ma_short'] = ta.momentum.kama(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_mid'] = ta.momentum.kama(close=self.data['midclose'], window=self.periods[1])
            self.data['ma_long'] = ta.momentum.kama(close=self.data['midclose'], window=self.periods[2])
        elif self.kind == 'WMA':
            self.data['ma_short'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.periods[0])
            self.data['ma_mid'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.periods[1])
            self.data['ma_long'] = ta.trend.wma_indicator(close=self.data['midclose'], window=self.periods[2])
            
        self.data['signal'] = np.where(np.all([self.data['ma_short'] > self.data['ma_mid'],
                                      self.data['ma_mid'] > self.data['ma_long']], axis=0), 1,
                              np.where(np.all([self.data['ma_short'] < self.data['ma_mid'],
                                               self.data['ma_mid'] < self.data['ma_long']], axis=0), -1, 0))
        self.data.dropna(inplace=True)
        self.signals = self.data['signal']