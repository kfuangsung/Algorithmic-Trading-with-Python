import numpy as np
import pandas as pd 
from tqdm import tqdm
import ta
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (14,6)

class IterativeBacktester():
    def __init__(self, data, signals, freq, init_money=10000):
        self.data = data
        self.signals = signals
        self.freq = freq
            # minutes: m1, m5, m15 and m30,
            # hours: H1, H2, H3, H4, H6 and H8,
            # one day: D1,
            # one week: W1,
            # one month: M1.
        self.init_money = init_money
        self.current_money = init_money
        self.portfolio_values = [init_money]
        self.current_pos = 0
        self.unit_cost = None
        self.holding_amount = None
        self.passive_values = [init_money]
        self.passive_amount = None
        self.passive_cost = None
        self.returns = {}
        self.stdevs = {}
        self.sharpe = {}
        self.drawdown = None
        
    def compute_amount(self, money, price):
        return int(np.floor(money/price))
    
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
        
    def backtest(self, progress_bar=True):
        if progress_bar is True:
            iterate = tqdm(range(len(self.data)-1))
        else:
            iterate = range(len(self.data)-1)
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
        self.add_returns()
        self.add_stdevs()
        self.add_drawdown()
                    
    def plot_portfolio(self):
        plt.plot(self.portfolio_values, color='tab:blue', linestyle="-")
        plt.plot(self.passive_values, color='tab:orange', linestyle=":")
        plt.plot([self.init_money]*len(self.portfolio_values),'--k')
#         plt.text(0, self.init_money, "Initial value", color="k", fontsize=15, fontweight="heavy")
        plt.text(len(self.portfolio_values), self.portfolio_values[-1], f"Portfolio({self.returns['Portfolio']['Total Return']*100:.2f}%)", color="tab:blue", fontsize=15, fontweight="heavy")
        plt.text(len(self.passive_values), self.passive_values[-1], f"Benchmark({self.returns['Benchmark']['Total Return']*100:.2f}%)", color="tab:orange", fontsize=15, fontweight="heavy")
        sns.despine()
        plt.title('Portfolio Values', fontsize=15)
        plt.show()
        
    def compute_total_return(self, values):
        return (values[-1]/values[0]) - 1
    
    def compute_annual_return(self, values, freq):
        # 250 trading days in a year
        ret = (values[-1]/values[0])
        multiplier = int(freq[1:])
        length = len(values)*multiplier
        if 'm' in freq:
            num = 360000
        elif 'H' in freq:
            num = 6000
        elif 'D' in freq:
            num = 250
        elif 'W' in freq:
            num = 52
        elif 'M' in freq:
            num = 12
        return (ret**(num/length))-1
            
    def compute_monthly_return(self, values, freq):
        # 21 trading days in a month
        ret = (values[-1]/values[0])
        multiplier = int(freq[1:])
        length = len(values)*multiplier
        if 'm' in freq:
            num = 30240
        elif 'H' in freq:
            num = 504
        elif 'D' in freq:
            num = 21
        elif 'W' in freq:
            num = 4
        elif 'M' in freq:
            num = 1
        return (ret**(num/length))-1
        
    def add_returns(self):
        # store returns 
        for name, values in zip(['Benchmark', 'Portfolio'], [self.passive_values, self.portfolio_values]):
            self.returns[name] = {}
            self.returns[name]['Total Return'] = self.compute_total_return(values)
            self.returns[name]['Annualized Return'] = self.compute_annual_return(values, self.freq)
            self.returns[name]['Monthly Return'] = self.compute_monthly_return(values, self.freq)
        
        self.returns['Alpha'] = {}
        for name in ['Total Return', 'Annualized Return', 'Monthly Return']:
            self.returns['Alpha'][name] = self.returns['Portfolio'][name] - self.returns['Benchmark'][name]
            
    def print_returns(self):
        print("="*40)
        print('Portfolio Returns')
        print(f"Data Length: {self.data.index[-1] - self.data.index[0]}")
        print("="*40)
        for head, vals in self.returns.items():
            print(f"***** {head} *****")
            for name, ret in vals.items():
                print(f"{name}: {ret*100:.2f} %")
            print("-"*40)
            
    def plot_returns(self):
        dfs = []
        for name, vals in self.returns.items():
            value = {}
            value['Return type'] = []
            value['Returns (%)'] = []
            value['Category'] = []
            for freq, ret in vals.items():
                value['Return type'].append(freq)
                value['Returns (%)'].append(ret*100)
                value['Category'].append(name)
            df = pd.DataFrame(data=value)
            dfs.append(df)
        df_cat = pd.concat(dfs)
        sns.catplot(data=df_cat, x='Return type', y='Returns (%)', hue='Category', kind='bar', orient="v",height=5, aspect=1.2, legend_out=False)
        sns.despine()
        plt.xlabel('')
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.title('Returns', fontsize=15)
        plt.show()
        
    def show_returns(self):
        self.print_returns()
        self.plot_returns()
        
    def compute_total_stdev(self, values):
        # rebase portfolio to 1
        return np.std(np.array(values)/self.init_money)
    
    def compute_annual_stdev(self, values, freq):
        # rebase portfolio to 1
        stdev = np.std(np.array(values)/self.init_money)
        multiplier = int(freq[1:])
        length = len(values)*multiplier
        if 'm' in freq:
            num = 360000
        elif 'H' in freq:
            num = 6000
        elif 'D' in freq:
            num = 250
        elif 'W' in freq:
            num = 52
        elif 'M' in freq:
            num = 12
        return stdev*np.sqrt(num/length)
    
    def compute_monthly_stdev(self, values, freq):
        # rebase portfolio to 1
        stdev = np.std(np.array(values)/self.init_money)
        multiplier = int(freq[1:])
        length = len(values)*multiplier
        if 'm' in freq:
            num = 30240
        elif 'H' in freq:
            num = 504
        elif 'D' in freq:
            num = 21
        elif 'W' in freq:
            num = 4
        elif 'M' in freq:
            num = 1
        return stdev*np.sqrt(num/length)
    
    def add_stdevs(self):
        # store returns 
        for name, values in zip(['Benchmark', 'Portfolio'], [self.passive_values, self.portfolio_values]):
            self.stdevs[name] = {}
            self.stdevs[name]['Total Stdev'] = self.compute_total_stdev(values)
            self.stdevs[name]['Annualized Stdev'] = self.compute_annual_stdev(values, self.freq)
            self.stdevs[name]['Monthly Stdev'] = self.compute_monthly_stdev(values, self.freq)
    
    def print_stdevs(self):
        print("="*40)
        print('Portfolio(rebased to 1) Standard Deviation')
        print(f"Data Length: {self.data.index[-1] - self.data.index[0]}")
        print("="*40)
        for head, vals in self.stdevs.items():
            print(f"***** {head} *****")
            for name, std in vals.items():
                print(f"{name}: {std*100:.2f} %")
            print("-"*40)
            
    def show_stdevs(self):
        self.print_stdevs()
        self.plot_stdevs()
            
    def plot_stdevs(self):
        dfs = []
        for name, vals in self.stdevs.items():
            value = {}
            value['Stdev type'] = []
            value['Standatd deviation (%)'] = []
            value['Category'] = []
            for freq, std in vals.items():
                value['Stdev type'].append(freq)
                value['Standatd deviation (%)'].append(std*100)
                value['Category'].append(name)
            df = pd.DataFrame(data=value)
            dfs.append(df)
        df_cat = pd.concat(dfs)
        sns.catplot(data=df_cat, x='Stdev type', y='Standatd deviation (%)', hue='Category', kind='bar', orient="v",height=5, aspect=1, legend_out=False)
        sns.despine()
        plt.xlabel('')
        plt.xticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.title('Standard Deviation', fontsize=15)
        plt.show()
        
    def compute_sharpe_ratio(self, rf=0):
        # annual sharpe ratio
        bench = (self.returns['Benchmark']['Annualized Return']-rf)/self.stdevs['Benchmark']['Annualized Stdev']
        port = (self.returns['Portfolio']['Annualized Return']-rf)/self.stdevs['Portfolio']['Annualized Stdev']
        self.sharpe['Benchmark'] = bench
        self.sharpe['Portfolio'] = port
    
    def print_sharpe_ratio(self, rf=0):
        self.compute_sharpe_ratio(rf=rf)
        print("="*40)
        print('Annual Sharpe Ratio')
        print(f"Data Length: {self.data.index[-1] - self.data.index[0]}")
        print("="*40)
        for name, vals in self.sharpe.items():
            print(f"{name}: {vals:.2f}")
        print('-'*40)
        
    def add_drawdown(self):
        df = pd.DataFrame({'Portfolio':self.portfolio_values, 'Benchmark':self.passive_values,})
        roll_max = df.cummax()
        self.drawdown = df/roll_max - 1
        
    def plot_drawdown(self):
        self.drawdown.plot(legend=False, color=['tab:blue', 'tab:orange'], style=['-', ':'])
        plt.text(len(self.portfolio_values), self.drawdown['Portfolio'].iloc[-1], 'Portfolio', fontsize=15, color='tab:blue', fontweight='heavy')
        plt.text(len(self.passive_values), self.drawdown['Benchmark'].iloc[-1], 'Benchmark', fontsize=15, color='tab:orange', fontweight='heavy')
        plt.title('Drawdown', fontsize=15)
        sns.despine()
        plt.show()
        
    def print_max_drawdown(self):
        print('='*40)
        print('Maximum Drawdown')
        print('='*40)
        print(f"Benchmark: {self.drawdown.min()['Benchmark']*100:.2f} %")
        print(f"Portfolio: {self.drawdown.min()['Portfolio']*100:.2f} %")
        print('-'*40)
    
    def show_drawdown(self):
        self.print_max_drawdown()
        self.plot_drawdown()
        
class MovingAverageBacktester(IterativeBacktester):
    def __init__(self, data, freq, periods = (20,100), kind='SMA', init_money=10000):
        # periods(tuple) ===> (short period, long period)
        # kind(str) ===> 'SMA' or 'EMA'
        IterativeBacktester.__init__(self, data=data, signals=None, freq=freq, init_money=init_money)
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
        self.data['signal'] = np.where(self.data['ma_short']>self.data['ma_long'], 1, np.where(self.data['ma_short']<self.data['ma_long'], -1, 0))
        self.data.dropna(inplace=True)
        self.signals = self.data['signal']
