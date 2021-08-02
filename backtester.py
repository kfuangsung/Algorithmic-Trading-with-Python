import numpy as np
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('bmh')
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
        self.trade_returns = {}
        
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
        
    def backtest(self):
        for i in tqdm(range(len(self.data)-1)):
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
        # compute returns
        self.add_returns()
                    
    def show_plot(self):
        plt.plot(self.portfolio_values, label='portfolio')
        plt.plot(self.passive_values, linestyle=":",label = 'benchmark')
        plt.plot([self.init_money]*len(self.portfolio_values),'--k', label='initial money')
        plt.title('porfolio values'.upper())
        plt.legend()
        plt.show()
        
    def compute_total_return(self, values):
        return (values[-1]/values[0]) - 1
    
    def compute_annual_return(self, values, freq):
        ret = (values[-1]/values[0])
        multiplier = int(freq[1:])
        length = len(values)*multiplier
        if 'm' in freq:
            return ret**(525600/length)-1
        elif 'H' in freq:
            return ret**(8760/length)-1
        elif 'D' in freq:
            return ret**(365/length)-1
        elif 'W' in freq:
            return ret**(52.143/length)-1
        elif 'M' in freq:
            return ret**(12/length)-1
            
        
    def compute_monthly_return(self, values, freq):
        ret = (values[-1]/values[0])
        multiplier = int(freq[1:])
        length = len(values)*multiplier
        if 'm' in freq:
            return ret**(43800/length)-1
        elif 'H' in freq:
            return ret**(730/length)-1
        elif 'D' in freq:
            return ret**(30.417/length)-1
        elif 'W' in freq:
            return ret**(4.345/length)-1
        elif 'M' in freq:
            return ret**(1/length)-1
    
    def add_returns(self):
        # store returns 
        for name, values in zip(['Benchmark', 'Portfolio'], [self.passive_values, self.portfolio_values]):
            self.trade_returns[name] = {}
            self.trade_returns[name]['Total Return'] = self.compute_total_return(values)
            self.trade_returns[name]['Annualized Return'] = self.compute_annual_return(values, self.freq)
            self.trade_returns[name]['Monthly Return'] = self.compute_monthly_return(values, self.freq)
        
        self.trade_returns['Alpha'] = {}
        for name in ['Total Return', 'Annualized Return', 'Monthly Return']:
            self.trade_returns['Alpha'][name] = self.trade_returns['Portfolio'][name] - self.trade_returns['Benchmark'][name]
            
    def print_returns(self):
        print("="*40)
        print(f"Data Length: {self.data.index[-1] - self.data.index[0]}")
        print("="*40)
        for head, vals in self.trade_returns.items():
            print(f"***** {head} *****")
            for name, ret in vals.items():
                print(f"{name}: {ret*100:.2f} %")
            print("-"*40)