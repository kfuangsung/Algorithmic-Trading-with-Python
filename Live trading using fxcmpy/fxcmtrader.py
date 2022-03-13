import sys
import os
import time
import pytz
import csv
import json
import ta
import talib
import fxcmpy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fxcmtoken import *

# -------------------------------------------------------------------------------------------------------------------#

# Base Class


class FXCMTrader():
    """
    A base class for algorithmic trading using fxcmpy

    ...

    Attributes
    ----------
    token : str
        account token for connection
    symbol_list : list
        list of symbols for trading
    time_frame : str
        time frame to trade. E.g., m1, m5, m15, H1, D1
    min_len : int
        number of candles to retreive in get_data
    end_time : datetime.datetime
        when to stop trading
    account_type : str
        demo or real account
    time_zone : str
        determine time zone 
    position_pct : float
        position percentage used to calculate trade amount
    margin_pct : float
        minimum margin percentage requires
    margin_dict : dict
        dictionary contains margin of assets. symbol(str) --> int
    log_path : str
        path to save log files
    log_header : list
        headers in log files
    log_freq_min : int
        frequency in minutes to save log files
    update_amount_freq_minute : int
        frequency in minutes to update trade amount
    con : None 
        connection to fxcm server. After connect, con : fxcmpy object 
    balance : None
        account balance. After get_account_balance, balance : float
    current_positions : dict
        dictionary to save current position of assets. symbol(str) --> int
        1 --> LONG
        0 --> NEUTRAL
        -1 --> SHORT
    next_candles_time : dict
        dictionary to save next candle time for assets, symbol(str) --> datetime.datetime
    trade_amount : dict
        dictionary to save trading acount for assets. symbol(str) --> int
    log_time : datetime.datetime
        when to save log files
    update_amount_time : datetime.datetime
        when to update trade amount
    add_log_time : datetime.timedelta
        add to log_time to get the next log_time
    add_amount_time : datetime.timedelta
        add to update_amount_time to get the next update_amount_time
    add_candle_time : datetime.timedelta
        add to next_candles_time to get the next next_candles_time
    """

    def __init__(self, TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                 start_time, end_time, position_pct=0.02, margin_pct=0.5, 
                 account_type='DEMO', time_zone="UTC", log_path='log_files',
                 log_header=['balance', 'equity', 'dayPL', 'usdMr', 'usableMargin', 'date'],
                 log_freq_min=30, update_amount_freq_min=30, #restart_freq_min=60, 
                 update_pos_freq_min=30):

        self.token = TOKEN
        self.symbol_list = symbol_list
        self.time_frame = time_frame
        self.min_len = int(min_len)
        self.start_time = start_time
        self.end_time = end_time
        self.account_type = account_type
        self.time_zone = time_zone
        self.position_pct = position_pct
        self.margin_pct = margin_pct
        self.margin_dict = margin_dict
        self.log_path = log_path
        self.log_header = log_header
        self.log_freq_min = int(log_freq_min)
        self.update_amount_freq_min = int(update_amount_freq_min)
        # self.restart_freq_min = int(restart_freq_min)
        self.update_pos_freq_min = int(update_pos_freq_min)
        self.con = None
        self.balance = None
        self.current_positions = {}  # symbol(str) --> position sign(int)
        self.next_candles_time = {} # symbol(str) --> lastest candles time(datetime.datetime)
        self.trade_amount = {}  # symbol(str) --> amount to trade(int)
        self.log_time = self.get_current_time()
        self.update_amount_time = self.get_current_time()
        self.update_pos_time = self.get_current_time()
        # self.next_restart_time = self.get_current_time() + timedelta(minutes=self.restart_freq_min) # routinely restart script to deal with "packet queue is empty, aborting"
        self.add_log_time = timedelta(minutes=self.log_freq_min)
        self.add_amount_time = timedelta(minutes=self.update_amount_freq_min)
        self.add_pos_time = timedelta(minutes=self.update_pos_freq_min)
        self.add_candle_time = self.get_add_candle_time()

    def connect(self):
        """Make connection to FXCM server."""

        if self.con is None:
            try:
                self.con = fxcmpy.fxcmpy(access_token=self.token, log_level='error', server='demo', log_file='log.txt')
                time.sleep(1)
                if self.con.is_connected() is True:
                    self.print_connection_status()
                elif self.con.is_connected() is False:
                    self.connect()
            except:
                self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| connect |')
        else:
            if self.con.is_connected() is True:
                self.log_trade(f'| {self.get_current_time()} | already connected |')
                self.print_connection_status()
            elif self.con.is_connected() is False:
                self.con.connect()
                self.connect()

    def disconnect(self):
        """Disconnect from fxcm server."""

        if self.con is not None:
            # self.log_trade(f'| {self.get_current_time()} | self.con is not None |')
            if self.con.is_connected() is False:
                self.log_trade(f'| {self.get_current_time()} | already disconnected |')
            elif self.con.is_connected() is True:
                try:
                    self.con.close()
                    if self.con.is_connected() is False:
                        self.print_connection_status()
                    elif self.con.is_connected() is True:
                        self.disconnect()
                except:
                    self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| disconnect |')
        else:
            self.log_trade(f'| {self.get_current_time()} | self.con is None |')

    def print_connection_status(self):
        """Print current connection status, either 'established' or 'unset'."""

        try:
            self.log_trade(f"| {self.get_current_time()} | Connection Status: {self.con.connection_status} |")
        except:
            self.log_trade(f"| {self.get_current_time()} |*** Not yet connected. ***|")

    def get_account_balance(self):
        """save current account balance to self.balance"""

        try:
            account_info = self.con.get_model(models=['Account'], summary=False)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| get_account_balance |')
        account_info = account_info['accounts'][0]
        self.balance = account_info['balance']

    def get_trade_amount(self, symbol):
        """get trade amount for the asset depends on account balance
        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        int
            amount to place trade
        """

        return int(self.balance*self.position_pct/self.margin_dict[symbol])

    def update_trade_amount(self):
        """update trade amount since account balance may change"""

        self.get_account_balance()
        for symbol in self.symbol_list:
            self.trade_amount[symbol] = self.get_trade_amount(symbol)
        self.log_trade(f'| {self.get_current_time()} | Updated trade amount |')
        self.update_amount_time += self.add_amount_time

    def update_current_positions(self):
        """get current positions of all symbols"""

        try:
            open_positions = self.con.get_model(
                models=['OpenPosition'], summary=False)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| update_current_positions |')
        open_positions_dict = {i['currency']: i['isBuy']
                               for i in open_positions['open_positions']}
        for symbol in self.symbol_list:
            if symbol in open_positions_dict:
                if open_positions_dict[symbol] is True: 
                    self.current_positions[symbol] = 1
                elif open_positions_dict[symbol] is False:
                    self.current_positions[symbol] = -1
            else:
                self.current_positions[symbol] = 0
        self.log_trade(f'| {self.get_current_time()} | Updated current positions |')
        self.update_pos_time += self.add_pos_time

    def init_candle_time(self, delay_minute=3):
        """retreive current candle time"""

        for symbol in self.symbol_list:
            candle_time = self.get_candles_time(symbol)
            if (self.get_current_time() - candle_time) < timedelta(minutes=delay_minute):
                self.next_candles_time[symbol] = candle_time
            else:
                self.next_candles_time[symbol] = candle_time + self.add_candle_time
            time.sleep(1)

    def initiate(self):
        """initiate all necessity before start trading."""
        
        try:
            self.connect()
            self.update_current_positions()
            self.update_trade_amount()
            self.init_candle_time()
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| initiate |')

    def check_order(self, symbol):
        """Check trade position of the asset 

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        'Buy' : str
            open and Long position
        'Sell' : str
            open and Short position
        'Closed' : str
            closed position
        """

        try:
            open_positions = self.con.get_model(models=['OpenPosition'], summary=False)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| check_order | {symbol} |')
        open_positions_dict = {i['currency']: i['isBuy'] for i in open_positions['open_positions']}
        if symbol in open_positions_dict:
            if open_positions_dict[symbol] is True:
                return 'Buy'
            elif open_positions_dict[symbol] is False:
                return 'Sell'
        else:
            return 'Closed'

    def open_long(self, symbol, amount):
        """Open Long positions for the asset

        Parameters
        ----------
        symbol : str
            symbol of the asset
        amount : int
            position amount to open trade
        """

        try:
            self.con.create_market_buy_order(symbol, amount)
            time.sleep(1)
            if self.check_order(symbol) == 'Buy':
                self.current_positions[symbol] = 1
                self.log_trade(f"| {self.get_current_time()} |*** LONG ***| {symbol} | amount:{amount} |")
            else:
                self.log_trade(f'| {self.get_current_time()}|*** FAIL ***| open_long | {symbol} is not long |')
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| open_long | {symbol} |')

    def open_short(self, symbol, amount):
        """Open Short positions for the asset

        Parameters
        ----------
        symbol : str
            symbol of the asset
        amount : int
            position amount to open trade
        """

        try:
            self.con.create_market_sell_order(symbol, amount)
            time.sleep(1)
            if self.check_order(symbol) == 'Sell':
                self.current_positions[symbol] = -1
                self.log_trade(f"| {self.get_current_time()} |*** SHORT ***| {symbol} | amount:{amount} |")
            else:
                self.log_trade(f'| {self.get_current_time()}|*** FAIL ***| open_short | {symbol} is not short |')
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| open_short | {symbol} |')

    def close_symbol(self, symbol):
        """Close open position for the asset

        Parameters
        ----------
        symbol : str
            symbol of the asset
        """

        try:
            self.con.close_all_for_symbol(symbol)
            time.sleep(1)
            if self.check_order(symbol) == 'Closed':
                self.current_positions[symbol] = 0
                self.log_trade(f"| {self.get_current_time()} |*** CLOSE ***| {symbol} |")
            else:
                self.log_trade(f'| {self.get_current_time()}|*** FAIL ***| close_symbol | {symbol} is not closed |')
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| close_symbol | {symbol} |')

    def close_to_long(self, symbol, amount):
        """Close current position then open Long position

        Parameters
        ----------
        symbol : str
            symbol of the asset
        amount : int
            position amount to open trade
        """

        try:
            self.close_symbol(symbol)
            self.open_long(symbol, amount)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| close_to_long | {symbol} |')

    def close_to_short(self, symbol, amount):
        """Close current position then open Short position

        Parameters
        ----------
        symbol : str
            symbol of the asset
        amount : int
            position amount to open trade
        """

        try:
            self.close_symbol(symbol)
            self.open_short(symbol, amount)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| close_to_short | {symbol} |')

    def close_all_positions(self):
        """Closing all open positions."""

        try:
            self.con.close_all()
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| close_all_positions |')
        for symbol in self.symbol_list:
            self.current_positions[symbol] = 0
        self.log_trade(f"| {self.get_current_time()} |***** CLOSED ALL POSITIONS *****|")
        print("-"*50)

    def get_current_time(self):
        """get current time based on predetermined time zone"""

        return datetime.now(pytz.timezone(self.time_zone))

    def get_candles_time(self, symbol):
        """ get the lastest candle time for the asset, convert to predetermined time zone 

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        candle_time : datetime.datetime
            the lastest candle time
        """

        try:
            candle_time = self.con.get_candles(symbol, period=self.time_frame, number=1, with_index=True, columns=['date']).index[-1]
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| get_candles_time | {symbol} |')
        candle_time = pytz.UTC.localize(candle_time.to_pydatetime())
        candle_time = candle_time.astimezone(pytz.timezone(self.time_zone))
        return candle_time

    def get_data(self, symbol):
        """ get historical prices (for calculating technical indicators)

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        data : pandas.DataFrame
            DataFrame contains historical prices
        """

        try:
            data = self.con.get_candles(
                symbol, period=self.time_frame, number=self.min_len)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ****| get_data | {symbol} |')
        return data

    def get_add_candle_time(self):
        """get timedelta that use to approximate time for next the candles
        e.g. "m15" --> return timedelta(minutes=15)

        Returns
        -------
        datetime.timedelta
            timedelta to approximate nexxt candle time
        """

        if self.time_frame[0] == 'm':
            return timedelta(minutes=int(self.time_frame[1:]))
        elif self.time_frame[0] == 'H':
            return timedelta(hours=int(self.time_frame[1:]))
        elif self.time_frame[0] == 'D':
            return timedelta(days=int(self.time_frame[1:]))
        elif self.time_frame[0] == 'W':
            return timedelta(days=7*int(self.time_frame[1:]))
        elif self.time_frame[0] == 'M':
            return timedelta(days=30*int(self.time_frame[1:]))

    def is_trading_time(self):
        """check whether it's time for trading

        Returns
        -------
        bool
            True if current time less than end time 
            False otherwise
        """
        return self.get_current_time() >= self.start_time and self.get_current_time() <= self.end_time

    def get_signal(self, symbol):
        """implement in child class (different for each strategy).

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        int 
            trade signal
            1 --> LONG
            0 --> NEUTRAL
            -1 --> SHORT
        """

        pass

    def check_margin(self, symbol):
        """check if account has enough margin to make trade.

        Parameters
        ----------
        symbol : str
            symbol of the asset
        """

        try:
            account_info = self.con.get_model(models=['Account'], summary=False)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| check_margin |')
        account_info = account_info['accounts'][0]
        account = account_info['usableMargin'] > (self.margin_pct*account_info['equity']) # check overall account margin
        trade = account_info['usableMargin'] > (self.trade_amount[symbol]*self.margin_dict[symbol]) # check margin for specific symbol
        return account and trade # return True if both margin check passed

    def create_csv(self):
        """create csv file for logging portfolio info."""

        name = datetime.now(pytz.UTC).strftime("%Y_%m_%d") + "_PortfolioLog.csv"
        with open(os.path.join(self.log_path, name), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.log_header)

    def log_portfolio(self):
        """write current portfolio info to csv file"""

        name = self.get_current_time().strftime("%Y_%m_%d") + "_PortfolioLog.csv"
        if not os.path.exists(os.path.join(self.log_path, name)):
            self.create_csv()
        try:
            account_info = self.con.get_model(models=['Account'], summary=False)
        except:
            self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| log_portfolio |')
        account_info = account_info['accounts'][0]
        val = []
        for head in self.log_header:
            if head == 'date':
                continue
            val.append(account_info[head])
        val.append(pd.Timestamp(self.get_current_time()))
        with open(os.path.join(self.log_path, name), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(val)
        # self.log_trade(f"| {self.get_current_time()} | {self.next_restart_time - self.get_current_time()} until next restart | {self.end_time - self.get_current_time()} until termination |")
        self.log_trade(f"| {self.get_current_time()} | {self.end_time - self.get_current_time()} until termination |")
        self.log_time += self.add_log_time

    def log_trade(self, content):
        """write content to text file.

        Parameters
        ----------
        content : str
            content to write 
        """

        name = self.get_current_time().strftime("%Y_%m_%d") + "_TradeLog.txt"
        with open(os.path.join(self.log_path, name), 'a') as file:
            file.write(content + '\n')
        print(content)

    def check_positions(self):
        """check if all positions are closed (after terminate trading). If not, try closing all open positions"""

        attempts = 1
        while True:
            print('Checking positions.....', end='')
            if self.con.get_open_positions().empty:
                print('No open positions.')
                break
            else:
                print(f'Try closing all positions(attempt{attempts}).')
                self.con.close_all()
                time.sleep(1)
                attempts += 1
                continue

    def trading(self):
        """Begin trading process"""

        while self.is_trading_time():
            
            # if self.get_current_time() >= self.next_restart_time:
            #     self.log_trade(f'| {self.get_current_time()} |*** scheduled restarting ***|')
            #     self.restart_python_script()
                
            if self.get_current_time() >= self.update_pos_time:
                self.update_current_positions()
                time.sleep(1)
                
            if self.get_current_time() >= self.update_amount_time:
                self.update_trade_amount()
                time.sleep(1)
                
            if self.get_current_time() >= self.log_time:
                self.log_portfolio()
                time.sleep(1)
                
            for symbol in self.symbol_list:
                
                if self.get_current_time() >= self.next_candles_time[symbol]:
                    candles_time = self.get_candles_time(symbol)
                    
                    if candles_time >= self.next_candles_time[symbol]:
                        if self.check_margin(symbol):
                            signal = self.get_signal(symbol)
                            current_pos = self.current_positions[symbol]
                            amount = self.trade_amount[symbol]
                            # 1 --> LONG
                            # 0 --> NEUTRAL
                            # -1 --> SHORT
                            if signal == 1:
                                if current_pos == 1:
                                    self.log_trade(f"| {self.get_current_time()} | {symbol} | already LONG |")
                                elif current_pos == -1:
                                    self.close_to_long(symbol, amount)
                                elif current_pos == 0:
                                    self.open_long(symbol, amount)
                                    
                            elif signal == -1:
                                if current_pos == -1:
                                    self.log_trade(f"| {self.get_current_time()} | {symbol} | already SHORT |")
                                elif current_pos == 1:
                                    self.close_to_short(symbol, amount)
                                elif current_pos == 0:
                                    self.open_short(symbol, amount)
                            
                            elif signal == 0:    
                                if current_pos == 0:
                                    self.log_trade(f"| {self.get_current_time()} | {symbol} | already NEUTRAL |")
                                elif current_pos == 1:
                                    self.close_symbol(symbol)
                                elif current_pos == -1:
                                    self.close_symbol(symbol)
                                    
                            self.next_candles_time[symbol] = candles_time + self.add_candle_time
                            time.sleep(1)
                        else:
                            self.log_trade(f'| {self.get_current_time()} |*** Not enough margin to trade ***|')
                            continue
                    else:
                        time.sleep(5)
                        continue
                else:
                    continue
                
        self.log_trade("-"*50)
        self.log_trade(f'| {self.get_current_time()} |***** Finishing trading *****|')
        self.close_all_positions()
        time.sleep(5)
        self.check_positions()
        self.log_portfolio()
        self.log_trade(f'| {self.get_current_time()} |***** TERMINATE TRADING *****|')
        self.log_trade("="*50)
        self.disconnect()
        if not self.is_trading_time():
            sys.exit()
        
    def restart_python_script(self):
        self.log_trade(f'| {self.get_current_time()} |*** restarting python script ***|')
        os.execv(sys.executable, ['python'] + sys.argv)

    def start_trading(self):
        """Start algorithmic trading"""
        if self.is_trading_time():
            # self.initiate()
            # self.trading()
            try:
                self.initiate()
                self.trading()
            except:
                self.log_trade(f'| {self.get_current_time()} |*** ERROR ***| start_trading |')
                self.restart_python_script()
        else:
            self.log_trade(f'| {self.get_current_time()} | NOT trading time | Exiting |')
            sys.exit()
                
# -------------------------------------------------------------------------------------------------------------------#

# Random Trade


class RandomTrader(FXCMTrader):
    def __init__(self, TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                 start_time, end_time, position_pct=0.02, margin_pct=0.5,
                 account_type='DEMO', time_zone="UTC", log_path='log_files',
                 log_header=['balance', 'equity', 'dayPL', 'usdMr', 'usableMargin', 'date'],
                 log_freq_min=10, update_amount_freq_min=20, #restart_freq_min=60, 
                 update_pos_freq_min=20):

        super().__init__(TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                         start_time, end_time, position_pct, margin_pct,
                         account_type, time_zone, log_path, log_header,
                         log_freq_min, update_amount_freq_min, #restart_freq_min, 
                         update_pos_freq_min)

    def get_signal(self, symbol):
        return np.random.choice([1, -1])

# -------------------------------------------------------------------------------------------------------------------#

# SMA crossover


class SMATrader(FXCMTrader):
    def __init__(self, short_period, long_period,
                 TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                 start_time, end_time, position_pct=0.02, margin_pct=0.5,
                 account_type='DEMO', time_zone="UTC", log_path='log_files',
                 log_header=['balance', 'equity', 'dayPL', 'usdMr', 'usableMargin', 'date'],
                 log_freq_min=10, update_amount_freq_min=20, #restart_freq_min=60, 
                 update_pos_freq_min=20):

        super().__init__(TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                         start_time, end_time, position_pct, margin_pct,
                         account_type, time_zone, log_path, log_header,
                         log_freq_min, update_amount_freq_min, #restart_freq_min, 
                         update_pos_freq_min)
        self.short_period = short_period
        self.long_period = long_period

    def get_signal(self, symbol):
        """implement in child class (different for each strategy).

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        int 
            trade signal
            1 --> LONG
            0 --> NEUTRAL
            -1 --> SHORT
        """

        data = self.get_data(symbol)
        data = self.get_sma(data)
        # cut to the last 10 row for faster run time
        data = data.iloc[-10:]
        data['signal'] = np.where(data['ma_short'] > data['ma_long'], 1, np.where(
            data['ma_short'] < data['ma_long'], -1, 0))
        return data['signal'].iloc[-1]

    def get_sma(self, data):
        """"calculate Simple Moving Average

        Parameters
        ----------
        data : pandas.DataFrame
            input data

        Returns
        -------
        data : pandas.DataFrame
            data with SMA        
        """
        data['midclose'] = data[['bidclose', 'askclose']].mean(axis=1)
        data['ma_short'] = ta.trend.sma_indicator(
            close=data['midclose'], window=self.short_period)
        data['ma_long'] = ta.trend.sma_indicator(
            close=data['midclose'], window=self.long_period)
        data.dropna(axis=0, inplace=True)
        return data

# -------------------------------------------------------------------------------------------------------------------#

# Multiple MA crossover


class MultiMATrader(FXCMTrader):
    def __init__(self, params_path,
                 TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                 start_time, end_time, position_pct=0.02, margin_pct=0.5,
                 account_type='DEMO', time_zone="UTC", log_path='log_files',
                 log_header=['balance', 'equity', 'dayPL', 'usdMr', 'usableMargin', 'date'],
                 log_freq_min=10, update_amount_freq_min=20, #restart_freq_min=60, 
                 update_pos_freq_min=20):

        super().__init__(TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                         start_time, end_time, position_pct, margin_pct,
                         account_type, time_zone, log_path, log_header,
                         log_freq_min, update_amount_freq_min, #restart_freq_min, 
                         update_pos_freq_min)
        self.params_path = params_path
        self.parameters = self.load_parameters()

    def load_parameters(self):
        with open(self.params_path, 'r') as f:
            parameters = json.load(f)
        return parameters

    def get_signal(self, symbol):
        """implement in child class (different for each strategy).

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        int 
            trade signal
            1 --> LONG
            0 --> NEUTRAL
            -1 --> SHORT
        """

        # dictionary maps name --> function
        indicators = {
            'SMA': talib.SMA,
            'EMA': talib.EMA,
            'KAMA': talib.KAMA,
            'MIDPOINT': talib.MIDPOINT,
            'MIDPRICE': talib.MIDPRICE,
            'TRIMA': talib.TRIMA,
            'WMA': talib.WMA
        }
        
        data = self.get_data(symbol) # get data
        columns = ['open', 'high', 'low', 'close']
        for col in columns:
            data[col] = data[['bid'+col, 'ask'+col]].mean(axis=1)
        
        for name, func in indicators.items(): # compute all indicators
            params = self.parameters[name][symbol]
            for i in range(len(params)):
                if name == 'MIDPRICE':
                    data[f"{name}_{i+1}"] = func(data['high'],data['low'], params[i])
                else:
                    data[f"{name}_{i+1}"] = func(data['close'], params[i])
        
        data = data.iloc[-5:] # cut to the last 5 rows for faster run time
        signal_col = [] # getting all signals
        for name in indicators.keys():
            data[f"{name}_signal"] = np.where(data[f"{name}_1"] > data[f"{name}_2"], 1,
                                     np.where(data[f"{name}_1"] < data[f"{name}_2"], -1, np.nan))
            data[f"{name}_signal"] = data[f"{name}_signal"].ffill().fillna(0)
            signal_col.append(f"{name}_signal")
        
        signals = data[signal_col].mode(axis=1).iloc[:, 0] # get signals mode (i.e., majority vote)
        return int(signals[-1]) # return the last signal

# -------------------------------------------------------------------------------------------------------------------#

# Three Indicators --> EMA, RSI, Bollinger Bands


class ThreeIndicatorsTrader(FXCMTrader):
    def __init__(self, params_path,
                 TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                 start_time, end_time, position_pct=0.02, margin_pct=0.5,
                 account_type='DEMO', time_zone="UTC", log_path='log_files',
                 log_header=['balance', 'equity', 'dayPL', 'usdMr', 'usableMargin', 'date'],
                 log_freq_min=5, update_amount_freq_min=30, #restart_freq_min=60, 
                 update_pos_freq_min=30):

        super().__init__(TOKEN, symbol_list, margin_dict, time_frame, min_len, 
                         start_time, end_time, position_pct, margin_pct,
                         account_type, time_zone, log_path, log_header,
                         log_freq_min, update_amount_freq_min, #restart_freq_min, 
                         update_pos_freq_min)
        self.params_path = params_path
        self.parameters = self.load_parameters()
        
    def load_parameters(self):
        with open(self.params_path, 'r') as f:
            parameters = json.load(f)
        return parameters

    def get_signal(self, symbol):
        """implement in child class (different for each strategy).

        Parameters
        ----------
        symbol : str
            symbol of the asset

        Returns
        -------
        int 
            trade signal
            1 --> LONG
            0 --> NEUTRAL
            -1 --> SHORT
        """
        
        data = self.get_data(symbol) # get data
        columns = ['open', 'high', 'low', 'close']
        for col in columns:
            data[col] = data[['bid'+col, 'ask'+col]].mean(axis=1)
            
        # get indicators
        params = self.parameters['ThreeInds'][symbol]
        data['ema'] = talib.EMA(data.close, params[0])
        data['rsi'] = talib.RSI(data.close, params[1])
        data['bb_upper'], data['bb_mid'], data['bb_lower'] = talib.BBANDS(data.close, params[2])
        data.dropna(axis=0, inplace=True)
        
        # get signals
        data['ema_signal'] = np.where(data.close > data.ema, 1, -1)

        data['rsi_signal'] = np.where(data.rsi > 70, -1, np.where(data.rsi < 30, 1, np.nan))
        data['rsi_less_50'] = np.where(data.rsi < 50, 1, -1)
        data.loc[(data.rsi_less_50 * data.rsi_less_50.shift(1) < 0), 'rsi_signal'] = 0
        data.rsi_signal.ffill(inplace=True)
        data.rsi_signal.fillna(0, inplace=True)
        
        data['bb_signal'] = np.where(data.close > data.bb_upper, -1, 
                                    np.where(data.close < data.bb_lower, 1, np.nan))
        data['bb_less_mid'] = np.where(data.close < data.bb_mid, 1, -1)
        data.loc[(data.bb_less_mid * data.bb_less_mid.shift(1) < 0), 'bb_signal'] = 0
        data.bb_signal.ffill(inplace=True)
        data.bb_signal.fillna(0, inplace=True)

        data = data.iloc[-5:, :]
        signal_cols = ['ema_signal', 'rsi_signal', 'bb_signal']
        signals = data.loc[:,signal_cols].mode(axis=1).iloc[:, 0]
        
        return int(signals[-1])