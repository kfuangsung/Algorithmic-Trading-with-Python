import sys
# import traceback
import csv
import copy
import datetime as dt
import os
import time
import fxcmpy 
import numpy as np
import pandas as pd 
import timeout_decorator
import fxcmtoken as token


class BaseTrader:
    
    
    def __init__(self, 
                 symbols, 
                 timeframe,
                 pct_equity_trade, 
                 margin_csv, 
                 n_warmup, 
                 log_freq, 
                 update_amount_freq,
                 update_pos_freq,
                 check_duplicate_freq):
        
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.symbols = symbols
        self.timeframe = timeframe
        self.pct_equity_trade = pct_equity_trade
        self.margin_dict = self.load_margin(margin_csv)
        self.n_warmup = n_warmup
        self.log_freq = log_freq
        self.n_assets = len(self.symbols)
        self.log_time_delta = dt.timedelta(minutes=self.log_freq)
        self.next_log = None
        self.need_log = True
        self.update_amount_freq = update_amount_freq
        self.update_amount_delta = dt.timedelta(minutes=self.update_amount_freq)
        self.next_update = None
        self.need_update = True
        self.update_pos_freq = update_pos_freq
        self.update_pos_delta = dt.timedelta(minutes=self.update_pos_freq)
        self.next_pos = None
        self.need_pos = True
        self.check_duplicate_freq = check_duplicate_freq
        self.check_duplicate_delta = dt.timedelta(minutes=self.check_duplicate_freq)
        self.next_duplicate = None
        self.need_duplicate = True
        self.date_offset = None
        self.con = None
        self.raw_data_dict = {}
        self.last_bar_dict = {}
        self.signal_dict = {}
        self.position_dict = {}
        self.amount_dict = {}
        self.fail_to_subscribe_tickers = []
        self.next_subscribe_time = dt.datetime.utcnow()
        self.need_subscribe = False
        
        self.set_date_offset()
        self.set_next_log()
        self.set_next_update()
        self.set_next_position()
        self.set_next_duplicate()
        self.make_connection()
        self.initiate_info()
        self.update_current_positions()
        self.check_duplicate_positions()
        self.get_warmup_data()
        self.update_amounts()
        
    
    @timeout_decorator.timeout(60)
    def init_fxcmpy(self):
        self.con = fxcmpy.fxcmpy(access_token=token.DEMO_TOKEN, log_level='error', server='demo')
             
             
    def make_connection(self, n_count=0):
        counter = n_count
        if counter > 5:
            self.restart_python_script()
            
        self.log_report('-'*50)
        self.log_report("Connecting...")
        counter += 1
        try:
            self.init_fxcmpy()
            is_connected = self.con.is_connected()
            if is_connected:
                self.log_report('CONNECTED')
                self.log_report(f"connection status: {self.con.connection_status}")
                self.log_report('-'*50)
            else:
                self.log_report('FAILED to connect...try again in 60 seconds')
                time.sleep(60)
                self.make_connection(n_count=counter)
        except Exception as e:
            self.log_report('ERROR to connect...try again in 60 seconds')
            self.log_report(e)
            time.sleep(60)
            self.make_connection(n_count=counter)
        
            
    @timeout_decorator.timeout(30)
    def close_fxcmpy(self):
        self.con.close()
        
        
    def close_connection(self, n_count=0):
        counter = n_count
        if counter > 5:
            self.restart_python_script()
        
        self.log_report('-'*50)
        self.log_report("Disconnecting...")
        counter += 1
        try:
            self.close_fxcmpy()
            if not self.con.is_connected():
                self.log_report('DISCONNECTED')
                self.log_report(f"connection status: {self.con.connection_status}")
                self.log_report('-'*50)
                return
            else:
                self.log_report('FAILED to close connect...try again in 10 seconds')
                time.sleep(10)
                self.close_connection(n_count=counter)
        except Exception as e:
            self.log_report('ERROR | close connection...try again in 10 seconds')
            self.log_report(e)
            time.sleep(10)
            self.close_connection(n_count=counter)
    
    
    def restart_python_script(self):
        self.log_report(f'*** restarting python script ***')
        os.execv(sys.executable, ['/home/azureuser/anaconda3/envs/fxcm/bin/python'] + sys.argv)

    
    def set_date_offset(self):
        if not self.timeframe in token.time_frame:
            raise ValueError(f"Invalid timeframe: {self.timeframe}")
            
        if 'm' in self.timeframe:
            self.date_offset = self.timeframe[1:] + 'min'
        
        elif 'H' in self.timeframe:
            self.date_offset = self.timeframe[1:] + 'H'
        
        elif 'D' in self.timeframe:
            self.date_offset = self.timeframe[1:] + 'D'
        
        elif 'W' in self.timeframe:
            self.date_offset = self.timeframe[1:] + 'W'
        
        elif 'M' in self.timeframe:
            self.date_offset = self.timeframe[1:] + 'M'
            
            
    def load_margin(self, file_csv):
        return pd.read_csv(os.path.join(self.dir_path, file_csv), index_col=['symbol']).to_dict()['margin']
    
    
    def set_next_log(self):
        now = dt.datetime.utcnow()
        new_min = self.log_freq * (now.minute // self.log_freq)
        self.next_log = now.replace(minute=new_min, second=0) + self.log_time_delta
        
    
    def set_next_update(self):
        now = dt.datetime.utcnow()
        new_min = self.update_amount_freq * (now.minute // self.update_amount_freq)
        self.next_update = now.replace(minute=new_min, second=0) + self.update_amount_delta
        
        
    def set_next_position(self):
        now = dt.datetime.utcnow()
        new_min = self.update_pos_freq * (now.minute // self.update_pos_freq)
        self.next_pos = now.replace(minute=new_min, second=0) + self.update_pos_delta
        
    
    def set_next_duplicate(self):
        now = dt.datetime.utcnow()
        new_min = self.check_duplicate_freq * (now.minute // self.check_duplicate_freq)
        self.next_duplicate = now.replace(minute=new_min, second=0) + self.check_duplicate_delta
        
        
    def set_next_subscribe(self):
        now = dt.datetime.utcnow()
        new_min = 5 * (now.minute // 5)
        self.next_subscribe_time = now.replace(minute=new_min, second=0) + dt.timedelta(minutes=5)
        
        
    def initiate_info(self):
        for sym in self.symbols:
            self.raw_data_dict[sym] = None
            self.last_bar_dict[sym] = None
            self.signal_dict[sym] = None
            self.position_dict[sym] = 0
    
    
    def get_fxcm_accounts_summary(self, n_count=0):
        counter = n_count
        if counter > 5:
            # self.close_connection()
            return self.restart_python_script()
        
        counter += 1
        try:
            current_equity = self.con.get_accounts_summary(kind='list')[0]['equity']
            if current_equity is None:
                raise ValueError('equity is None')
            
            return current_equity
        
        except Exception as e:
            self.log_report(f'ERROR | get_accounts_summary | attempt{counter}.....retrying in 10 seconds')
            self.log_report(e)
            time.sleep(10)
            return self.get_fxcm_accounts_summary(n_count=counter)
                
    
    def update_amounts(self, n_count=0):
        
        counter = n_count
        if counter > 5:
            return self.restart_python_script()
        
        self.log_report("Update trade amounts...BEGIN")
        counter += 1
        try:
            current_equity = self.get_fxcm_accounts_summary()
            for sym in self.symbols:
                self.amount_dict[sym] = int(np.floor((current_equity * self.pct_equity_trade) / (self.n_assets * self.margin_dict[sym])))
            
        except Exception as e:
            self.log_report("ERROR | update_amounts.....retrying in 10 seconds")
            self.log_report(e)
            time.sleep(10)
            return self.update_amounts(n_count=counter)
            
        else:
            self.log_report("Update trade amounts...DONE")
            return
        
    
    def get_fxcm_open_positions(self, kind='dataframe', n_count=0):
        counter = n_count
        if counter > 5:
            # self.close_connection()
            return self.restart_python_script()
        
        counter += 1
        try:
            open_positions = self.con.get_open_positions(kind=kind) 
            
            if open_positions is None:
                raise ValueError("open position is None")
            
            _ = len(open_positions)
            
            return open_positions
        
        except Exception as e:
            self.log_report(f'ERROR | get_open_positions | attempt{counter}.....retrying in 10 seconds')
            self.log_report(e)
            time.sleep(10)
            return self.get_fxcm_open_positions(kind=kind, n_count=counter)
    
    
    def update_current_positions(self, n_count=0):
        """get current positions of all symbols"""
        
        counter = n_count
        if counter > 5:
            # self.close_connection()
            return self.restart_python_script()

        self.log_report('Update current positions...BEGIN')
        counter += 1
        
        try:
            open_positions = self.get_fxcm_open_positions(kind='list')
            open_symbols = []
            
            if len(open_positions) != 0:
                for pos in open_positions: 
                    self.position_dict[pos['currency']] = 1 if pos['isBuy'] else 2
                    open_symbols.append(pos['currency'])
                    
            for sym in self.symbols:
                if not sym in open_symbols:  self.position_dict[sym] = 0
        
        except Exception as e:
            self.log_report("ERROR | update_current_positions.....retrying in 10 seconds")
            self.log_report(e)
            time.sleep(10)
            return self.update_current_positions(n_count=counter)
        
        else:
            self.log_report('Update current positions...DONE')
            self.log_report(self.position_dict)
            return
   
    
    def check_duplicate_positions(self, n_count=0):
        
        counter = n_count
        if counter > 5:
            # self.close_connection()
            return self.restart_python_script()
        
        self.log_report('Check duplicate positions...BEGIN')
        counter += 1
        
        try:
            open_positions = self.get_fxcm_open_positions(kind='dataframe')
            if len(open_positions) > 0:
                open_symbols = set(open_positions.currency.values)
                for ticker in open_symbols:
                    pos = open_positions.query('currency == @ticker')
                    pos_dict = pos.to_dict(orient='records')
                    if len(pos_dict) > 1:
                        for p_i in pos_dict[:-1]:
                            try:
                                p = self.con.get_open_position(p_i.get('tradeId'))
                                p.close()
                                p.close(p.get_amount()*1000)
                                self.log_report(f"{ticker} | closed duplicate positions")
                            except Exception as e:
                                self.log_report(f"ERROR | {ticker} | cannot close duplicate positions")
                                self.log_report(e)
        
        except Exception as e:
            self.log_report(f"ERROR | check_duplicate_positions.....retrying in 10 seconds")
            self.log_report(e)
            time.sleep(10)
            return self.check_duplicate_positions(n_count=counter)
            
        else:
            self.log_report('Check duplicate positions...DONE')
            return
        
        
    def get_historical_candles(self, symbol, n_count=0):
        counter = n_count
        if counter > 5:
            # self.close_connection()
            return self.restart_python_script()
        
        counter += 1
        try:
            candles = self.con.get_candles(instrument=symbol, period=self.timeframe, 
                                           number=self.n_warmup, with_index=True, 
                                           columns=['bidclose', 'askclose',
                                                    'bidhigh', 'askhigh',
                                                    'bidlow', 'asklow'])    
            candles = candles.assign(
                close=lambda x: np.mean([x.bidclose, x.askclose], axis=0),
                high=lambda x: np.mean([x.bidhigh, x.askhigh], axis=0),
                low=lambda x: np.mean([x.bidlow, x.asklow], axis=0) 
                )
            candles = candles.loc[:,['close', 'high', 'low']].copy(deep=True)
            idx = candles.index[-1]
            
            if candles is None:
                raise ValueError('get candles return None')

            if candles.empty:
                raise ValueError('get candles return Empty dataframe')
            
            if len(candles) == 0:
                raise ValueError('get candle return length of zero')
            
            return idx, candles
        
        except Exception as e:
            self.log_report(f'ERROR | get_candles | {symbol} | attempt{counter}.....retrying in 10 seconds')
            self.log_report(e)
            time.sleep(10)
            return self.get_historical_candles(symbol=symbol, n_count=counter)
            
        
    def get_warmup_data(self, n_count=0):
        counter = n_count
        if counter > 5:
            return self.restart_python_script()
        
        self.log_report("Warming up data.....BEGIN")
        counter += 1
        for sym in self.symbols:
            try:
                idx, candles = self.get_historical_candles(sym)
                self.raw_data_dict[sym] = candles
                self.last_bar_dict[sym] = idx
            except Exception as e:
                self.log_report(f'ERROR | get_warmup_data......retrying in 10 seconds')
                self.log_report(e)
                time.sleep(10)
                return self.get_warmup_data(n_count=counter)
     
        # del idx, candles
        self.log_report("Warming up data.....DONE")
    
    
    def define_strategy(self, symbol):
        # define in child class
        pass
    
    
    def open_long_position(self, symbol, trade_amount, n_count=0):
        counter = n_count
        if counter > 3:
            return
       
        counter += 1 
        try:
            self.con.create_market_buy_order(symbol=symbol, amount=trade_amount)
            self.position_dict[symbol] = 1
            self.log_report(f"OPEN LONG.....{symbol}.....{trade_amount} units")
            return
        except Exception as e:
            self.log_report(f'FAILED | open long | {symbol}')
            self.log_report(e)
            time.sleep(5)
            return self.open_long_position(symbol, trade_amount, counter)
        
    
    def open_short_position(self, symbol, trade_amount, n_count=0):
        counter = n_count
        if counter > 3:
            return
    
        counter += 1        
        try:
            self.con.create_market_sell_order(symbol=symbol, amount=trade_amount)
            self.position_dict[symbol] = 2
            self.log_report(f"OPEN SHORT.....{symbol}.....{trade_amount} units")
            return
        except Exception as e:
            self.log_report(f'FAILED | open short | {symbol}')
            self.log_report(e)
            time.sleep(5)
            return self.open_short_position(symbol, trade_amount, counter)
    
    
    def close_position(self, symbol, n_count=0):
        counter = n_count
        if counter > 3:
            return
        counter += 1
        try:
            self.con.close_all_for_symbol(symbol=symbol)
            self.position_dict[symbol] = 0
            self.log_report(f"NEUTRAL.....{symbol}")
            return
        except Exception as e:
            self.log_report(f'FAILED | close position | {symbol}')
            self.log_report(e)
            time.sleep(5)
            return self.close_position(symbol, counter)
            

    def close_to_long(self, symbol, trade_amount):
        
        self.close_position(symbol)
        time.sleep(10)
        self.open_long_position(symbol, trade_amount)
        return
        
        # try:
        #     self.con.close_all_for_symbol(symbol=symbol)
        #     self.position_dict[symbol] = 0
        #     self.log_report(f"CLOSE SHORT.....{symbol}")
        # except Exception as e:
        #     self.log_report(f'FAILED | close short | {symbol}')
        #     self.log_report(e)
        #     return
        
        # time.sleep(30)
        
        # try:
        #     self.con.create_market_buy_order(symbol=symbol, amount=trade_amount)
        #     self.position_dict[symbol] = 1
        #     self.log_report(f"OPEN LONG.....{symbol}.....{trade_amount} units")
        # except Exception as e:
        #     self.log_report(f'FAILED | open long | {symbol}')
        #     self.log_report(e)
        #     return
        
        
    def close_to_short(self, symbol, trade_amount):
        self.close_position(symbol)
        time.sleep(10)
        self.open_short_position(symbol, trade_amount)
        return
        
        # try:
        #     self.con.close_all_for_symbol(symbol=symbol)
        #     self.position_dict[symbol] = 0
        #     self.log_report(f"CLOSE LONG.....{symbol}")
        # except Exception as e:
        #     self.log_report(f'FAILED | close long | {symbol}')
        #     self.log_report(e)
        #     return
        
        # time.sleep(30)
        
        # try:
        #     self.con.create_market_sell_order(symbol=symbol, amount=trade_amount)
        #     self.position_dict[symbol] = 2
        #     self.log_report(f"OPEN SHORT.....{symbol}.....{trade_amount} units")
        # except Exception as e:
        #     self.log_report(f'FAILED | open short | {symbol}')
        #     self.log_report(e)
        #     return
    
            
    def execute_trade(self, symbol):
        trade_amount = self.amount_dict[symbol]
        
        if self.signal_dict[symbol] == 0: # NEUTRAL
            
            if self.position_dict[symbol] == 0:
                pass
            
            elif self.position_dict[symbol] == 1 or self.position_dict[symbol] == 2:
                self.close_position(symbol)
                    
        elif self.signal_dict[symbol] == 1: # LONG
            
            if self.position_dict[symbol] == 0:
                self.open_long_position(symbol, trade_amount)
                
            elif self.position_dict[symbol] == 1:
                pass
            
            elif self.position_dict[symbol] == 2:
                self.close_to_long(symbol, trade_amount)
                
        elif self.signal_dict[symbol] == 2: # SHORT
            
            if self.position_dict[symbol] == 0:
                self.open_short_position(symbol, trade_amount)
                
            elif self.position_dict[symbol] == 1:
                self.close_to_short(symbol, trade_amount)
                
            elif self.position_dict[symbol] == 2:
                pass
            
        self.signal_dict[symbol] = None
        return


    def log_report(self, msg):
        content = f"{dt.datetime.utcnow()} | {msg}"
        file_name = dt.datetime.utcnow().strftime("%Y_%m_%d") + "_TradeReport.txt"
        file_name = os.path.join(self.dir_path, file_name)
        
        with open(file_name, 'a') as file:
            file.write(content + '\n')
        
        print(content)
        
    
    def get_fxcm_account_info(self, n_count=0):
        counter = n_count
        if counter > 5:
            # self.close_connection()
            return self.restart_python_script()
        
        counter += 1
        try:
            account_info = self.con.get_model(['Account'], summary=False)['accounts'][0] 
            if account_info is None:
                raise ValueError('Account is None')
            
            return account_info
        except Exception as e:
            self.log_report(f'ERROR | get_model | attempt{counter}.....retrying in 10 seconds')
            self.log_report(e)
            time.sleep(10)
            return self.get_fxcm_account_info(n_count=counter)    
    
        
    def account_report(self, n_count=0):
        
        counter = n_count
        if counter > 5:
            return self.restart_python_script()
        
        try:
            self.log_report("Log account report.....BEGIN")
            file_name = dt.datetime.utcnow().strftime("%Y_%m") + "_AccountReport.csv"
            file_name = os.path.join(self.dir_path, file_name)
            account_info = self.get_fxcm_account_info()
            headers = ['accountId', 'balance', 'usdMr', 'mc', 'mcDate', 'accountName','hedging', 
                    'usableMarginPerc', 'equity', 'usableMargin', 'dayPL', 'grossPL', 'date']
            content = [account_info.get(h) for h in headers[:-1]]
            content.append(pd.Timestamp(dt.datetime.utcnow()))
            
            if not os.path.exists(file_name):
                with open(file_name, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerow(content)
            else:
                with open(file_name, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(content) 
        
        except Exception as e:
            self.log_report(f'ERROR | account_report.....retrying in 10 seconds')
            self.log_report(e)
            time.sleep(10)
            return self.account_report(n_count=counter)
        
        else:
            self.log_report("Log account report.....DONE")
            return
    
    
    def stream_data(self, data, dataframe):
        symbol = data['Symbol']
        
        # log account stats
        if (dt.datetime.utcnow() > self.next_log) and self.need_log:
            self.need_log = False
            self.account_report()
            self.set_next_log()
            self.need_log = True
            
        # update trade amounts
        if (dt.datetime.utcnow() > self.next_update) and self.need_update:
            self.need_update = False
            self.update_amounts()
            self.set_next_update()
            self.need_update = True

        # update current positions
        if (dt.datetime.utcnow() > self.next_pos) and self.need_pos:
            self.need_pos = False
            self.update_current_positions()
            self.set_next_position()
            self.need_pos = True
            
        # check duplicate positions
        if (dt.datetime.utcnow() > self.next_duplicate) and self.need_duplicate:
            self.need_duplicate = False
            self.check_duplicate_positions()
            self.set_next_duplicate()
            self.need_duplicate = True
            
        if (dt.datetime.utcnow() > self.next_subscribe_time) and self.need_subscribe:
            self.need_subscribe = False
            self.check_subscribe()
        
        last_tick = dataframe.index[-1]
        if last_tick > self.last_bar_dict[symbol]:
            tick_data = dataframe[dataframe.index > self.last_bar_dict[symbol]]
            if len(tick_data) > 0:
                # tick_data = tick_data.loc[:, ['Bid', 'Ask']].mean(axis=1)
                tick_data = tick_data.assign(
                    close=lambda x: np.mean([x.Bid, x.Ask], axis=0),
                    high=lambda x: x.High,
                    low=lambda x: x.Low
                )
                tick_data = tick_data.loc[:,['close', 'high', 'low']].copy(deep=True)
                tick_data = tick_data.resample(self.date_offset, label='right').last().ffill().iloc[:-1]#.to_frame()
                # tick_data.rename(columns={0: 'close'}, inplace=True)
                tmp = self.raw_data_dict[symbol]
                tmp = pd.concat([tmp, tick_data], axis=0)
                self.raw_data_dict[symbol] = tmp.iloc[-self.n_warmup:, :].copy(deep=True)        

        if self.raw_data_dict[symbol].index[-1] > self.last_bar_dict[symbol]:
            self.last_bar_dict[symbol] = copy.deepcopy(self.raw_data_dict[symbol].index[-1])
            self.define_strategy(symbol)
            self.execute_trade(symbol)
            
        
    def get_subscribe_data(self, symbol):
        
        try: 
            self.con.subscribe_market_data(symbol=symbol, add_callbacks=(self.stream_data,))
            self.log_report(f"SUBSCRIBED Market Data | {symbol}")
            return
        
        except Exception as e:
            self.log_report(f"ERROR | subscribe market data | {symbol}")
            self.log_report(e)
            return
        
    
    def check_subscribe(self):
        try:
            subscribed_symbols = self.con.get_subscribed_symbols()
            if len(subscribed_symbols) != len(self.symbols):
                self.need_subscribe = True
                self.set_next_subscribe()
                for sym in self.symbols:
                    if sym not in subscribed_symbols:
                        self.get_subscribe_data(sym)
            else:
                self.need_subscribe = False
                
        except Exception as e:
            self.log_report(f"ERROR | check_subscribe | {e}")
            self.need_subscribe = True
            self.set_next_subscribe()
        
    
    def begin_trading(self):
        self.con.set_max_prices(self.n_warmup)
        for sym in self.symbols:
            self.get_subscribe_data(sym)
        self.check_subscribe()

    
    def verify_close_all(self):
        """check if all positions are closed. If not, try closing all open positions"""

        attempts = 1
        while True:
            self.log_report('Checking for open positions.....')
            if self.con.get_open_positions(kind='dateframe').empty:
                self.log_report('No open positions')
                break
            else:
                self.log_report(f'Try closing all positions(attempt{attempts})')
                self.con.close_all()
                time.sleep(30)
                attempts += 1
                continue
