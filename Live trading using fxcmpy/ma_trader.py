from base_trader import *

class MATrader(BaseTrader):
    def __init__(self, symbols, timeframe, pct_equity_trade, 
                 margin_csv, n_warmup, log_freq, update_amount_freq):
        super().__init__(symbols, timeframe, pct_equity_trade, margin_csv, 
                         n_warmup, log_freq, update_amount_freq)
        
    def define_strategy(self, symbol):
        ma_short = float(talib.SMA(self.raw_data_dict[symbol].values.flatten(), timeperiod=20)[-1])
        ma_long = float(talib.SMA(self.raw_data_dict[symbol].values.flatten(), timeperiod=self.n_warmup)[-1])
        self.signal_dict[symbol] = copy.deepcopy(int(np.where(ma_short > ma_long, 1, 2)))
        
        
if __name__ == "__main__":
    symbols = token.my_assets
    timeframe = token.set_time_frame
    print(timeframe)
    pct_equity_trade = 0.01
    margin_csv = token.margin_file
    n_warmup = 100
    log_freq = 10
    update_amount_freq = 10
    
    trader = MATrader(symbols, timeframe, pct_equity_trade, margin_csv, n_warmup, 
                      log_freq, update_amount_freq)
    trader.verify_close_all()
    trader.begin_trading()
    time.sleep(60*10)
    trader.verify_close_all()
    trader.account_report()
    trader.close_connection()