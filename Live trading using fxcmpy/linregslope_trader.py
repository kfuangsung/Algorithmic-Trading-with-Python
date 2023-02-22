from base_trader import *

class LinRegSlopeTrader(BaseTrader):
    def __init__(self, symbols, timeframe, linreg_periods, pct_equity_trade,
                 margin_csv, n_warmup, log_freq, update_amount_freq):
        super().__init__(symbols, timeframe, pct_equity_trade, margin_csv, 
                         n_warmup, log_freq, update_amount_freq)
        self.linreg_periods = linreg_periods
        
    def define_strategy(self, symbol):
        slope = talib.LINEARREG_SLOPE(self.raw_data_dict[symbol].values.flatten(),
                                      timeperiod=self.linreg_periods[symbol])[-1]
        self.signal_dict[symbol] = copy.deepcopy(int(np.where(slope > 0, 1, 2)))
        
        
if __name__ == "__main__":
    symbols = token.my_assets
    timeframe = token.set_time_frame
    print(timeframe)
    pct_equity_trade = 0.05
    margin_csv = token.margin_file
    n_warmup = 300
    log_freq = 10
    update_amount_freq = 30
    linreg_periods = {'EUR/USD':100, 'USD/JPY':50, 'GBP/USD':150, 'USOil':100, 'SPX500':150,
                      'VOLX':150, 'XAU/USD':100, 'Bund':250, 'BTC/USD':50}
    
    trader = LinRegSlopeTrader(symbols, timeframe, linreg_periods, pct_equity_trade, 
                               margin_csv, n_warmup, log_freq, update_amount_freq)
    # trader.verify_close_all()
    # trader.begin_trading()
    # time.sleep(60*60*6)
    trader.verify_close_all()
    trader.account_report()
    trader.close_connection()