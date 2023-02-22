import json
import talib
from base_trader import *


class MAMATrader(BaseTrader):
    
    def __init__(self, symbols, timeframe, parameters, pct_equity_trade,
                 margin_csv, n_warmup, log_freq, update_amount_freq, update_pos_freq):
        
        super().__init__(symbols, timeframe, pct_equity_trade, margin_csv, 
                         n_warmup, log_freq, update_amount_freq, update_pos_freq)
        self.parameters = parameters
        # (fastlimit, slowlimit)
        
    def define_strategy(self, symbol):
        self.raw_data_dict[symbol]['mama'], self.raw_data_dict[symbol]['fama'] = talib.MAMA(self.raw_data_dict[symbol][symbol],
                                                                                            self.parameters[symbol][0],
                                                                                            self.parameters[symbol][1])
        self.raw_data_dict[symbol]['signal'] = np.where(self.raw_data_dict[symbol]['mama'] > self.raw_data_dict[symbol]['fama'], 1,
                                                        np.where(self.raw_data_dict[symbol]['mama'] < self.raw_data_dict[symbol]['fama'], -1, np.nan)) 
        self.raw_data_dict[symbol]['signal'].ffill(inplace=True)               
        self.signal_dict[symbol] = int(self.raw_data_dict['signal'].iloc[-1])
        
    
if __name__ == "__main__":
    symbols = token.my_assets
    timeframe = token.set_time_frame
    print(timeframe)
    pct_equity_trade = 0.1
    margin_csv = token.margin_file
    n_warmup = 1000
    log_freq = 90
    update_amount_freq = 80
    update_pos_freq = 70
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'my_parameters.json'), 'r') as f:
        parameters = json.load(f)
    
    trader = MAMATrader(symbols, timeframe, parameters, pct_equity_trade, margin_csv, 
                        n_warmup, log_freq, update_amount_freq, update_pos_freq)
    trader.begin_trading()
    target_time = dt.datetime.now() + dt.timedelta(hours=1)
    while dt.datetime.now() < target_time:
        time.sleep(60*60)
        target_time = dt.datetime.now() + dt.timedelta(hours=1)
    trader.close_connection()