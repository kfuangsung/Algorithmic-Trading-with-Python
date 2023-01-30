import json
import talib
import bottleneck as bn
from base_trader import *

class DemaTemaTrader(BaseTrader):
    def __init__(self, symbols, timeframe, parameters, pct_equity_trade,
                 margin_csv, n_warmup, log_freq, update_amount_freq, 
                 update_pos_freq, check_duplicate_freq):
        
        super().__init__(symbols, timeframe, pct_equity_trade, margin_csv, 
                         n_warmup, log_freq, update_amount_freq, 
                         update_pos_freq, check_duplicate_freq)
        self.parameters = parameters
        
        
    def define_strategy(self, symbol):
        n_period = int(self.parameters[symbol])
        
        self.raw_data_dict[symbol] = self.raw_data_dict[symbol].assign(
            # medprice=lambda x: talib.MEDPRICE(x.high, x.low),
            dema=lambda x: talib.DEMA(x.close, n_period),
            tema=lambda x: talib.TEMA(x.close, n_period),
            signal=lambda x: bn.push(np.where(x.tema > x.dema, 1, np.where(x.tema < x.dema, 2, np.nan)), axis=0)
        )
        
        self.signal_dict[symbol] = int(self.raw_data_dict[symbol].signal.values[-1])
        
        
if __name__ == "__main__":
    symbols = token.my_assets
    timeframe = token.set_time_frame
    print(timeframe)
    pct_equity_trade = 0.1
    margin_csv = token.margin_file
    n_warmup = 10000
    log_freq = 70
    update_amount_freq = 70
    update_pos_freq = 70
    check_duplicate_freq = 70
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'my_parameters.json'), 'r') as f:
        parameters = json.load(f)
    
    trader = DemaTemaTrader(symbols, timeframe, parameters, pct_equity_trade, 
                            margin_csv, n_warmup, log_freq, update_amount_freq, 
                            update_pos_freq, check_duplicate_freq)
    trader.begin_trading()
    target_time = dt.datetime.now() + dt.timedelta(hours=1)
    while dt.datetime.now() < target_time:
        time.sleep(60*60)
        target_time = dt.datetime.now() + dt.timedelta(hours=1)
    trader.close_connection()
