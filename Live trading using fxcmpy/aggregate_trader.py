import json
import talib
from scipy import stats
from base_trader import *


class AggregateTrader(BaseTrader):
    def __init__(self, symbols, timeframe, parameters, pct_equity_trade,
                 margin_csv, n_warmup, log_freq, update_amount_freq, 
                 update_pos_freq, check_duplicate_freq):
        
        super().__init__(symbols, timeframe, pct_equity_trade, margin_csv, 
                         n_warmup, log_freq, update_amount_freq, 
                         update_pos_freq, check_duplicate_freq)
        self.parameters = parameters
        
    def define_strategy(self, symbol):
        ht = talib.HT_TRENDLINE(self.raw_data_dict[symbol].values.flatten())
        mama, fama = talib.MAMA(ht, fastlimit=self.parameters[symbol]['mama'], slowlimit=self.parameters[symbol]['mama']/10)
        slope = talib.LINEARREG_SLOPE(ht, timeperiod=self.parameters[symbol]['slope'])
        tsf = talib.TSF(ht, self.parameters[symbol]['tsf'])
        signal = bool(stats.mode([mama[-1] > fama[-1], slope[-1] > 0, ht[-1] > tsf[-1]])[0][0])
        self.signal_dict[symbol] = int(np.where(signal, 1, 2))
        
        
if __name__ == "__main__":
    symbols = token.my_assets
    timeframe = token.set_time_frame
    print(timeframe)
    pct_equity_trade = 0.1
    margin_csv = token.margin_file
    n_warmup = 1100
    log_freq = 100
    update_amount_freq = 90
    update_pos_freq = 70
    check_duplicate_freq = 80
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'my_parameters.json'), 'r') as f:
        parameters = json.load(f)
    
    trader = AggregateTrader(symbols, timeframe, parameters, pct_equity_trade, 
                             margin_csv, n_warmup, log_freq, update_amount_freq, 
                             update_pos_freq, check_duplicate_freq)
    # trader.verify_close_all()
    trader.begin_trading()
    target_time = dt.datetime.now() + dt.timedelta(hours=1)
    while dt.datetime.now() < target_time:
        time.sleep(60*60)
        target_time = dt.datetime.now() + dt.timedelta(hours=1)
    # time.sleep(60*60*5)
    # trader.verify_close_all()
    # trader.account_report()
    trader.close_connection()
