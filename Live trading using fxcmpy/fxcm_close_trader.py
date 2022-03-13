from fxcmtrader import *
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# use for closing all positions (if not yet all closed)
# just in case when something was wrong with main script

start_time = datetime.now(pytz.timezone('UTC'))
minute_delta_time = timedelta(minutes=30)
end_time = start_time + minute_delta_time

margin_dict = pd.read_csv(os.path.join(dir_path, margin_file), index_col=['symbol']).to_dict()['margin']
log_path = os.path.join(dir_path, 'log_files')
min_data_length = int(optim_file.split('_')[-2])

params_path = os.path.join(dir_path, optim_file)
min_len = min_data_length
time_frame = set_time_frame
trader = MultiMATrader(params_path = params_path, 
                       TOKEN=DEMO_TOKEN, 
                       symbol_list=major_forex_pairs,
                       margin_dict=margin_dict, 
                       time_frame=time_frame, 
                       min_len=min_len,
                       start_time=start_time,
                       end_time=end_time, 
                       position_pct=0.001,
                       log_path=log_path)

trader.connect()
trader.check_positions()
trader.log_portfolio()
trader.disconnect()

