from fxcmtrader import *
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

start_time = datetime.now(pytz.timezone('UTC'))
# when to stop trade 
full_week_delta_time = timedelta(days=4, hours=23, minutes=21)
hour_delta_time = timedelta(hours=3)
minute_delta_time = timedelta(minutes=30)
# end_time = start_time + minute_delta_time
end_time = datetime(year=2021, month=12, day=24, hour=18, minute=20, tzinfo=pytz.timezone('UTC'))

margin_dict = pd.read_csv(os.path.join(dir_path, 'fxcm_margin_2021_12_03.csv'), index_col=['symbol']).to_dict()['margin']
log_path = os.path.join(dir_path, 'log_files')

params_path = os.path.join(dir_path, "OPTIMIZE_params_m15.json")
min_len = 120
time_frame = 'm15'
trader = MultiMATrader(params_path = params_path, 
                       TOKEN=DEMO_TOKEN, 
                       symbol_list=major_forex_pairs,
                       margin_dict=margin_dict, 
                       time_frame=time_frame, 
                       min_len=min_len,
                       end_time=end_time, 
                       position_pct=0.001,
                       log_path=log_path)

trader.connect()
trader.check_positions()
trader.log_portfolio()
trader.disconnect()