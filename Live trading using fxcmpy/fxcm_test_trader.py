from fxcmtrader import *
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

start_time = datetime.now(pytz.timezone('UTC'))
# when to stop trade 
full_week_delta_time = timedelta(days=4, hours=23, minutes=21)
hour_delta_time = timedelta(hours=3)
minute_delta_time = timedelta(minutes=30)
# end_time = start_time + minute_delta_time
end_time = datetime(year=2021, month=12, day=31, hour=19, minute=32, tzinfo=pytz.timezone('UTC'))

margin_dict = pd.read_csv(os.path.join(dir_path, 'fxcm_margin_2021_12_03.csv'), index_col=['symbol']).to_dict()['margin']
log_path = os.path.join(dir_path, 'log_files')

# period = (2, 5)
# min_len = period[1] + 10
# time_frame = 'm1'
# trader = SMATrader(short_period=period[0], 
#                    long_period=period[1],
#                    TOKEN=DEMO_TOKEN, 
#                     symbol_list=major_forex_pairs,
#                     margin_dict=margin_dict, 
#                     time_frame=time_frame, 
#                     min_len=min_len,
#                     end_time=end_time, 
#                     position_pct=0.001,
#                     log_path=log_path)
# trader.log_trade(f'***** SMATrader|TimeFrame:{trader.time_frame} | period:{period} | *****')

# min_len = 1000
# time_frame = 'm5'
# trader = RandomTrader(TOKEN=DEMO_TOKEN, 
#                       symbol_list=major_forex_pairs,
#                       margin_dict=margin_dict, 
#                       time_frame=time_frame, 
#                       min_len=min_len,
#                       end_time=end_time, 
#                       position_pct=0.001,
#                       log_path=log_path)
# trader.log_trade(f'***** RandomTrader| TimeFrame:{trader.time_frame} | *****')

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
trader.log_trade("-"*50)
trader.log_trade(f'***** | MultiMATrader| TimeFrame:{trader.time_frame} | *****')

trader.log_trade(f"| UTC | start: {start_time} | end: {end_time} |")
trader.log_trade(f"| BKK | start: {start_time.astimezone(pytz.timezone('Asia/Bangkok'))} | end: {end_time.astimezone(pytz.timezone('Asia/Bangkok'))} |")
trader.log_trade("-"*50)
trader.start_trading()
