from fxcmtrader import *
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# dates will be automatically adjusted
# start --> Sunday 23:00:00 PM
# end --> Friday 21:00:00 PM

start_time = datetime.now(pytz.timezone('UTC'))
if start_time.weekday() == 6:
    start_time = start_time.replace(hour=23, minute=0, second=0)
elif start_time.weekday() == 5:
    start_time += timedelta(days=1)
    start_time = start_time.replace(hour=23, minute=0, second=0)

to_friday = timedelta(days=(4 - start_time.weekday()) % 7)
end_time = start_time + to_friday
end_time = end_time.replace(hour=21, minute=0, second=0)

margin_dict = pd.read_csv(os.path.join(dir_path, margin_file), index_col=['symbol']).to_dict()['margin']
log_path = os.path.join(dir_path, 'log_files')

params_path = os.path.join(dir_path, optim_file)
min_len = int(optim_file.split('_')[-2]) + 10
trader = ThreeIndicatorsTrader(
    params_path = params_path, 
    TOKEN=DEMO_TOKEN, 
    symbol_list=major_forex_pairs,
    margin_dict=margin_dict, 
    time_frame=set_time_frame, 
    min_len=min_len,
    start_time=start_time,
    end_time=end_time, 
    position_pct=0.001,
    log_path=log_path
    )

filename = trader.get_current_time().strftime("%Y_%m_%d") + "_PortfolioLog.csv"
if not os.path.exists(os.path.join(log_path, filename)):
    trader.log_trade("-"*50)
    trader.log_trade(f'***** | MultiMATrader| TimeFrame:{trader.time_frame} | *****')
    trader.log_trade(f"| UTC | start: {start_time} | end: {end_time} |")
    trader.log_trade(f"| BKK | start: {start_time.astimezone(pytz.timezone('Asia/Bangkok'))} | end: {end_time.astimezone(pytz.timezone('Asia/Bangkok'))} |")
    trader.log_trade("-"*50)

trader.start_trading()
