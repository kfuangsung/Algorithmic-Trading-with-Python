from backtester import *
from tqdm import tqdm
import concurrent
import os 

def get_ma_return(path, periods, kind, freq):
    data = pd.read_csv(path, parse_dates=['date'], index_col=['date'])
    ma_backtest = MovingAverageBacktester(data=data, freq=freq, periods=(periods[0],periods[1]), kind=kind)
    ma_backtest.backtest(progress_bar=False, is_notebook=False, leave=False)
    return (periods, ma_backtest.returns['Portfolio']['Annualized Return'])

def optim_ema(periods):
    return get_ma_return(path, periods, 'SMA', freq)

periods_list = []
for i in range(5, 505, 5):
    for j in range(5, 505, 5):
        if i < j:
            periods_list.append((i, j))

main_path = 'forex_prices'
freq = 'm5'

for file in tqdm(os.listdir(os.path.join(main_path, freq))):
    path = os.path.join(main_path, freq, file)
    print(path)
    with concurrent.futures.ProcessPoolExecutor() as excecutor:
        results = list(tqdm(excecutor.map(optim_ema, periods_list), total=len(periods_list), leave=False))
    results_dict = {}
    for period, ret in results:
        results_dict[period] = ret
    sorted_results = dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))
    with open('sma_optim.txt', 'a') as file:
        file.write(f'{path}' + '\n')
        count = 1
        for key, value in sorted_results.items():
            if count > 5: break
            file.write(f"{count}|{key}|{value*100:.4f} %" + "\n")
            count += 1
        file.write("-"*50 + "\n")
