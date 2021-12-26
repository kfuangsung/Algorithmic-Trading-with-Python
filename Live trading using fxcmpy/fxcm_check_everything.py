import fxcmpy
import time
from fxcmtoken import *


# connection
def connection():
    print('-'*50)
    print("Connecting...",end="")
    global con
    con = fxcmpy.fxcmpy(access_token=DEMO_TOKEN, log_level='error', server='demo', log_file='log.txt')
    print("Done")
    print(f"Connection Status: {con.connection_status}")
    print("-"*50)

# check get_accounts
def get_accounts():
    print("Checking 'get_accounts()'.....", end='')
    try:
        con.get_accounts()[['accountId', 'equity', 'usableMargin', 'mc']]
        return 'OK'
    except:
        return "ERROR"

# check get_candles
def get_candles():
    print("Checking 'get_candles()'.....", end='')
    try:
        con.get_candles('EUR/USD', period='m1', number=1).index[-1]
        return 'OK'
    except:
        return 'ERROR'

# check market order
def buy_order():
    print("Checking 'create_market_buy_order()'.....", end='')
    try:
        order = con.create_market_buy_order('EUR/USD', amount=1)
        if order.get_tradeId() in con.get_open_positions()['tradeId'].values.astype(int) and order.get_isBuy() is True:
            try:
                con.close_trade(order.get_tradeId(), amount=1)
            except:
                pass
            return 'OK'
        else:
            try:
                con.close_trade(order.get_tradeId(), amount=1)
            except:
                pass
            return 'ERROR'
    except:
        return 'ERROR'


def sell_order():
    print("Checking 'create_market_sell_order()'.....", end='')
    try:
        order2 = con.create_market_sell_order('EUR/USD', amount=1)
        if order2.get_tradeId() in con.get_open_positions()['tradeId'].values.astype(int) and order2.get_isBuy() is False:
            try:
                con.close_trade(order2.get_tradeId(), amount=1)
            except:
                pass
            return 'OK'
        else:
            try:
                con.close_trade(order2.get_tradeId(), amount=1)
            except:
                pass
            return 'ERROR'
    except:
        return 'ERROR'
        

# check close_all_for_symbol
def close_symbol():
    print("Checking 'close_all_for_symbol()'.....", end='')
    try:
        order = con.create_market_buy_order('EUR/USD', amount=1)
        order2 = con.create_market_sell_order('EUR/USD', amount=1)
        con.close_all_for_symbol('EUR/USD')
        time.sleep(5)
        if order.get_tradeId() in con.get_closed_positions()['tradeId'].values.astype(int) and order2.get_tradeId() in con.get_closed_positions()['tradeId'].values.astype(int):
            if not con.get_open_positions().empty:    
                try:
                    con.close_trade(order.get_tradeId(), amount=1)
                    con.close_trade(order2.get_tradeId(), amount=1)
                except:
                    pass
            else:
                pass
            return 'OK'
        else:
            if not con.get_open_positions().empty:    
                try:
                    con.close_trade(order.get_tradeId(), amount=1)
                    con.close_trade(order2.get_tradeId(), amount=1)
                except:
                    pass
            else:
                pass
            return 'ERROR'
    except:
        if not con.get_open_positions().empty:    
            try:
                con.close_trade(order.get_tradeId(), amount=1)
                con.close_trade(order2.get_tradeId(), amount=1)
            except:
                pass
        else:
            pass
        return 'ERROR'

# check close_all
def close_all():
    print("Checking 'close_all()'.....", end='')
    try:
        order = con.create_market_buy_order('EUR/USD', amount=1)
        order2 = con.create_market_sell_order('GBP/USD', amount=1)
        con.close_all()
        time.sleep(5)
        if order.get_tradeId() in con.get_closed_positions()['tradeId'].values.astype(int) and order2.get_tradeId() in con.get_closed_positions()['tradeId'].values.astype(int):
            if not con.get_open_positions().empty:    
                try:
                    con.close_trade(order.get_tradeId(), amount=1)
                    con.close_trade(order2.get_tradeId(), amount=1)
                except:
                    pass
            else:
                pass
            return 'OK'
        else:
            if not con.get_open_positions().empty:    
                try:
                    con.close_trade(order.get_tradeId(), amount=1)
                    con.close_trade(order2.get_tradeId(), amount=1)
                except:
                    pass
            else:
                pass
            return 'ERROR'
    except:
        if not con.get_open_positions().empty:    
            try:
                con.close_trade(order.get_tradeId(), amount=1)
                con.close_trade(order2.get_tradeId(), amount=1)
            except:
                pass
        else:
            pass
        return 'ERROR'

# check if there open_position left
def check_positions():
    attempts = 1
    while True:
        print('Checking positions.....', end='')
        if con.get_open_positions().empty:
            print('No open positions.')
            break
        else:
            print(f'Try closing all positions(attempt{attempts}).')
            con.close_all()
            attempts += 1

def disconnect():
    print("-"*50)
    print("Disconnecting...", end="")
    con.close()
    print("Done")
    print(f"Connection Status: {con.connection_status}")
    print('-'*50)

def retrying(func, n=5):
    i = 0
    while i <= n:
        res = func()
        if res == 'OK':
            print(res)
            break
        elif res == 'ERROR':
            print(res)
            i += 1
            if i <= n:
                print(f"Retrying|Attempt{i}.....", end='')
            

if __name__ == '__main__':
    connection()
    retrying(get_accounts)
    retrying(get_candles)
    retrying(buy_order)
    retrying(sell_order)
    retrying(close_symbol)
    retrying(close_all)
    check_positions()
    disconnect()