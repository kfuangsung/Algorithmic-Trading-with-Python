import sys
import os
import time
import datetime as dt
import timeout_decorator
import fxcmpy 
import fxcmtoken as token


class CloseTrader:
    
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.con = None
        
        self.make_connection()
        self.close_all_positions()
        self.close_connection()
        
        
    def log_report(self, msg):
        content = f"{dt.datetime.utcnow()} | {msg}"
        file_name = dt.datetime.utcnow().strftime("%Y_%m_%d") + "_TradeReport.txt"
        file_name = os.path.join(self.dir_path, file_name)
        
        with open(file_name, 'a') as file:
            file.write(content + '\n')
        
        print(content)
        

    @timeout_decorator.timeout(60)
    def init_fxcmpy(self):
        self.con = fxcmpy.fxcmpy(access_token=token.DEMO_TOKEN, log_level='error', server='demo')
            
                
    def make_connection(self, n_count=0):
        counter = n_count
        if counter > 5:
            self.restart_python_script()
            
        self.log_report('-'*50)
        self.log_report("Connecting...")
        counter += 1
        try:
            self.init_fxcmpy()
            is_connected = self.con.is_connected()
            if is_connected:
                self.log_report('CONNECTED')
                self.log_report(f"connection status: {self.con.connection_status}")
                self.log_report('-'*50)
            else:
                self.log_report('FAILED to connect...try again in 60 seconds')
                time.sleep(60)
                self.make_connection(n_count=counter)
        except Exception as e:
            self.log_report('ERROR to connect...try again in 60 seconds')
            self.log_report(e)
            time.sleep(60)
            self.make_connection(n_count=counter)
            
            
    @timeout_decorator.timeout(60)
    def close_fxcmpy(self):
        self.con.close()
        
        
    def close_connection(self):
        
        self.log_report('-'*50)
        self.log_report("Disconnecting...")
        try:
            self.close_fxcmpy()
            if not self.con.is_connected():
                self.log_report('DISCONNECTED')
                self.log_report(f"connection status: {self.con.connection_status}")
                self.log_report('-'*50)
                return
            else:
                self.log_report('FAILED to close connect')
                return
        except Exception as e:
            self.log_report('ERROR | close connection')
            self.log_report(e)
            return
    
    
    def restart_python_script(self):
        self.log_report(f'*** restarting python script ***')
        os.execv(sys.executable, ['/home/azureuser/anaconda3/envs/fxcm/bin/python'] + sys.argv)
        
        
    def close_all_positions(self):
        """try closing all open positions"""

        counter = 0
        
        counter += 1
        self.log_report(f'Try closing all positions(attempt{counter})')
        self.con.close_all()
        time.sleep(60)
        
        counter += 1
        self.log_report(f'Try closing all positions(attempt{counter})')
        self.con.close_all()
        time.sleep(60)
        
        counter += 1
        self.log_report(f'Try closing all positions(attempt{counter})')
        self.con.close_all()
        time.sleep(60)
        
        
        
if __name__ == '__main__':
    trader = CloseTrader()
        
