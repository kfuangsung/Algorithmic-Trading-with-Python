import os 
import glob 
import datetime as dt 
import pathlib


def check_last_log_time():
    
    # get the lastest tradereport
    curr_path = pathlib.Path(__file__).parent.resolve()
    txt_reports = glob.glob(os.path.join(curr_path,'*TradeReport.txt'))
    lastest_report = sorted(txt_reports, reverse=True)[0]

    # read the last line
    with open(lastest_report, 'rb') as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode()
    
    # get datetime from the last line
    last_time_str = last_line.split(' | ')[0]
    last_time_dt = dt.datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S.%f')
    
    # compare with current time
    curr_time_utc = dt.datetime.utcnow()
    if (curr_time_utc - last_time_dt) > dt.timedelta(hours=1):
        # error in log time
        return 1
    else:
        # still okay
        return 0
    
    
if __name__ == '__main__':
    var = check_last_log_time()
    print(var)