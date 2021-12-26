import fxcmpy
from fxcmtoken import *

print("Connecting...",end="")
con = fxcmpy.fxcmpy(access_token=DEMO_TOKEN, log_level='error', server='demo', log_file='log.txt')
print("Done")
print(f"Connection Status: {con.connection_status}")

print("Disconnecting...", end="")
con.close()
print("Done")
print(f"Connection Status: {con.connection_status}")
