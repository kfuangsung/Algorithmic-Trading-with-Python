#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fxcmpy
import socketio
import os
from tqdm import tqdm
from fxcmtoken import *


# In[3]:


print("Connecting...",end="")
con = fxcmpy.fxcmpy(access_token=DEMO_TOKEN, log_level='error', server='demo', log_file='log.txt')
print("Done")
print(f"Connection Status: {con.connection_status}")


# In[ ]:


print("Disconnecting...", end="")
con.close()
print("Done")
print(f"Connection Status: {con.connection_status}")

