# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:58:54 2018

@author: 2014_Joon_IBS
"""

import numpy as np
from io import StringIO
import requests

image_count = 1
image_size = 48 * 48
image_data = np.random.random((image_count, image_size))#, dtype=np.float32)
s = StringIO()

np.savetxt(s, image_data, delimiter=',')#, newline='\n')

csv_str = s.getvalue()

model_url = 'http://143.248.92.116:8002/model/serving_default/input_image'
rsp = requests.post(model_url, data=csv_str,
                        headers={'Content-Type': 'text/csv'})
print(rsp.status_code, rsp.reason)
print(rsp.headers)
print(rsp.text)
result = rsp.text
type(result)
#result['floatVal']
