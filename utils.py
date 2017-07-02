'''
Created on Jun 29, 2017

@author: Gajo
'''

def convert_to_float(value, default = None):
    try:
        return float(value)
    except:
        return default

def convert_to_int(value, default = None):
    try:
        return int(value)
    except:
        return default