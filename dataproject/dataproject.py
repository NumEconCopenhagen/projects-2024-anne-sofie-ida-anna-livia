import pandas as pd
from IPython.display import display
import requests
from dstapi import DstApi

def data_NAN1():
    """Fetches data on GDP growth in pct from DST using an API call"""
    # We import the data using an API.
    NAN1 = DstApi('NAN1') 
    # We specify which variables we are interested in with the function "params".
    params = {'table': 'nan1',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'TRANSAKT', 'values': ['B1GQK']},
    {'code': 'PRISENHED', 'values': ['L_V']},
    {'code': 'Tid', 'values': ['*']}]}

    BNP = NAN1.get_data(params=params)
    return BNP

def data_HISB3():
    """Fetches data on childbirths from DST using their API call"""
    # We import the data using an API.
    HISB3 = DstApi('HISB3') 
    # We specify which variables we are interested in with the function "params".
    params = {'table': 'hisb3',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BEVÆGELSE', 'values': ['LFT','K']},
    {'code': 'Tid', 'values': ['*']}]}

    FOEDSLER = HISB3.get_data(params=params)
    return FOEDSLER

def data_HFUDD11():
    """Fetches data on women's educational atainment from DST using their API call"""
    # We import the data using an API.
    HFUDD11 = DstApi('HFUDD11')
    # We specify which variables we are interested in with the function "params".
    params={'table': 'hfudd11',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BOPOMR', 'values': ['000']},
    {'code': 'HERKOMST', 'values': ['TOT']}, {'code': 'HFUDD', 'values': ['TOT', 'H10', 'H20', 'H30', 'H40', 'H50', 'H60', 'H70', 'H80']},
    {'code': 'ALDER', 'values': ['20-24', '25-29', '30-34', '35-39', '40-44']},
    {'code': 'KØN', 'values': ['K']},
    {'code': 'Tid', 'values': ['*']}]}
    HFUDD = HFUDD11.get_data(params=params)
    return HFUDD