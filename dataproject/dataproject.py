import pandas as pd
from IPython.display import display
import requests
from dstapi import DstApi

def data_NAN1():
    """Fetches data on GDP growth in pct from DST using an API call"""
    #write in api request
    NAN1 = DstApi('NAN1') 
    #get the correct variables
    params = {'table': 'nan1',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'TRANSAKT', 'values': ['B1GQK']},
    {'code': 'PRISENHED', 'values': ['L_V']},
    {'code': 'Tid', 'values': ['*']}]}

    BNP = NAN1.get_data(params=params)
        
    #fertility = pd.DataFrame(data_FAM).reset_index
    return BNP

def data_HISB3():
    """Fetches data on childbirths from DST using their API call"""
    #api call
    HISB3 = DstApi('HISB3') 
    #select variables
    params = {'table': 'hisb3',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BEVÆGELSE', 'values': ['LFT','K']},
    {'code': 'Tid', 'values': ['*']}]}

    FOEDSLER = HISB3.get_data(params=params)
    return FOEDSLER

def data_HFUDD10():
   hfudd = DstApi('HFUDD10')
   params1={'table': 'hfudd10',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BOPOMR', 'values': ['000']},
    {'code': 'HERKOMST', 'values': ['TOT']},
    {'code': 'HFUDD', 'values': ['TOT', 'H10', 'H20', 'H30', 'H40', 'H50', 'H60', 'H70', 'H80', 'H90']},
    {'code': 'ALDER', 'values': ['20-24', '25-29', '30-34', '35-39', '40-44']},
    {'code': 'KØN', 'values': ['K']},
    {'code': 'Tid', 'values': ['>=2007<2008']}]}
   HFUDD10 = hfudd.get_data(params=params1)
   return HFUDD10

def data_HFUDD11():
    hfudd = DstApi('HFUDD11')
    params2={'table': 'hfudd11',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'BOPOMR', 'values': ['000']},
    {'code': 'HERKOMST', 'values': ['TOT']},
    {'code': 'HFUDD', 'values': ['TOT', 'H10', 'H20', 'H30', 'H40', 'H50', 'H60', 'H70', 'H80', 'H90']},
    {'code': 'ALDER', 'values': ['20-24', '25-29', '30-34', '35-39', '40-44']},
    {'code': 'KØN', 'values': ['K']},
    {'code': 'Tid', 'values': ['*']}]}
    HFUDD11 = hfudd.get_data(params=params2)
    return HFUDD11

def keep_regs(df, regs):
        """ Example function. Keep only the subset regs of regions in data.

        Args:
            df (pd.DataFrame): pandas dataframe 

        Returns:
            df (pd.DataFrame): pandas dataframe

        """ 
        
        for r in regs:
            I = df.reg.str.contains(r)
            df = df.loc[I == False] # keep everything else
        
        return df 