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