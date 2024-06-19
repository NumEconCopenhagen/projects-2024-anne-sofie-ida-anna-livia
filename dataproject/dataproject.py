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

def clean_NAN1(): 
    """ Cleaning data from NAN1 """
    # Loading data using the function data_NAN1
    BNP=data_NAN1()
    # Drop the irrelevant descriptive columns "TRANSAKT" and "PRISENHED".
    drop_these = ['TRANSAKT','PRISENHED']
    BNP.drop(drop_these, axis=1, inplace=True)

    # Rename columns to more suitable variable names.
    BNP.rename(columns = {'INDHOLD':'growth', 'TID':'year'}, inplace=True)

    # Drop missing entries.
    I = BNP.loc[BNP.growth == '..'] 
    BNP.drop(I.index, inplace=True)

    # Sort dataset by year.
    BNP.sort_values(by = ['year'], inplace=True)

    # Convert all entries to float.
    BNP['growth'] = BNP['growth'].astype('float')
    BNP_clean=BNP
    return BNP_clean

def clean_HISB3():
    """ Cleaning data from HISB3 """
    HIS=data_HISB3()
    # Sort dataset on year and variable.
    HIS.sort_values(by = ['TID','BEVÆGELSE'], inplace=True)

    # Rename time variable. We rename the column with number of births and women after sorting. 
    HIS.rename(columns = {'TID':'year'}, inplace=True)

    # Create subsets of odd and even rows to separate births and amount of women.
    births_sub = HIS.iloc[::2].copy()
    women_sub = HIS.iloc[1::2].copy()

    # Rename the column "INDHOLD" to number of births.
    births_sub.rename(columns={'INDHOLD':'births'}, inplace=True)

    # Rename the column "INDHOLD" to number of women.
    women_sub.rename(columns = {'INDHOLD': 'women'}, inplace=True)

    # Merge the subsets on year.
    fertility = pd.merge(births_sub, women_sub, how='inner', on=['year'])

    # Drop the irrelevant descriptive columns.
    drop = ['BEVÆGELSE_x', 'BEVÆGELSE_y']
    fertility.drop(drop,axis=1,inplace=True)

    # Drop missing entries.
    fertility = fertility.dropna()
    I = fertility.loc[fertility.births == '..'] 
    fertility.drop(I.index, inplace=True)
    I_2 = fertility.loc[fertility.women == '..'] 
    fertility.drop(I_2.index, inplace=True)

    # Convert entry type to float.
    fertility['births'] = fertility['births'].astype('float')
    fertility['women'] = fertility['women'].astype('float')

    fertility_clean=fertility
    return fertility_clean

def clean_HFUDD11():
    """ Cleaning data from HFUDD11 """
    hfudd11=data_HFUDD11()
    # Drop irrelevant columns
    drop_these = ['BOPOMR','HERKOMST', 'KØN']
    hfudd11.drop(drop_these, axis=1, inplace=True)

    # Rename colums
    hfudd11.rename(columns={'TID':'Year' ,'INDHOLD':'Number of women','ALDER':'Age'}, inplace=True)  

    hfudd11_clean=hfudd11
    return hfudd11_clean