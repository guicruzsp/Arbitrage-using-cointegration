# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:06:50 2019

@author: gui_c
"""

import pandas as pd
import quandl
import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as ts
import statsmodels.stats.stattools as tst
import operator
import xlsxwriter
import math



#Exemplo
itub = pdr.DataReader('ITUB4.SA', 'yahoo' , start, end)


acoes =   [('OIBR3.SA - OIBR4.SA', 7.213162346146367)]






w = range(0,len (acoes))
d = {i:[] for i in w}
#result = {i:[] for i in w}
result = {}
trades = 0

carteira = pd.DataFrame()


#Meu modelo
for i in range(0, len(acoes)):
    a, b = acoes[i][0].split(' - ')
    ac1 = pdr.DataReader(b, 'yahoo' ,
                          start=datetime.datetime(2013, 7, 1), 
                          end=datetime.datetime(2017, 1, 1))
    ac2 = pdr.DataReader(a, 'yahoo' ,
                          start=datetime.datetime(2013, 7, 1), 
                          end=datetime.datetime(2017, 1, 1))

    ac1 = ac1['Close']
    ac2 = ac2['Close']
 
        
    window = 250
    
    comp = pd.DataFrame()
    comp['ac1'] = ac1
    comp['ac2'] = ac2
    comp['beta'] = 0
    comp['med'] = 0
    comp['dp'] = 0
    comp['resid'] = 0
    
    for j in range(window, len(comp)):
            comp['beta'].iloc[j] = sm.OLS(comp['ac2'].iloc[j-window:j], 
                  comp['ac1'].iloc[j-window:j]).fit().params['ac1']
    del j
    
    
    comp['resid'] = np.where(comp['beta'] != 0, (comp['ac2'] - comp['beta']*comp['ac1']), 0)
    comp['resid'].iloc[:window] = sm.OLS(comp['ac2'].iloc[:window], 
        comp['ac1'].iloc[:window]).fit().resid
    
    for j in range(window, len(comp)):
            comp['med'].iloc[j] = comp['resid'].iloc[j-window:j].mean()
    del j
    
    for j in range(window, len(comp)):
            comp['dp'].iloc[j] = np.std(comp['resid'].iloc[j-window:j])
    del j
    
    comp['normal'] = 0
    comp['normal'] = (comp['resid'] - comp['med'])/comp['dp']

   
    signals = pd.DataFrame(index = ac2.index)
    signals['signal'] = 0.0
    signals['exit'] = 0.0
    signals['entry'] = 0.0
    signals['up'] = 0.0
    #signals['hl'] = 0 
    #signals['resid'] = ac2 - (beta*ac1) - const
    signals['resid'] = comp['normal']
    signals['med'] = comp['med']
    signals['dp'] = comp['dp']
        
    signals['entry'] = np.where((abs(signals['resid'] - signals['med']) > 2.5*signals['dp']), 0, (
            np.where(abs(signals['resid'] - signals['med']) > 1.5*signals['dp'], 1.0, 0)))
    #for o in range(1,len(ac1)):
    #    signals['entry'] = np.where((abs(signals['resid'] - signals['med']) < 2*signals['dp']), 
    #        np.where(((abs(signals['resid'].shift()) - signals['med']) > 2*signals['dp']), 1, 0), 0)
            
        
        
    signals['up'] = np.where((signals['resid'] > signals['med']), 1.0, 0.0)
    #signals['up'] = np.where((abs(signals['resid'])) > abs(signals['med'] - 0.5*signals['dp']), 1.0, 0.0)
    
    signals['exit'] = np.where(signals['up'].diff() != 0, signals['up'].diff(), 
               np.where(signals['resid'] > signals['med'] + 4*signals['dp'], 1, np.where(
                       signals['resid'] < signals['med'] - 4*signals['dp'], 1, 0)))
    
    for j in range(1, len(ac1)):
             signals['signal'] = np.where(signals['entry']==1.0 , np.where(signals['exit'] == 0, 1.0, 0.0),
    #         signals['signal'] = np.where(signals['entry']==1.0 , np.where(signals['exit'] == 0, np.where(signals['hl']<halflife, 1.0, 0.0), 0.0),
                   (np.where(signals['exit']!=0, 
                             0.0, (np.where(signals['signal'].shift()==0.0, 0.0, 1.0)))))
    #         signals['hl'] = np.where(signals['signal'] != 0, signals['hl'].shift()+1, 0)  
    
    
    
    #for k in range(1, len(ac1)):
    #    signals['hl'] = np.where(signals['signal'] != 0, signals['hl'].shift()+1, 0)      
             
    signals['positions'] = signals['signal'].diff()
        
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['resid'] = signals['resid']
    positions['med'] = signals['med']
    positions['beta'] = comp['beta']
    
    positions['ac2'] = 0
    positions['ac1'] = 0
    
    #modelo beta neutro
#    for r in range(1,len(ac1)):
#        positions['ac2'] = np.where(signals['positions'] == 1, np.where(positions['resid'] > positions['med'], -10000*signals['signal'], 10000*signals['signal']), np.where(
#            signals['positions'] == -1, 0, np.where(
#                    positions['ac2'].shift() != 0, positions['ac2'].shift(), 0)))
#        positions['ac1'] = np.where(signals['positions'] == 1, np.where(positions['resid'] > positions['med'], 10000*positions['beta']*signals['signal'], -10000*positions['beta']*signals['signal']), np.where(
#            signals['positions'] == -1, 0, np.where(
#                    positions['ac1'].shift() != 0, positions['ac1'].shift(), 0)))
#    positions['ac2'] = positions['ac2'].fillna(0.0)
#    positions['ac1'] = positions['ac1'].fillna(0.0)
    
    
    #modelo cash neutro
    for r in range(1,len(ac1)):
        positions['ac2'] = np.where(signals['positions'] == 1, np.where(positions['resid'] > positions['med'], -50000/(ac2[:len(positions)]), 50000/(ac2[:len(positions)])), np.where(
            signals['positions'] == -1, 0, np.where(
                    positions['ac2'].shift() != 0, positions['ac2'].shift(), 0)))
        positions['ac1'] = np.where(signals['positions'] == 1, np.where(positions['resid'] > positions['med'], 50000/(ac1[:len(positions)]), -50000/(ac1[:len(positions)])), np.where(
            signals['positions'] == -1, 0, np.where(
                    positions['ac1'].shift() != 0, positions['ac1'].shift(), 0)))
    positions['ac2'] = positions['ac2'].fillna(0.0)
    positions['ac1'] = positions['ac1'].fillna(0.0) 
    
    
        
    portfolio = positions.multiply(ac2, axis=0)
    portfolio['resid'] = signals['resid']
    portfolio['ac2'] = positions['ac2'].multiply(ac2, axis=0)
    portfolio['ac1'] = positions['ac1'].multiply(ac1, axis=0)
    pos_diff = positions.diff()
    pos_diff['ac2'] = pos_diff['ac2'].fillna(positions['ac2'])
    pos_diff['ac1'] = pos_diff['ac1'].fillna(positions['ac1'])
    portfolio['cash'] = ((-pos_diff['ac2']*ac2).cumsum() +
         (-pos_diff['ac1']*ac1).cumsum())
    portfolio['total'] = portfolio['cash'] + portfolio['ac2'] + portfolio['ac1']
    portfolio['ac1 +ac1'] = abs(portfolio['ac2']) + abs(portfolio['ac1'])
    
    for o in range (0, len(pos_diff[pos_diff['ac1']!=0])):
        trades += 2
    
    d[i]=portfolio
    
    
    portfolio['rec'] = portfolio['total']
    for k in range(1, len(ac1)):
        portfolio['rec'] = np.where(portfolio['ac1 +ac1'] != 0, np.where(
         portfolio['ac1 +ac1'].shift() == 0, portfolio['total'], np.where(
         portfolio['ac1 +ac1'].shift() != 0, portfolio['rec'].shift(), 0)), np.where(
         portfolio['ac1 +ac1'] == 0, np.where(portfolio['ac1 +ac1'].shift() != 0, portfolio['rec'].shift(), 0), 0))

    portfolio['valor op'] = portfolio['total']
    for l in range(1, len(ac1)):
        portfolio['valor op'] = np.where(portfolio['ac1 +ac1'] != 0, np.where(
         portfolio['ac1 +ac1'].shift() == 0, portfolio['ac1 +ac1'], np.where(
         portfolio['ac1 +ac1'].shift() != 0, portfolio['valor op'].shift(), 0)), np.where(
         portfolio['ac1 +ac1'] == 0, np.where(portfolio['ac1 +ac1'].shift() != 0, portfolio['valor op'].shift(), 0), 0))
    
    portfolio['vlp'] = portfolio['valor op']     
    for o in range(1, len(ac1)):
        portfolio['vlp'] = np.where(portfolio['valor op']!=0,
                 np.where(portfolio['valor op'].shift() == 0,
                          portfolio['valor op'],
                          (portfolio['total']-portfolio['total'].shift())+portfolio['vlp'].shift()),
                          0)
        

    portfolio['profit'] = portfolio['total']
    portfolio['profit'] = np.where(portfolio['ac1 +ac1'] == 0, 
         np.where(portfolio['ac1 +ac1'].shift() != 0, 
                  ((portfolio['total'] - portfolio['rec'])/portfolio['valor op']) + 1, 0), 0)
    
    portfolio['rent'] = portfolio['total']
    for p in range(1, len(ac1)):
        portfolio['rent'] = np.where(portfolio['vlp']!=0,
             np.where(portfolio['vlp'].shift()==0, 0,pd.DataFrame.pct_change(portfolio['vlp'])),
             0)
    
    pd.DataFrame.pct_change(portfolio['vlp'])
    portfolio['rent'] = portfolio['rent'].fillna(0.0)

    portfolio['profit'] = portfolio['profit'].fillna(0.0)
    resultado = list( filter(lambda a: a!=0, portfolio['profit']))
    result[a[5:] + ' - ' + b[5:]] = resultado
    
    
    portfolio['exp'] = 0
    portfolio['exp']
    
   
    initial_capital = float(100000.0)
    portfolio['carteira'] = portfolio['total']
    portfolio['carteira'] = 0
    portfolio['carteira'][0] = float(100000.0)
    for r in range(1, len(ac1)):
        portfolio['carteira'] = np.where(pd.isnull(portfolio['carteira'].shift()),
             100000,
#             portfolio['carteira'].shift()*(portfolio['rent']+1))
             np.where(portfolio['profit']==0, portfolio['carteira'].shift()*(portfolio['rent']+1),
                      #portfolio['carteira'].shift()*(portfolio['rent']+1) - 0.005*portfolio['carteira'].shift()))
                      portfolio['carteira'].shift()*(portfolio['rent']+1)))

    carteira[a+ ' - ' +b] = portfolio['carteira']
    if carteira[a+ ' - ' +b][0] == carteira[a+ ' - ' +b][-1]:
        carteira[a+ ' - ' +b] = 0
    carteira[a+ ' - ' +b] = carteira[a+ ' - ' +b].fillna(method='ffill')
    carteira[a+ ' - ' +b + ' ret'] = portfolio['rent']
    carteira[a+ ' - ' +b + ' ret'] = carteira[a+ ' - ' +b + ' ret'].fillna(0)
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel = 'Resid')
    signals[['resid']].plot(ax=ax1, lw=2)
    #buy signal
    ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.resid[signals.positions == 1.0],
         '^', markersize=10, color='m')
    #sell signal
    ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.resid[signals.positions == -1.0],
         '^', markersize=10, color='k')
    plt.show()
    
    


    
    

carteira['total'] = pd.DataFrame.sum(carteira, axis=1) 
carteira['retorno'] = carteira['total'].pct_change()

plt.plot(carteira['total']) # plotting total returns
plt.show()
    
z = pd.DataFrame.from_dict(result, 'index')
z = z.transpose()


