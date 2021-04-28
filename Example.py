# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:57:43 2019

@author: Jimena
"""

from Version0.KPMV import KPMV_Tool

def Data():
    import numpy as np
    x= np.random.uniform(low=-5, high=5, size=(100,2))
    y= [ x[pos,0]**2 + x[pos,1]**2 for pos in range(len(x))]
    y=np.asarray(y).reshape(-1,1)

    return x,y

ops= {'add', 'sub', 'mul', 'div'}
y_max=[2]
x,y = Data()
BestPop, BestQual, BestBeta, EcEst, RMSE_Best, RMSE_Best_Norm, logbook, mod_sim= KPMV_Tool(x,y, ngen=2000,size=8, ops=ops, Deph=(0,3),Deph_max=10, y_max=y_max)
