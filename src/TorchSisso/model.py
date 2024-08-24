
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:22:50 2023

@author: muthyala.7
"""

from .import FeatureSpaceConstruction as fc
from .import DimensionalFeatureSpaceConstruction as dfc
from .Regressor import Regressor
from .Regressor_dimension import Regressor


import sys
import time
import pdb
import numpy as np 
import pandas as pd 
import time
from sympy import symbols
from sklearn.model_selection import train_test_split





class sisso_model:

  def __init__(self,df,operators=None,multi_task = None,no_of_operators=None,dimension=None,sis_features=20,device='cpu',relational_units = None,initial_screening = None,dimensionality=None,output_dim = None,regressor_screening = None):

    self.operators = operators
    self.df=df
    self.no_of_operators = no_of_operators
    self.device = device
    if dimension == None: self.dimension = 3#dimension
    else: self.dimension = dimension
    if sis_features == None: self.sis_features = 10
    else: self.sis_features = sis_features
    self.relational_units = relational_units
    self.initial_screening = initial_screening
    self.dimensionality = dimensionality
    self.output_dim = output_dim
    self.regressor_screening = regressor_screening
    
    self.multi_task = multi_task
    if multi_task!=None:
        self.multi_task_target = multi_task[0]
        self.multi_task_features = multi_task[1]
    

      
  def fit(self):
      
    if self.dimensionality == None:
        
        if self.operators==None: sys.exit('Please provide the operators set for the non dimensional Regression!!')
        
        if self.multi_task!=None:
            
            print('Performing MultiTask Symbolic regression!!..')
            
            equations =[]
            
            for i in range(len(self.multi_task_target)):
                
                #Get the target variable and feature variabls 
                print('Performing symbolic regression of',i+1,'Target variables....')
                
                list1 =[]
                
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                
                df1 = self.df.iloc[:,list1]
                
                
                x,y,names = fc.feature_space_construction(self.operators,df1,self.no_of_operators,self.device,self.initial_screening).feature_space()
                    
                from Regressor import Regressor
                  
                rmse, equation,r2 =  Regressor(x,y,names,self.dimension,self.sis_features,self.device).regressor_fit()
                
                equations.append(equation)
                    
                if i+1 == len(self.multi_task_target):
                    print('Equations found::',equations)
                    return rmse, equation, r2,equations
                else: continue
                
        
        else:
            
            x,y,names = fc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device,self.initial_screening).feature_space()
            
            from Regressor import Regressor
            
            rmse, equation,r2 =  Regressor(x,y,names,self.dimension,self.sis_features,self.device).regressor_fit()
        
            return rmse, equation, r2
  
    else: 
        
        if self.multi_task!=None:
            
            print('************************************************ Performing MultiTask Symbolic regression!!..************************************************ \n')
            
            equations =[]
            
            for i in range(len(self.multi_task_target)):
                
                #Get the target variable and feature variabls 
                print('************************************************ Performing symbolic regression of',i+1,'Target variables....************************************************ \n')
                
                list1 =[]
                
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                
                df1 = self.df.iloc[:,list1]
                
                
                    
                x,y,names,dim = dfc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality).feature_expansion()
                    
                    #print(names)
                from Regressor_dimension import Regressor
                    
                rmse,equation,r2 = Regressor(x,y,names,dim,self.dimension,self.sis_features,self.device,self.output_dim,self.regressor_screening).regressor_fit()
                    
                equations.append(equation)
                    
                if i+1 == len(self.multi_task_target):
                        
                    print('Equations found::',equations)
                        
                    return rmse, equation, r2,equations
                    
                else: continue
                
        
        
        else:
            
            x,y,names,dim = dfc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality).feature_expansion()
            
            #print(names)
            from Regressor_dimension import Regressor
            rmse,equation,r2 = Regressor(x,y,names,dim,self.dimension,self.sis_features,self.device,self.output_dim,self.regressor_screening).regressor_fit()
            
            return rmse,equation,r2


