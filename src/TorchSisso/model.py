
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





class SissoModel:

  def __init__(self,data,operators=None,multi_task = None,n_expansion=None,n_term=None,k=20,device='cpu',use_gpu = False,
               relational_units = None,initial_screening = None,dimensionality=None,output_dim = None,
               regressor_screening = None,custom_unary_functions = None ,custom_binary_functions=None):

    self.operators = operators
    
    if self.operators == None: sys.exit('!! Please provide the valid operator set!!')
    
    self.df=data

    self.no_of_operators = n_expansion
    
    if self.no_of_operators == None: 
        
        print('!! Warning:: Number of feature expansions is not provided, considering default configuration!! \n')
        
        self.no_of_operators = 3
        
    self.device = device
    
    if n_term == None: 
        
        print('Warning:: Number of terms in equation is not provided, considering default configuration..!! \n')
        
        self.dimension = 3
        
    else: self.dimension = n_term
        
    self.sis_features = k
    
    self.relational_units = relational_units
    
    self.initial_screening = initial_screening
    
    self.dimensionality = dimensionality
    
    self.output_dim = output_dim
    
    self.regressor_screening = regressor_screening
    
    self.use_gpu = use_gpu
    
    self.custom_unary_functions = custom_unary_functions
    
    self.custom_binary_functions = custom_binary_functions
    
    self.multi_task = multi_task
    
    if multi_task!=None:
        
        self.multi_task_target = multi_task[0]
        
        self.multi_task_features = multi_task[1]
    

      
  def fit(self):
      
    if self.dimensionality == None:
        
        if self.operators==None: sys.exit('Please provide the operators set for the non dimensional Regression!!')
        
        if self.multi_task!=None:
            
            print('Performing MultiTask Symbolic regression!!..')
            
            equations,rmses,r2s =[],[],[]
            
            for i in range(len(self.multi_task_target)):
                
                #Get the target variable and feature variabls 
                print('Performing symbolic regression of',i+1,'Target variables....')
                
                list1 =[]
                
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                
                df1 = self.df.iloc[:,list1]
                
                
                x,y,names = fc.feature_space_construction(self.operators,df1,self.no_of_operators,self.device,self.initial_screening,self.custom_unary_functions,self.custom_binary_functions).feature_space()
                    
                from .Regressor import Regressor
                  
                rmse, equation,r2,final_eq =  Regressor(x,y,names,self.dimension,self.sis_features,self.device,self.use_gpu).regressor_fit()
                
                equations.append(final_eq)
                
                rmses.append(rmse)
                
                r2s.append(r2)
                    
                if i+1 == len(self.multi_task_target):
                    
                    print('Equations found::',equations)
                    
                    return rmses, equations, r2s
                
                else: continue
                
        
        else:
            
            x,y,names = fc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device,self.initial_screening,self.custom_unary_functions,self.custom_binary_functions).feature_space()
            
            from .Regressor import Regressor
            
            rmse, equation,r2,final_eq =  Regressor(x,y,names,self.dimension,self.sis_features,self.device,self.use_gpu).regressor_fit()
        
            return rmse, equation, r2,final_eq
  
    else: 
        
        if self.multi_task!=None:
            
            print('************************************************ Performing MultiTask Symbolic regression!!..************************************************ \n')
            
            equations,rmses,r2s =[],[],[]
            
            for i in range(len(self.multi_task_target)):
                
                #Get the target variable and feature variabls 
                print('************************************************ Performing symbolic regression of',i+1,'Target variables....************************************************ \n')
                
                list1 =[]
                
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                
                df1 = self.df.iloc[:,list1]
                
                
                    
                x,y,names,dim = dfc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality).feature_expansion()
                    
                    #print(names)
                from .Regressor_dimension import Regressor
                    
                rmse,equation,r2 = Regressor(x,y,names,dim,self.dimension,self.sis_features,self.device,self.output_dim,self.regressor_screening,self.use_gpu).regressor_fit()
                    
                equations.append(equation)
                
                rmses.append(rmse)
                
                r2s.append(r2)
                    
                    
                if i+1 == len(self.multi_task_target):
                        
                    print('Equations found::',equations)
                        
                    return rmses, equations, r2s
                    
                else: continue
                
        
        
        else:
            
            x,y,names,dim = dfc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality).feature_expansion()
            
            #print(names)
            from .Regressor_dimension import Regressor
            
            rmse,equation,r2 = Regressor(x,y,names,dim,self.dimension,self.sis_features,self.device,self.output_dim,self.regressor_screening,self.use_gpu).regressor_fit()
            
            return rmse,equation,r2


