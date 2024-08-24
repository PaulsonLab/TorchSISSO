#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:17:36 2024

@author: muthyala.7
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:22:25 2024

@author: muthyala.7
"""

import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
from itertools import combinations
import sys 
from scipy.stats import spearmanr
import pdb
from sympy import symbols,Pow,powdenest



import math


class feature_space_construction:
    
    '''
    Define the function to get the variables
    '''
    
    def __init__(self,df,operators=None,relational_units = None,initial_screening = None,no_of_operators=None,device='cpu',dimensionality=None,metrics=[0.06,0.995],output_dim=None):
    
      print(f'************************************* Starting Feature Space Construction in {device} ****************************************************************')
      print('\n')
      '''
      ###########################################################################################
    
      no_of_operators - defines the presence of operators (binary or unary) in the expanded features space
    
      For example: if no_of_operators = 2 then the space will be limited to formation of features with 3 operators (x1+x2)/x3 or exp(x1+x2)
    
      ###########################################################################################
      '''
      self.no_of_operators = no_of_operators
    
      self.df = df
      
      '''
      ###########################################################################################
    
      operators [list type]: Defines the mathematical operators needs to be used in the feature expansion
    
      Please look at the README.md for type of mathematical operators allowed
    
      ###########################################################################################
      '''
      if operators !=None: self.operators = operators
      
      else:
          
          print('************************************************ WARNING:: INPUT OPERATOR SET IS NOT PROVIDED, WILL BE USING THE DEFAULT SET OF OPERATORS WHICH MIGHT TAKE MEMORY DEPENDIN THE FEATURE SPACE PROVIDED.. ********************************************** \n')
          self.operators = ['+','-','*','/','exp','sin','cos','tanh','pow(1/2)','pow(1/3)','log','ln','^-1','pow(2)','pow(3)','exp(-1)','/2','+1','-1','/2pi','*2pi']
          
      self.device = torch.device(device)
      
      print(f'************************************ Provided Operator Set to perform Symbolic Regression is:: {self.operators} ************************************************************* \n')
    
      # Filter the dataframe by removing the categorical datatypes and zero variance feature variables
      print('***************************************  Removing the categorical variable columns, if there are any!!  *********************************************************** \n')
      
      self.df = self.df.select_dtypes(include=['float64','int64'])
      
      
      #Checking if we should go for the dimensionality 
      
      self .dimensionality = dimensionality
      
      if len(self.dimensionality) != self.df.shape[1] - 1: sys.exit('Given dimensionality is not matching the number of features given.!!')
      
      if self.dimensionality !=None:
          
          print('**********************************************************  Extracting the dimensions of the same variables and will perform the feature expansion accordingly.... *********************************************************************** \n')
          
          self.relational_units = relational_units
          
    
      # Pop out the Targer variable of the problem and convert to tensor
      self.df.rename(columns = {f'{self.df.columns[0]}':'Target'},inplace=True)
      
      self.Target_column = torch.tensor(self.df.pop('Target')).to(self.device)
      # If initial screening is  yes then do the mic screening.....
      
      if initial_screening != None:
          
          self.screening = initial_screening[0]
          
          self.quantile = initial_screening[1]
          
          self.df, self.dimensionality = self.feature_space_screening(self.df, self.dimensionality)
          
          self.dimensionality = list(self.dimensionality)
          
          
          
    
      # Create the feature values tensor
      self.df_feature_values = torch.tensor(self.df.values).to(self.device)
      
      self.feature_names = self.df.columns.tolist()
      
      self.dimensionality = symbols(self.dimensionality)
      
      self.rmse_metric = metrics[0]
      
      self.r2_metric = metrics[1]
      
      self.metrics = metrics
      
      self.output_dim = output_dim
    



    def get_dimensions_list(self):
        
        # Check for the shape of the feature variables and the length of the provided dimension
        
        if self.df_feature_values.shape[1] == len(self.dimensionality):
            
            print('************************************************ Shape of the dimension list and feature variable count matched... proceeding for further extraction and feature expansion..************************************************ \n')
            
        else:
            
            sys.exit('Mismatch between the dimension list provided and the number of feature variables... \n Please check the dimension list and feature variables and rerun the scipt.. \n ')
        
        #get the same dimensions from the list along with their index position.. 
        result ={}
        
        for index, value in enumerate(self.dimensionality):
            
            if value not in result:
                
                result[value] = []
                
            result[value].append(index)
        
        print('************************************ Extraction of dimensionless and same dimension variables is completed!!.. ********************************************** \n')
        
        if symbols('1') in result.keys():
            
            self.dimension_less = result[symbols('1')]
            
            del result[symbols('1')]
            
            print(f'************************************************ {len(self.dimension_less)} dimension less feature variables found in the given list!! ************************************************ \n')
        
            self.dimensions_index_dict = result
            
            del result
            
            #print(self.dimension_less,self.dimensions_index_dict)
            return self.dimensions_index_dict, self.dimension_less
        
        else:
            
            self.dimensions_index_dict = result
            
            self.dimension_less = None
            
            return self.dimensions_index_dict,self.dimension_less
        
        
    def replace_strings_with_other_elements(self,target_strings,relational_units):

        # Function to find the other element for a single target string
        def find_other_element(target_string):
            
            found_tuple = next((tup for tup in relational_units if target_string in tup), None)
            
            if found_tuple:
                
                return found_tuple[1] if found_tuple[0] == target_string else found_tuple[1]
            
            return target_string  # Return the target string itself if no other element is found

        # List comprehension to replace each target string with its other element
        #print('Replaced the dimensions with the relational units..')
        
        return [find_other_element(target_string) for target_string in target_strings]


    
    
    #### Feature expansion using dimension less numbers....
    
    def dimensionless_feature_expansion(self):
        
        feature_values_non_dimensional = torch.empty(self.df.shape[0],0).to(self.device)
        
        feature_names_non_dimensional =[]
        
        if self.dimension_less is None: 
            
            non_dimensions =[]
            
            print('****************************  Non-Dimension feature expansion is skipped because of no non-dimension features.... ********************************************************** \n ')
            
            return feature_values_non_dimensional,feature_names_non_dimensional,non_dimensions
        
        non_dim_features = self.df_feature_values[:,self.dimension_less]
        
        non_dimensional_features = np.array(self.feature_names)[self.dimension_less]
        
        print('******************************************************  Doing feature expansion of the non dimension feature variables... ******************************************************************** \n')
        
        for op in self.operators:
            
            
            ## Create an empty mutatable tensor and feature names list ####
            transformed_features = torch.empty(self.df.shape[0],0).to(self.device)
            
            transformed_feature_names = []
            
            #Transform the feature variables with exponential mathematical operator
            
            if op == 'exp':
                
                exp = torch.exp(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,exp),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(exp('+ x + "))", non_dimensional_features)))
                
            elif op == '/2':
                
                div2 = non_dim_features/2
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + ")/2)", non_dimensional_features)))
                
            elif op == '+1':
                
                div2 = non_dim_features +1
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "+1))", non_dimensional_features)))
                
            elif op == '-1':
                
                div2 = non_dim_features -1
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "-1))", non_dimensional_features)))
            
            elif op == '/2pi':
                
                div2 = non_dim_features/(2*math.pi)
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "/2pi))", non_dimensional_features)))
                
            elif op == '*2pi':
                
                div2 = non_dim_features*(2*math.pi)
                
                transformed_features = torch.cat((transformed_features,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+ x + "*2pi))", non_dimensional_features)))

            #Transform the feature variables with natural log mathematical operator
            
            elif op =='ln':
                
                ln = torch.log(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,ln),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(ln('+x + "))", non_dimensional_features)))
                
            
            #Transform the feature variables with log10 mathematical operator
            
            elif op =='log':
                
                log10 = torch.log10(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,log10),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(log('+x + "))", non_dimensional_features)))
                
            elif "pow" in op:
                
                import re
                
                pattern = r'\(([^)]*)\)'
                
                matches = re.findall(pattern, op)
                
                op = eval(matches[0])
                
                transformation = torch.pow(non_dim_features,op)
                
                transformed_features = torch.cat((transformed_features,transformation),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '('+x + f")^{matches[0]}", non_dimensional_features)))

            #Transform the feature variables with SINE mathematical operator
            
            elif op =='sin':
                
                sin = torch.sin(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,sin),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(sin('+x + "))", non_dimensional_features)))
             #Transform the feature variables with COSINE mathematical operator
             
            elif op =='cos':
                
                cos = torch.cos(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,cos),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(cos('+x + "))", non_dimensional_features)))

                
            #Transform the feature variables with reciprocal transformation
            
            elif op =='^-1':
                
                reciprocal = torch.reciprocal(non_dim_features)
                
                transformed_features = torch.cat((transformed_features,reciprocal),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(('+x + ")^(-1))", non_dimensional_features)))
                
            
            #Transform the feature variables with inverse exponential mathematical operator
            
            elif op =='exp(-1)':
                
                exp = torch.exp(non_dim_features)
                
                expreciprocal = torch.reciprocal(exp)
                
                transformed_features = torch.cat((transformed_features,expreciprocal),dim=1)
                
                transformed_feature_names.extend(list(map(lambda x: '(exp('+x + ")^(-1))", non_dimensional_features)))
                
                
            elif op == '+':
                
                if non_dim_features.shape[1] ==1: continue
                
                #generate the combinations on the fly
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                del combinations2
                
                comb_tensor = comb_tensor.permute(0,2,1)
                
                addition = torch.sum(comb_tensor,dim=2).T
                
                transformed_features = torch.cat((transformed_features,addition),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'+'.join(comb)+')', combinations1)))
                
                del combinations1
                
                
            elif op == '-':
                
                if non_dim_features.shape[1] ==1: continue
                
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                del combinations2
                
                comb_tensor = comb_tensor.permute(0,2,1)
                
                
                
                sub = torch.sub(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                
                transformed_features = torch.cat((transformed_features,sub),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'-'.join(comb)+')', combinations1)))
                
                del combinations1
                
            elif op == '*':
                
                if non_dim_features.shape[1] ==1: continue
                
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                del combinations2
                
                comb_tensor = comb_tensor.permute(0,2,1)

                mul = torch.multiply(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                
                transformed_features = torch.cat((transformed_features,mul),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
                
                del combinations1
                
            elif op == '/':
                
                if non_dim_features.shape[1] ==1: continue
                
                combinations1 = list(combinations(non_dimensional_features,2))
                
                combinations2 = torch.combinations(torch.arange(non_dim_features.shape[1]),2)
                
                comb_tensor = non_dim_features.T[combinations2,:]
                
                del combinations2
                
                comb_tensor = comb_tensor.permute(0,2,1)
                
                
                
                div1 = torch.div(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                
                div2 = torch.div(comb_tensor[:,:,1],comb_tensor[:,:,0]).T
                
                transformed_features = torch.cat((transformed_features,div1,div2),dim=1)
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
                
                transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
                
                del combinations1
                
                
            feature_values_non_dimensional = torch.cat((feature_values_non_dimensional,transformed_features),dim=1)
            
            feature_names_non_dimensional.extend(transformed_feature_names)
            
            
            # Check for the list of the features created whether it is empty or not and if it is empty return the empty tensors
            
        if len(feature_names_non_dimensional) == 0:
            
            non_dimensions=[]
            
            return feature_values_non_dimensional,feature_names_non_dimensional,non_dimensions
        
        # if feature names of non dimensional expansion is not zero then check for the nan and inf columns 
        
        else:
            
            nan_columns = torch.any(torch.isnan(feature_values_non_dimensional), dim=0)
            
            inf_columns = torch.any(torch.isinf(feature_values_non_dimensional), dim=0)
            
            nan_inf_columns = nan_columns|inf_columns
            
            feature_values_non_dimensional = feature_values_non_dimensional[:,~nan_inf_columns]
            
            feature_names_non_dimensional = [elem for i,elem in enumerate(feature_names_non_dimensional) if not nan_inf_columns[i]]
            
            non_dimensions = [symbols('1')]*feature_values_non_dimensional.shape[1]
            
            print('************************************************ Completed non dimensional feature expansion with features:', feature_values_non_dimensional.shape[1],'************************************************************** \n')
            
            return feature_values_non_dimensional, feature_names_non_dimensional,non_dimensions
            
            
            
            
    def dimension_to_non_dimension_feature_expansion(self):
        
        print('************************************************ Starting the feature expansion for converting the dimensional to non dimensional features ************************************************ \n')
        
        # We are converting dimensional feature space to non-dimensional feature space
        dim_to_non_dim_feature_values = torch.empty(self.df.shape[0],0).to(self.device)
        
        dim_to_non_dim_feature_names=[]
        
        dim_to_non_dim_units =[]
        
        for dimension, batch in self.dimensions_index_dict.items():

            '''
            will perform the feature expansion converting the dimensional feature spaces to non dimensional feature space by applying operators like 
            
            exp, sin, cos, tan, log, ln, tan, tanh, sinh
            
            '''
            dim_features_values = self.df_feature_values[:,batch]
            
            dim_features_names = np.array(self.feature_names)[batch]
            
            for op in self.operators:
                
                transformed_features = torch.empty(self.df.shape[0],0).to(self.device)
                
                transformed_feature_names = []
                
                if op == 'exp':
                    exp = torch.exp(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,exp),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(exp('+ x + "))", dim_features_names)))
   
                #Transform the feature variables with natural log mathematical operator
                
                elif op =='ln':
                    
                    ln = torch.log(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,ln),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(ln('+x + "))", dim_features_names)))
                    
                #Transform the feature variables with log10 mathematical operator
                
                elif op =='log':
                    
                    log10 = torch.log10(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,log10),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(log('+x + "))", dim_features_names)))
                    
                 
                elif op =='sin':
                    
                    sin = torch.sin(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,sin),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(sin('+x + "))", dim_features_names)))
                    
                 #Transform the feature variables with COSINE mathematical operator
                 
                elif op =='cos':
                    
                    cos = torch.cos(dim_features_values)
                    
                    transformed_features = torch.cat((transformed_features,cos),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(cos('+x + "))", dim_features_names)))
                    
                 #Transform the feature variables with inverse exponential mathematical operator
                
                elif op =='exp(-1)':
                    
                    exp = torch.exp(dim_features_values)
                    
                    expreciprocal = torch.reciprocal(exp)
                    
                    transformed_features = torch.cat((transformed_features,expreciprocal),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(exp('+x + ")^(-1))", dim_features_names)))
                    
                    
                dim_to_non_dim_feature_values = torch.cat((dim_to_non_dim_feature_values,transformed_features),dim=1)
                
                dim_to_non_dim_feature_names.extend(transformed_feature_names)
            
            
    
                
        nan_columns = torch.any(torch.isnan(dim_to_non_dim_feature_values), dim=0)
        
        inf_columns = torch.any(torch.isinf(dim_to_non_dim_feature_values), dim=0)
        
        nan_inf_columns = nan_columns|inf_columns
        
        dim_to_non_dim_feature_values = dim_to_non_dim_feature_values[:,~nan_inf_columns]
        
        dim_to_non_dim_feature_names = [elem for i,elem in enumerate(dim_to_non_dim_feature_names) if not nan_inf_columns[i]]
        
        dim_to_non_dim_units = [symbols('1')]*dim_to_non_dim_feature_values.shape[1]
        
        print('************************************************ Dimension to nondimension feature expansion completed.... with feature space size:', dim_to_non_dim_feature_values.shape[1],'************************************************ \n')
        
        
        return dim_to_non_dim_feature_values, dim_to_non_dim_feature_names,dim_to_non_dim_units
        
        
    def dimension_feature_expansion(self,iteration):
        
        dimension_features_values = torch.empty(self.df.shape[0],0).to(self.device)
        
        dimension_features_names =[]
        
        dimension_values = [] #since we can't use tensors for strings we are going to add for the mul and other operators 
        
        
        for dimension,batch in self.dimensions_index_dict.items():
            
            dim_features_values = self.df_feature_values[:,batch]
            
            dim_features_names = np.array(self.feature_names)[batch]
            
            dimension_copy = dimension

            for op in self.operators:
                
                
                transformed_features = torch.empty(self.df.shape[0],0).to(self.device)
                
                transformed_feature_names = []
                
                if op == '+':
                    
                    if len(dim_features_names) == 1: 
                        continue

                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    addition = torch.sum(comb_tensor,dim=2).T
                    
                    del combinations2
                    #pdb.set_trace()
                    transformed_features = torch.cat((transformed_features,addition),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'+'.join(comb)+')', combinations1)))
                    
                    del addition,combinations1,comb_tensor
                    
                    dimensions_screened = [dimension]*transformed_features.shape[1]
                    
                    
                    if self.relational_units!=None:
                        
                        dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)

                    dimension_values.extend(dimensions_screened)
                    
                    
                    
                elif op =='-':
                    
                    if len(dim_features_names) == 1: 
                        continue
                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    sub = torch.sub(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                    
                    del combinations2
                    
                    transformed_features = torch.cat((transformed_features,sub),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'-'.join(comb)+')', combinations1)))
                    
                    del combinations1,comb_tensor,sub
                    
                    dimensions_screened = [dimension]*transformed_features.shape[1]
                    
                    if self.relational_units!=None:
                        
                        dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                    
                    #add the dimension to the feature variables created
                    dimension_values.extend(dimensions_screened)
                    
                elif op =='*':
                    
                    if len(dim_features_names) == 1: 
                        continue
                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    mul = torch.multiply(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                    
                    del combinations2,comb_tensor
                    
                    transformed_features = torch.cat((transformed_features,mul),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
                    
                    del combinations1,mul
                    
                    dimension = Pow(dimension,2)
                    
                    dimensions_screened = [dimension]*transformed_features.shape[1]
                    
                    if self.relational_units!=None:
                        
                        dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                    
                    #add the dimension to the feature variables created
                    dimension_values.extend(dimensions_screened)
                    
                    dimension=dimension_copy
                
                elif "pow" in op:
                    
                    import re
                    
                    pattern = r'\(([^)]*)\)'
                    
                    matches = re.findall(pattern, op)
                    
                    op = eval(matches[0])
                    
                    transformation = torch.pow(dim_features_values,op)
                    
                    transformed_features = torch.cat((transformed_features,transformation),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '('+x + f")^{matches[0]}", dim_features_names)))

                    dimension = powdenest(Pow(dimension,op), force=True)

                    if len(dim_features_names) == 1: 
                        #pdb.set_trace()
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]

                        if self.relational_units!=None:
                        
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                    
                
                elif op =='+1':

                    sum1 = dim_features_values + 1
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "+1))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]

                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                
                elif op =='-1':

                    sum1 = dim_features_values - 1
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "+1))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
                        
                        
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                elif op =='/2':

                    sum1 = dim_features_values/2
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "+1))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
                        
                        
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                
                elif op =='/2pi':

                    sum1 = dim_features_values/(2*math.pi)
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "/2pi))", dim_features_names)))
                    
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
 
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                
                elif op =='*2pi':

                    sum1 = dim_features_values*(2*math.pi)
                    
                    transformed_features = torch.cat((transformed_features,sum1),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + "*2pi))", dim_features_names)))
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]

                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
  
                    
                elif op =='^-1':

                    inverse = torch.pow(dim_features_values,-1)
                    
                    transformed_features = torch.cat((transformed_features,inverse),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda x: '(('+x + ")^(-1))", dim_features_names)))
                    
                    dimension = Pow(dimension,-1)
                    
                    if len(dim_features_names) == 1: 
                        
                        dimension=[dimension]
                        
                        if self.relational_units!=None:
                            
                            dimension = self.replace_strings_with_other_elements(dimension, self.relational_units)
                            
                        dimension_values.extend(dimension)
                        
                    else:
                        
                        dimensions_screened = [dimension]*transformed_features.shape[1]
                        
                        if self.relational_units!=None:
                            
                            dimensions_screened = self.replace_strings_with_other_elements(dimensions_screened, self.relational_units)
                        
                        #add the dimension to the feature variables created
                        dimension_values.extend(dimensions_screened)
                        
                    dimension=dimension_copy
                    
                
                
                
                
                elif op =='/':
                    
                    if len(dim_features_names) == 1: 
                        continue
                    
                    combinations1 = list(combinations(dim_features_names,2))
                    
                    combinations2 = torch.combinations(torch.arange(dim_features_values.shape[1]),2)
                    
                    comb_tensor = dim_features_values.T[combinations2,:]
                    
                    del combinations2
                    
                    comb_tensor = comb_tensor.permute(0,2,1)
                    
                    div1 = torch.div(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
                    
                    div2 = torch.div(comb_tensor[:,:,1],comb_tensor[:,:,0]).T
                    
                    transformed_features = torch.cat((transformed_features,div1,div2),dim=1)
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
                    
                    transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
                    
                    
                    del combinations1,comb_tensor,div1,div2
                    
  
                if op =='/':
                    non_dimensional_div = transformed_features
                    
                    non_dimensional_div_features = transformed_feature_names
                    
                    non_dimensions_units = [symbols('1')]*len(non_dimensional_div_features)
                    
                    nan_columns = torch.any(torch.isnan(non_dimensional_div), dim=0)
                    
                    inf_columns = torch.any(torch.isinf(non_dimensional_div), dim=0)
                    
                    nan_inf_columns = nan_columns|inf_columns
                    
                    non_dimensional_div = non_dimensional_div[:,~nan_inf_columns]
                    
                    non_dimensional_div_features = [elem for i,elem in enumerate(non_dimensional_div_features) if not nan_inf_columns[i]]
                    
                    non_dimensions_units = [elem for i,elem in enumerate(non_dimensions_units) if not nan_inf_columns[i]]
                
                else:
                    
                    #if len(dim_features_names) == 1: 
                    
                    non_dimensional_div = torch.empty(self.df_feature_values.shape[0],0)
                    
                    non_dimensional_div_features = []
                    
                    non_dimensions_units=[]
                    
                    dimension_features_values = torch.cat((dimension_features_values,transformed_features),dim=1)
                    
                    dimension_features_names.extend(transformed_feature_names)
                    #print('operators:',op,'featurenames:',len(dimension_features_names),'feature_values:',dimension_features_values.shape,'dimension_values:',len(dimension_values))
            
        nan_columns = torch.any(torch.isnan(dimension_features_values), dim=0)
        
        inf_columns = torch.any(torch.isinf(dimension_features_values), dim=0)
        
        nan_inf_columns = nan_columns|inf_columns
        
        dimension_features_values = dimension_features_values[:,~nan_inf_columns]
        
        #pdb.set_trace()
        
        dimension_features_names = [elem for i,elem in enumerate(dimension_features_names) if not nan_inf_columns[i]]
        dimension_values = [elem for i,elem in enumerate(dimension_values) if not nan_inf_columns[i]]
        #pdb.set_trace()
        
        
        print('************************************************ Dimensional feature expansion completed.... with feature space size: ',dimension_features_values.shape[1],'************************************************ \n')
        
        if '/' not in self.operators:
            
            non_dimensional_div = torch.empty(self.df_feature_values.shape[0],0)
            
            non_dimensional_div_features = []
            
            non_dimensions_units=[]
        
        try:
            return dimension_features_values, dimension_features_names, dimension_values,non_dimensional_div,non_dimensional_div_features,non_dimensions_units
            
        except:
            
            non_dimensional_div = torch.empty(self.df_feature_values.shape[0],0)
            
            non_dimensional_div_features = []
            
            non_dimensions_units=[]
        
            return dimension_features_values, dimension_features_names, dimension_values,non_dimensional_div,non_dimensional_div_features,non_dimensions_units
    
    
    def inter_dimension_feature_expansion(self,iteration):
        
        combined_batch = torch.empty(self.df.shape[0],0)
        
        combined_dimensions =[]
        
        combined_feature_names=[]
        
        for dimension,batch in self.dimensions_index_dict.items():
            
            combined_batch = torch.cat((combined_batch,self.df_feature_values[:,batch]),dim=1)
            
            combined_dimensions.extend([dimension]*len(batch))
            
            combined_feature_names.extend(np.array(self.feature_names)[batch])
            
        if self.dimension_less !=None:    
            combined_batch = torch.cat((combined_batch,self.df_feature_values[:,self.dimension_less]),dim=1)
                
            combined_dimensions.extend([1]*len(self.dimension_less))
                
            combined_feature_names.extend(np.array(self.feature_names)[self.dimension_less])
        
        
        # do the combinations and perform the multiplication and the division operations
        transformed_features = torch.empty(self.df.shape[0],0)
        
        transformed_feature_names=[]
        
        transformed_dimensions=[]
        
        if '*' in self.operators:
            
            combinations1 = list(combinations(combined_feature_names,2))
            
            combinations2 = torch.combinations(torch.arange(combined_batch.shape[1]),2)
            
            comb_tensor = combined_batch.T[combinations2,:]
            
            comb_tensor = comb_tensor.permute(0,2,1)
            
            mul = torch.multiply(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
            
            del combinations2,comb_tensor
            
            transformed_features = torch.cat((transformed_features,mul),dim=1)
            
            transformed_feature_names.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))
            
            del combinations1,mul
            
            combinations1 = list(combinations(combined_dimensions,2))
            
            process_tuple = lambda x, y: (Pow(x,2) if x == y 
                                          else x*y)
            transformed_dimensions.extend(list(map(lambda t: process_tuple(*t), combinations1)))
            
            del combinations1
        
        
        if '/' in self.operators:
            
            combinations1 = list(combinations(combined_feature_names,2))
            
            combinations2 = torch.combinations(torch.arange(combined_batch.shape[1]),2)
            
            comb_tensor = combined_batch.T[combinations2,:]
            
            del combinations2
            
            comb_tensor = comb_tensor.permute(0,2,1)

            div1 = torch.div(comb_tensor[:,:,0],comb_tensor[:,:,1]).T
            
            div2 = torch.div(comb_tensor[:,:,1],comb_tensor[:,:,0]).T
            
            transformed_features = torch.cat((transformed_features,div1,div2),dim=1)
            
            transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
            
            transformed_feature_names.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))
            
            del combinations1
            
            combinations1 = list(combinations(combined_dimensions,2))
            
            dimensions=[]
            
            dimensions1=[]
            
            for x,y in combinations1:
                
                if x ==y: 
                    dimensions.append(symbols('1'))
                    
                    dimensions1.append(symbols('1'))
                else: 
                    dimensions.append(x/y)
                    
                    dimensions1.append(y/x)
    
            
            transformed_dimensions.extend(dimensions+dimensions1)
            
            
            del combinations1,dimensions,dimensions1
            
        screened_dimensions = transformed_dimensions
        
        if self.relational_units!=None:
            
            screened_dimensions = self.replace_strings_with_other_elements(screened_dimensions, self.relational_units)
           
        
        transformed_dimensions = screened_dimensions
        
        
        print('************************************************ Inter dimensional feature expansion completed, with feature space size: ', transformed_features.shape[1],'************************************************ \n')
        
        return transformed_features,transformed_feature_names,transformed_dimensions
        
    
    def feature_space_screening(self,df_sub,dimensions_screening):
        
        from sklearn.feature_selection import mutual_info_regression

        if self.screening == 'spearman':
            
            spear = spearmanr(df_sub.to_numpy(),self.Target_column,axis=0)
            
            screen1 = abs(spear.statistic)
            
            if screen1.ndim>1:screen1 = screen1[:-1,-1]
            
        elif self.screening=='mi':
            
            screen1 = mutual_info_regression(df_sub.to_numpy(), self.Target_column.numpy())
            
        
        
        df_screening = pd.DataFrame()
        
        df_screening['Feature variables'] = df_sub.columns
        
        df_screening['screen1'] = screen1
        
        df_screening = df_screening.sort_values(by = 'screen1',ascending= False).reset_index(drop=True)
        
        quantile_screen=df_screening.screen1.quantile(self.quantile)
        
        filtered_df = df_screening[(df_screening.screen1 > quantile_screen)].reset_index(drop=True)
        
        if filtered_df.shape[0]==0:
            filtered_df = df_screening[:int(df_sub.shape[1]/2)]

        df_screening1 = df_sub.loc[:,filtered_df['Feature variables'].tolist()]
        
        if len(dimensions_screening) == 0:
            return df_screening1,dimensions_screening
        
        indices = [df_sub.columns.tolist().index(item) for item in df_screening1.columns.tolist() if item in df_sub.columns.tolist()]

        screened_dimensions = np.array(dimensions_screening)[indices]
        
        
        return df_screening1, screened_dimensions
    
    
    def feature_expansion(self):
        
        
        print('\nStarting the feature expansion for the given feature variables.... \n ')
        
        for i in range(1,self.no_of_operators):
        
            start_time = time.time()
            
            # Get the dimension and non dimension variables... 
            
            self.get_dimensions_list()
            
            #Get the non dimension expansion... 
            
            non_dimension_feature_values, non_dimension_feature_names, non_dimension_units = self.dimensionless_feature_expansion()
            
            # Transform the dimension to non-dimension featur expansion....
            
            dim_to_non_dim_values,dim_to_non_dim_names,dim_to_non_dim_units = self.dimension_to_non_dimension_feature_expansion()
            
            
            #Transform the dimensional feature expansion... 
            
            dim_exp_feature_values, dim_exp_feature_names, dim_exp_units,non_dim_expanded_values,non_dim_expanded_names,non_dim_expanded_units = self.dimension_feature_expansion(i)
            
            # Inter-dimension feature expansion
            
            dim_inter_exp_values, dim_inter_exp_names,dim_inter_exp_units = self.inter_dimension_feature_expansion(i)
            
            # Concatenate the values to feature values, variables, dimensionality......
           
            self.df_feature_values = torch.cat((self.df_feature_values,non_dimension_feature_values,dim_to_non_dim_values,dim_exp_feature_values,dim_inter_exp_values,non_dim_expanded_values),dim=1)
        
            self.feature_names.extend(non_dimension_feature_names + dim_to_non_dim_names + dim_exp_feature_names + dim_inter_exp_names+non_dim_expanded_names)
            
            self.dimensionality.extend(list(non_dimension_units) + list(dim_to_non_dim_units) + list(dim_exp_units) + list(dim_inter_exp_units) + list(non_dim_expanded_units))
            

            end_time = time.time()
            
            print(f'************************************************ Time taken for the {i} feature expansion: ',end_time - start_time, ' seconds.. ************************************************ \n')
            
            print(f'************************************************ Size of the feature space formed in the {i} expansion',self.df_feature_values.shape[1],'************************************************ \n')
            
            
        return self.df_feature_values,self.Target_column,self.feature_names,self.dimensionality
            
        
        

    
        
    




       
     

