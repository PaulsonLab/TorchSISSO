
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 21:42:41 2023

@author: muthyala.7
"""




'''
##############################################################################################

Importing the required libraries

##############################################################################################
'''
import torch
import pandas as pd
import numpy as np
import warnings
import itertools
import time
from itertools import combinations

class feature_space_construction:

  '''
  ##############################################################################################################

  Define global variables like number of operators and the input data frame and the operator set given

  ##############################################################################################################
  '''
  def __init__(self,operators,df,no_of_operators=3,device='cpu'):

    print(f'Starting Feature Space Construction in {device}')
    print('\n')
    '''
    ###########################################################################################

    no_of_operators - defines the presence of operators (binary or unary) in the expanded features space

    For example: if no_of_operators = 3 then the space will be limited to formation of features with 3 operators (x1+x2)/x3 or exp(x1+x2)

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
    self.operators = operators
    self.device = torch.device(device)

    # Filter the dataframe by removing the categorical datatypes and zero variance feature variables

    self.df = self.df.select_dtypes(include=['float64','int64'])

    # Compute the variance of each column
    variance = self.df.var()

    # Get the names of the zero variance columns
    zero_var_cols = variance[variance == 0].index

    # Drop the zero variance columns from the dataframe
    self.df = self.df.drop(zero_var_cols, axis=1)

    # Pop out the Targer variable of the problem and convert to tensor
    self.df.rename(columns = {f'{self.df.columns[0]}':'Target'},inplace=True)
    self.Target_column = torch.tensor(self.df.pop('Target')).to(self.device)

    # Create the feature values tensor
    self.df_feature_values = torch.tensor(self.df.values).to(self.device)
    self.columns = self.df.columns.tolist()

    #Create a dataframe for appending new datavalues
    self.new_features_values = pd.DataFrame()

    #Creating empty tensor and list for single operators (Unary operators)
    self.feature_values_unary = torch.empty(self.df.shape[0],0).to(self.device)
    self.feature_names_unary = []

    #creating empty tensor and list for combinations (Binary Operators)
    self.feature_values_binary = torch.empty(self.df.shape[0],0).to(self.device)
    self.feature_names_binary = []

  '''
  ###############################################################################################################

  Construct all the features that can be constructed using the single operators like log, exp, sqrt etc..

  ###############################################################################################################
  '''

  def single_variable(self,operators_set):

    #Check for the validity of the operator set given
    for op in operators_set:

      if op in ['exp','sin','cos','sqrt','cbrt','log','ln','^-1','^2','^3','exp(-1)']:
        continue
      else:
        raise TypeError('Invalid unary operator found in the given operator set')

    #Looping over operators set to get the new features/predictor variables

    for op in operators_set:

        self.feature_values_11 = torch.empty(self.df.shape[0],0).to(self.device)
        feature_names_12 =[]

        # Performs the exponential transformation of the given feature space
        if op == 'exp':
            exp = torch.exp(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,exp),dim=1)
            feature_names_12.extend(list(map(lambda x: '(exp('+ x + "))", self.columns)))

        # Performs the natural lograithmic transformation of the given feature space
        elif op =='ln':
            ln = torch.log(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,ln),dim=1)
            feature_names_12.extend(list(map(lambda x: '(ln('+x + "))", self.columns)))

        # Performs the lograithmic transformation of the given feature space
        elif op =='log':
            log10 = torch.log10(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,log10),dim=1)
            feature_names_12.extend(list(map(lambda x: '(log('+x + "))", self.columns)))

        # Performs the cuberoot transformation of the given feature space
        elif op =='cbrt':
            cbrt = torch.pow(self.df_feature_values,1/3)
            self.feature_values_11 = torch.cat((self.feature_values_11,cbrt),dim=1)
            feature_names_12.extend(list(map(lambda x: '(cbrt('+x + "))", self.columns)))

        # Performs the squareroot transformation of the given feature space
        elif op == 'sqrt':
            sqrt = torch.pow(self.df_feature_values,1/2)
            self.feature_values_11 = torch.cat((self.feature_values_11,sqrt),dim=1)
            feature_names_12.extend(list(map(lambda x: '(sqrt('+x + "))", self.columns)))

        # Performs the sine function transformation of the given feature space
        elif op =='sin':
            sin = torch.sin(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,sin),dim=1)
            feature_names_12.extend(list(map(lambda x: '(sin('+x + "))", self.columns)))

        # Performs the hyperbolic sine function transformation of the given feature space
        elif op =='sinh':
            sin = torch.sinh(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,sin),dim=1)
            feature_names_12.extend(list(map(lambda x: '(sin('+x + "))", self.columns)))

        # Performs the cosine transformation of the given feature space
        elif op =='cos':
            cos = torch.cos(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,cos),dim=1)
            feature_names_12.extend(list(map(lambda x: '(cos('+x + "))", self.columns)))

        # Performs the hyperbolic cosine transformation of the given feature space
        elif op =='cosh':
            cos = torch.cosh(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,cos),dim=1)
            feature_names_12.extend(list(map(lambda x: '(cos('+x + "))", self.columns)))

        # Performs the hyperbolic tan transformation of the given feature space
        elif op =='tanh':
            cos = torch.tanh(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,cos),dim=1)
            feature_names_12.extend(list(map(lambda x: '(cos('+x + "))", self.columns)))

        # Performs the square transformation of the given feature space
        elif op == '^2':
            square = torch.pow(self.df_feature_values,2)
            self.feature_values_11 = torch.cat((self.feature_values_11,square),dim=1)
            feature_names_12.extend(list(map(lambda x: '(('+x + ")^2)", self.columns)))

        # Performs the cube transformation of the given feature space
        elif op =='^3':
            cube = torch.pow(self.df_feature_values,3)
            self.feature_values_11 = torch.cat((self.feature_values_11,cube),dim=1)
            feature_names_12.extend(list(map(lambda x: '(('+x + ")^3)", self.columns)))

        # Performs the Inverse transformation of the given feature space
        elif op =='^-1':
            reciprocal = torch.reciprocal(self.df_feature_values)
            self.feature_values_11 = torch.cat((self.feature_values_11,reciprocal),dim=1)
            feature_names_12.extend(list(map(lambda x: '(('+x + ")^(-1))", self.columns)))

        # Performs the Inverse exponential transformation of the given feature space
        elif op =='exp(-1)':
            exp = torch.exp(self.df_feature_values)
            expreciprocal = torch.reciprocal(exp)
            self.feature_values_11 = torch.cat((self.feature_values_11,expreciprocal),dim=1)
            feature_names_12.extend(list(map(lambda x: '(exp('+x + ")^(-1))", self.columns)))

        self.feature_values_unary = torch.cat((self.feature_values_unary,self.feature_values_11),dim=1)
        self.feature_names_unary.extend(feature_names_12)

        del self.feature_values_11, feature_names_12



    #Check for empty lists
    if len(self.feature_names_unary) == 0:
        return self.feature_values_unary, self.feature_names_unary
    else:
        # create Boolean masks for NaN and Inf values

        nan_columns = torch.any(torch.isnan(self.feature_values_unary), dim=0)
        inf_columns = torch.any(torch.isinf(self.feature_values_unary), dim=0)
        nan_or_inf_columns = nan_columns | inf_columns

        # Remove columns from tensor
        self.feature_values_unary = self.feature_values_unary[:, ~nan_or_inf_columns]

        # Remove corresponding elements from list
        self.feature_names_unary = [elem for i, elem in enumerate(self.feature_names_unary) if not nan_or_inf_columns[i]]
        return self.feature_values_unary, self.feature_names_unary



  '''
  ################################################################################################

  Defining method to perform the combinations of the variables with the initial feature set
  ################################################################################################
  '''
  def combinations(self,operators_set):

    #Checking for the operator set
      for op in operators_set:
        if op in ['+','-','*','/']:
          continue
        else:
          raise TypeError("Valid set of operators +,-,*,/, abs please check the operators set")
          break

      #getting list of cobinations without replacement using itertools
      combinations1 = list(combinations(self.columns,2))

      combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)

      comb_tensor = self.df_feature_values.T[combinations2,:]

      #Reshaping to match
      x_p = comb_tensor.permute(0,2,1)

      del comb_tensor #Deleting to release the memory

      for op in operators_set:
          self.feature_values11 = torch.empty(self.df.shape[0],0).to(self.device)
          feature_names_11 = []

          # Performs the addition transformation of feature space with the combinations generated
          if op =='+':
              sum = torch.sum(x_p,dim=2).T
              self.feature_values11 = torch.cat((self.feature_values11,sum),dim=1)
              feature_names_11.extend(list(map(lambda comb: '('+'+'.join(comb)+')', combinations1)))

          # Performs the subtraction transformation of feature space with the combinations generated
          elif op =='-':
              sub = torch.sub(x_p[:,:,0],x_p[:,:,1]).T
              self.feature_values11 = torch.cat((self.feature_values11,sub),dim=1)
              feature_names_11.extend(list(map(lambda comb: '('+'-'.join(comb)+')', combinations1)))

          # Performs the division transformation of feature space with the combinations generated
          elif op == '/':
              div1 = torch.div(x_p[:,:,0],x_p[:,:,1]).T
              div2 = torch.div(x_p[:,:,1],x_p[:,:,0]).T
              self.feature_values11 = torch.cat((self.feature_values11,div1,div2),dim=1)
              feature_names_11.extend(list(map(lambda comb: '('+'/'.join(comb)+')', combinations1)))
              feature_names_11.extend(list(map(lambda comb: '('+'/'.join(comb[::-1])+')', combinations1)))

          # Performs the multiplication transformation of feature space with the combinations generated
          elif op == '*':
              mul = torch.multiply(x_p[:,:,0],x_p[:,:,1]).T
              self.feature_values11 = torch.cat((self.feature_values11,mul),dim=1)
              feature_names_11.extend(list(map(lambda comb: '('+'*'.join(comb)+')', combinations1)))

          self.feature_values_binary = torch.cat((self.feature_values_binary,self.feature_values11),dim=1)
          self.feature_names_binary.extend(feature_names_11)
          del self.feature_values11,feature_names_11
      
      #Checking whether the lists are empty
      if len(self.feature_names_binary) == 0:
          return self.feature_values_binary, self.feature_names_binary
      
      else:
          #Removing Nan and inf columns from tenosr and corresponding variable name form the list

          nan_columns = torch.any(torch.isnan(self.feature_values_binary), dim=0)
          inf_columns = torch.any(torch.isinf(self.feature_values_binary), dim=0)
          nan_or_inf_columns = nan_columns | inf_columns

          # Remove columns from tensor
          self.feature_values_binary = self.feature_values_binary[:, ~nan_or_inf_columns]

          # Remove corresponding elements from list
          self.feature_names_binary = [elem for i, elem in enumerate(self.feature_names_binary) if not nan_or_inf_columns[i]]

          #Returning the dataframe created
          return self.feature_values_binary,self.feature_names_binary #created_space


  '''
  ##########################################################################################################

  Creating the space based on the given set of conditions

  ##########################################################################################################

  '''

  def feature_space(self):

    #Initial check on the dimension
    if self.no_of_operators+1 > 9: print('****************************************************** \n','Currently TorchSisso supports Feature Space Expansion till complexity of 7, provided input argument > 7. Featureee Expansion with complexity of 7 will be returned \n', '*******************************************************')

    start_time = time.time()

    # Split the operator set into combinations set and unary set
    basic_operators = [op for op in self.operators if op in ['+', '-', '*', '/','abs']]
    other_operators = [op for op in self.operators if op not in ['+', '-', '*', '/']]

    print('Starting Initial Feature Expansion')


    #Performs the feature space expansion based on the binary operator set provided
    values, names = self.combinations(basic_operators)

    # Performs the feature space expansion based on the unary operator set provided
    values1, names1 = self.single_variable(other_operators)

    features_created = torch.cat((values,values1),dim=1)
    del values, values1
    #names.extend(names1)
    names2 = names + names1
    del names,names1

    space = pd.DataFrame(features_created,columns = names2)
    space = space.round(7)
    space = space.T.drop_duplicates().T
    space = space.dropna(axis=1, how='any')
    del features_created, names2

    print('\n')
    print('Number Of Features In The Initial Expanded Feature Space: ', space.shape[1])
    print('\n')

    self.df_feature_values = torch.cat((self.df_feature_values,torch.tensor(space.values)),dim=1)
    self.columns.extend(space.columns.tolist())
    del space

    print('Number Of Features in First Feature Space Expansion After Screening: ',int(self.df_feature_values.shape[1]))
    print('\n')

    print('Time taken to create and screen features for first level feature expansion:  ', round(time.time() - start_time,3), ' seconds')
    print('\n')

    print('Starting Second Level Feature Expansion')

    start_time = time.time()

    values, names = self.combinations(basic_operators)
    values1, names1 = self.single_variable(other_operators)
    features_created = torch.cat((values,values1),dim=1)
    #names.extend(names1)
    names2 = names + names1
    del names,names1

    print('\n')
    print('Number Of Features In The Second Expanded Feature Space: ', features_created.shape[1])

    space = pd.DataFrame(features_created,columns = names2)
    del values, values1,features_created, names2
    space = space.round(7)
    space = space.T.drop_duplicates().T
    space = space.dropna(axis=1, how='any')

    self.df_feature_values = torch.cat((self.df_feature_values,torch.tensor(space.values)),dim=1)
    self.columns.extend(space.columns.tolist())
    del space
    print('\n')

    print('Number Of Features in Second Feature Space Expansion After Screening: ',int(self.df_feature_values.shape[1]))
    print('\n')

    print('Time taken to create and screen features for second level feature expansion:  ', round(time.time() - start_time,3), ' seconds')
    print('\n')

    # Check for the dimension complexity and performs the feature expansion as needed
    if self.no_of_operators >3:

      for i in range(4,self.no_of_operators+1):
        if i == 4:
          print('Starting Third Level Feature Expansion')
          print('\n')
          start_time = time.time()

          values, names = self.combinations(basic_operators)
          values1, names1 = self.single_variable(other_operators)
          features_created = torch.cat((values,values1),dim=1)
          #names.extend(names1)
          names2 = names + names1
          del names,names1          

          space = pd.DataFrame(features_created,columns = names2)
          
          space = space.T.drop_duplicates().T
          space = space.dropna(axis=1, how='any')

          del values, values1,names2, features_created

          if self.no_of_operators+1 == 5:
            self.df_feature_values = torch.cat((self.df_feature_values,torch.tensor(space.values)),dim=1)
            self.columns.extend(space.columns.tolist())
            del space

            print('Number Of Features in Third Feature Space Expansion: ',int(self.df_feature_values.shape[1]))
            print('\n')
            print('Time taken to create and screen features for third level feature expansion: ',  round(time.time()-start_time,3) ,'seconds')
            print('\n')
            return self.df_feature_values,self.Target_column,self.columns
          else:
            print('Continuing to Next level of feature expansion')
            print('\n')
            continue

        if i == 5:
          print('Starting Fourth Level Feature Expansion')
          print('\n')
          start_time = time.time()

          values, names = self.combinations(basic_operators)
          values1, names1 = self.single_variable(other_operators)
          features_created = torch.cat((values,values1),dim=1)
          #names.extend(names1)
          names2 = names+names1

          space = pd.DataFrame(features_created,columns = names2)
          space = space.T.drop_duplicates().T
          space = space.dropna(axis=1, how='any')

          del values, values1,names,names1, features_created,names2

          if self.no_of_operators+1 == 6:
            self.df_feature_values = torch.cat((self.df_feature_values,torch.tensor(space.values)),dim=1)
            self.columns.extend(space.columns.tolist())
            del space

            print('Number Of Features in Fourth Feature Space Expansion: ',int(self.df_feature_values.shape[1]))
            print('\n')
            print('Time taken to create and screen features for fourth level feature expansion: ',  round(time.time()-start_time,3) ,'seconds')
            print('\n')
            return self.df_feature_values,self.Target_column,self.columns
          else:
            print('Continuing to Next level of feature expansion')
            print('\n')
            continue
        if i == 6:
          print('Starting Fifth Level Feature Expansion')
          print('\n')
          start_time = time.time()

          values, names = self.combinations(basic_operators)
          values1, names1 = self.single_variable(other_operators)
          features_created = torch.cat((values,values1),dim=1)
          #names.extend(names1)
          names2 = names+names1
          del names,names1

          space = pd.DataFrame(features_created,columns = names2)
          space = space.T.drop_duplicates().T
          space = space.dropna(axis=1, how='any')

          del values, values1, names2, features_created


          if self.no_of_operators+1 == 7:
            self.df_feature_values = torch.cat((self.df_feature_values,torch.tensor(space.values)),dim=1)
            self.columns.extend(space.columns.tolist())
            del space

            print('Number Of Features in fifth Feature Space Expansion: ',int(self.df_feature_values.shape[1]))
            print('\n')
            print('Time taken to create and screen features for fifth level feature expansion: ',  round(time.time()-start_time,3) ,'seconds')
            print('\n')
            return self.df_feature_values,self.Target_column,self.columns

          else:
            
            print('Continuing to Next level of feature expansion')
            print('\n')
            continue
        if i == 7:

          print('Starting Sixth Level Feature Expansion')
          print('\n')

          start_time = time.time()
          values, names = self.combinations(basic_operators)
          values1, names1 = self.single_variable(other_operators)
          features_created = torch.cat((values,values1),dim=1)
          #names.extend(names1)
          names2 = names+names1
          del names,names1
          space = pd.DataFrame(features_created,columns = names2)
          space = space.T.drop_duplicates().T
          space = space.dropna(axis=1, how='any')

          del values, values1,features_created, names2

          if self.no_of_operators+1 == 8:
            self.df_feature_values = torch.cat((self.df_feature_values,torch.tensor(space.values)),dim=1)
            self.columns.extend(space.columns.tolist())
            del space
            print('Number Of Features in sixth Feature Space Expansion: ',int(self.df_feature_values.shape[1]))
            print('\n')
            print('Time taken to create and screen features for sixth level feature expansion: ',  round(time.time()-start_time,3) ,'seconds')
            print('\n')
            return self.df_feature_values,self.Target_column,self.columns
          else:
            print('\n')
            print('Currently TorchSisso supports Feature Space Expansion till complexity of 7 mathematical operators, provided input argument > 7. Further Feature Expansion cannot be performed, returning the last expanded feature space')
            return self.df_feature_values,self.Target_column,self.columns

    else:
      return self.df_feature_values, self.Target_column,self.columns
