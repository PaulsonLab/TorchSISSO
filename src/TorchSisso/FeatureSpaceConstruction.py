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
from . import combinations_construction
import re
from .combinations_construction import FeatureConstructor

class feature_space_construction:

  '''
  ##############################################################################################################

  Define global variables like number of operators and the input data frame and the operator set given

  ##############################################################################################################
  '''
  def __init__(self,operators,df,no_of_operators=None,device='cpu',initial_screening=None,custom_unary_functions=None,custom_binary_functions=None):

    print(f'************************************************ Starting Feature Space Construction in {device} ************************************************ \n')
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
    
    
    if initial_screening != None:
        
        self.screening = initial_screening[0]
        
        self.quantile = initial_screening[1]
        
        self.df = self.feature_space_screening(self.df)
        


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
    
    self.custom_unary_functions = custom_unary_functions
    
    self.custom_binary_functions = custom_binary_functions
    
    

  '''
  ###############################################################################################################

  Construct all the features that can be constructed using the single operators like log, exp, sqrt etc..

  ###############################################################################################################
  '''
  def feature_space_screening(self,df_sub):
        
        from sklearn.feature_selection import mutual_info_regression
        from scipy.stats import spearmanr

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
        
        
        return df_screening1
    
  def construct_function(self,functions_tuple, var):
      
    result = var  # Start with the input variable
    text1 = ''  # Initialize text1 as an empty string
    text2 = ''  # Initialize text2 as an empty string
    
    for func in functions_tuple:
        # Check for power function of the form pow(n)
        if func.startswith('pow'):
            # Extract the power value inside parentheses
            power_value = int(re.search(r'pow\((\d+)\)', func).group(1))
            result = result ** power_value  # Use ** for power operation
            text2 = f")**{power_value}" + text2
            text1 = "(" + text1
        # Check if the function has a negative sign
        elif func.startswith('-'):
            func_name = func[1:]  # Remove the negative sign
            result = -getattr(torch, func_name)(result)  # Apply the function with a negative sign
            text1 = f"-{getattr(torch, func_name).__name__}(" + text1
            text2 = ")" + text2
        else:
            result = getattr(torch, func)(result)  # Apply the function normally
            text1 = f"{getattr(torch, func).__name__}(" + text1
            text2 = ")" + text2

    return result, text1, text2  


  def single_variable(self,operators_set,i):


    #Looping over operators set to get the new features/predictor variables
    
    if len(operators_set)==0 and self.custom_unary_functions!=None:
        
        for func in self.custom_unary_functions:
            
            self.feature_values_11 = torch.empty(self.df.shape[0],0).to(self.device)
            
            feature_names_12 =[]
            
            self.feature_values_11,t1,t2 = self.construct_function(func,self.df_feature_values)
            
            feature_names_12.extend(list(map(lambda x: str(t1) +x + str(t2), self.columns)))
            
            self.feature_values_unary = torch.cat((self.feature_values_unary,self.feature_values_11),dim=1)
            
            self.feature_names_unary.extend(feature_names_12)

            del self.feature_values_11, feature_names_12

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

        # Performs the power transformations of the feature variables..
        
        elif "pow" in op:
            
            import re
            
            pattern = r'\(([^)]*)\)'
            matches = re.findall(pattern, op)
            op = eval(matches[0])
            
            transformation = torch.pow(self.df_feature_values,op)
            self.feature_values_11 = torch.cat((self.feature_values_11,transformation),dim=1)
            feature_names_12.extend(list(map(lambda x: '('+x + f")^{matches[0]}", self.columns)))

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
            
        elif op =='+1':
            add1 = self.df_feature_values + 1
            self.feature_values_11 = torch.cat((self.feature_values_11,add1),dim=1)
            feature_names_12.extend(list(map(lambda x: '('+x + "+1)", self.columns)))
            
        elif op =='-1':
            sub1 = self.df_feature_values - 1
            self.feature_values_11 = torch.cat((self.feature_values_11,sub1),dim=1)
            feature_names_12.extend(list(map(lambda x: '('+x + "-1)", self.columns)))
            
        elif op =='/2':
            div2 = self.df_feature_values/2
            self.feature_values_11 = torch.cat((self.feature_values_11,div2),dim=1)
            feature_names_12.extend(list(map(lambda x: '('+x + "/2)", self.columns)))
            
        elif self.custom_unary_functions!=None:
            
            for func in self.custom_functions:
                
                
                self.feature_values_11,t1,t2 = self.construct_function(func,self.df_feature_values_11)
                
                feature_names_12.extend(list(map(lambda x: str(t1) +x + str(t2), self.columns)))
        

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
        
        if self.no_of_operators!=None:
            
            if i+1!=self.no_of_operators:
                
                #Get the duplicate columns in the feature space created..
                unique_columns, indices = torch.unique(self.feature_values_unary,sorted=False, dim=1, return_inverse=True)
                
                # Get the indices of the unique columns
                unique_indices = indices.unique()
        
                # Remove duplicate columns
                self.feature_values_unary = self.feature_values_unary[:, unique_indices]
    
                # Remove the corresponding elements from the list of feature names..
                self.feature_names_unary = [self.feature_names_unary[i] for i in unique_indices.tolist()]
            
        
        return self.feature_values_unary, self.feature_names_unary



  '''
  ################################################################################################

  Defining method to perform the combinations of the variables with the initial feature set
  ################################################################################################
  '''
  def combinations(self,operators_set,i):
      
      
      if len(operators_set)==0 and self.custom_binary_functions!=None:
          
          self.feature_values11 = torch.empty(self.df.shape[0],0).to(self.device)
          feature_names_11 = []
          
          constructor = FeatureConstructor(self.df_feature_values, self.columns)
          
          results, expressions = constructor.construct_generic_features(self.custom_binary_functions)
          self.feature_values11 = torch.cat((self.feature_values11,results),dim=1)
          feature_names_11.extend(expressions)
          self.feature_values_binary = torch.cat((self.feature_values_binary,self.feature_values11),dim=1)
          self.feature_names_binary.extend(feature_names_11)
          del self.feature_values11,feature_names_11
          
          
          
      for op in operators_set:
          s = time.time()
          #getting list of cobinations without replacement using itertools
          combinations1 = list(combinations(self.columns,2))

          combinations2 = torch.combinations(torch.arange(self.df_feature_values.shape[1]),2)

          comb_tensor = self.df_feature_values.T[combinations2,:]

          #Reshaping to match
          x_p = comb_tensor.permute(0,2,1)

          del comb_tensor,combinations2 #Deleting to release the memory
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
        
          elif self.custom_binary_functions!=None:
              
              constructor = FeatureConstructor(self.df_feature_values, self.columns)
              
              results, expressions = constructor.construct_generic_features(self.custom_binary_functions)
              self.feature_values11 = torch.cat((self.feature_values11,results),dim=1)
              feature_names_11.extend(expressions)
              
          self.feature_values_binary = torch.cat((self.feature_values_binary,self.feature_values11),dim=1)
          self.feature_names_binary.extend(feature_names_11)
          del self.feature_values11,feature_names_11
          #print('Operator::',op,time.time()-s)
      
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

          #Get the duplicate columns in the feature space created..
          if self.operators !=None:
              
              if i+1 != self.no_of_operators:
                  
                  unique_columns, indices = torch.unique(self.feature_values_binary, sorted=False,dim=1, return_inverse=True)
                  
                  # Get the indices of the unique columns
                  unique_indices = indices.unique()
        
                  # Remove duplicate columns
                  self.feature_values_binary = self.feature_values_binary[:, unique_indices]
                  
                  
                  # Remove the corresponding elements from the list of feature names..
                  self.feature_names_binary = [self.feature_names_binary[i] for i in unique_indices.tolist()]

          #Returning the featurespace created
          return self.feature_values_binary,self.feature_names_binary #created_space


  '''
  ##########################################################################################################

  Creating the space based on the given set of conditions

  ##########################################################################################################

  '''

  def feature_space(self):
 
        basic_operators = [op for op in self.operators if op in ['+','-','*','/']]
        other_operators = [op for op in self.operators if op not in ['+','-','*','/']]
        for i in range(1,self.no_of_operators):
            
            start_time = time.time()
            
            print(f'************************************************ Starting {i} level of feature expansion...************************************************ \n')
    
            #Performs the feature space expansion based on the binary operator set provided
            values, names = self.combinations(basic_operators,i)
        
            # Performs the feature space expansion based on the unary operator set provided
            values1, names1 = self.single_variable(other_operators,i)
        
            features_created = torch.cat((values,values1),dim=1)
            
            del values, values1
            
            names2 = names + names1
            
            del names,names1
            
            self.df_feature_values = torch.cat((self.df_feature_values,features_created),dim=1)
            
            self.columns.extend(names2)
            
            del features_created,names2
            
            print(f'************************************************ {i} Feature Expansion Completed with feature space size:::',self.df_feature_values.shape[1],'************************************************ \n')
            
            print('************************************************ Time taken to create the space is:::', time.time()-start_time, ' Seconds...************************************************ \n')
            
            
        return self.df_feature_values, self.Target_column,self.columns
    
