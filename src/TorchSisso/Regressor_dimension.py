#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:00:15 2024

@author: muthyala.7
"""

import pandas as pd 
import numpy as np 
import torch 
import time 
from scipy.stats import spearmanr
import sys
import pdb


class Regressor:
    
    def __init__(self,x,y,names,dimensionality,dimension=None,sis_features=10,device='cpu',output_dim=None,screening=None):

        '''
        ###################################################################################################################

        x, y, names - are the outputs of the Feature Expansion class which defines the expanded feature space, target tensor, names of the expanded features to use in the equation
        
        dimensionality -- describes the units of the variables created.. 
        
        dimension - defines the number of terms in the linear equation generation 
        
        output_dim - dimensionality of the target variable

        sis_features - defines the number of top features needs to be considered for building the equation

        ###################################################################################################################
        '''
        self.device = device
        
        self.x = x.to(self.device)
        
        self.y = y.to(self.device)
        
        self.names = names
        
        if dimension !=None: 
            
            self.dimension = dimension
            
            self.sis_features = sis_features
            
        else: 
            self.dimension = 3 
            #Maximum terms we will be looking at...
            self.sis_features = 10
            
        #self.dimension = dimension
        self.dimensionality = dimensionality
        
        self.output_dim = output_dim
        #self.sis_features = sis_features
        
        if screening!=None:
            
            self.screening = screening[0]
            
            self.quantile = screening[1]
            
            df_sub = pd.DataFrame(x,columns=names)
            
            screened_space,screened_dimensions = self.feature_space_screening(df_sub,self.dimensionality)
            
            self.x = torch.tensor(screened_space.values)
            
            self.names = screened_space.columns.tolist()
            
            self.dimensionality = screened_dimensions
           
        if self.output_dim!=None:
            
            self.get_dimensions_list()
            
            self.x = self.x[:,self.dimension_less]
            
            x = pd.Series(self.names)
            
            self.names = x.iloc[self.dimension_less].tolist()
            #print(self.names)
            
            

        # Transform the features into standardized format
        self.x_mean = self.x.mean(dim=0)
        
        self.x_std = self.x.std(dim=0)
        
        self.y_mean = self.y.mean()
        
        self.names1 = self.names.copy()
        

        # Transform the target variable value to mean centered
        self.y_centered = self.y - self.y_mean
        
        self.x_standardized = ((self.x - self.x_mean)/self.x_std)
        
        self.scores = []
        
        self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
        
        self.residual = torch.empty(self.y_centered.shape).to(self.device)
        
        self.x_std_clone = torch.clone(self.x_standardized)
        
        
    
    
    def feature_space_screening(self,df_sub,dimensions_screening):
        
        from sklearn.feature_selection import mutual_info_regression

        if self.screening == 'spearman':
            
            spear = spearmanr(df_sub.to_numpy(),self.y,axis=0)
            
            screen1 = abs(spear.statistic)
            
            if screen1.ndim>1:screen1 = screen1[:-1,-1]
            
        elif self.screening=='mi':
            
            screen1 = mutual_info_regression(df_sub.to_numpy(), self.y.numpy())
            
        
        
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
        
        

    '''
    #######################################################################################################

    Constructs the linear equation based on the number of top sis features and the dimension requested.

    #######################################################################################################
    '''
    def higher_dimension(self,iteration):

        #Indices values that needs to be assinged zero, because we already calculated the SIS score of those values..
        ind = (self.indices[:,-1][~torch.isnan(self.indices[:,-1])]).to(self.device)
        
        # Assign them to zero and calculate the score
        self.x_standardized[:,ind.tolist()] = 0

        scores= torch.abs(torch.mm(self.residual,self.x_standardized))
        
        #Assign nan values to zero...

        scores[torch.isnan(scores)] = 0
        
        #Replce the zero values with the original values to compute the combinations and multivariable regression..

        self.x_standardized[:,ind.tolist()] = self.x_std_clone[:,ind.tolist()]

        sorted_scores, sorted_indices = torch.topk(scores, k= self.sis_features)

        sorted_indices = sorted_indices.T
        
        sorted_indices_earlier = self.indices[:((iteration-1)*self.sis_features),(iteration-1)].unsqueeze(1)

        sorted_indices = torch.cat((sorted_indices_earlier,sorted_indices),dim=0)
        
        #Checks for the shape of dimension and sis features needs to be screened and if it is less then adds the remaining count of nan rows..
        if sorted_indices.shape[0] < self.indices.shape[0]:
            
            remaining = (self.sis_features*self.dimension) - int(sorted_indices.shape[0])
            
            nan = torch.full((remaining,1),float('nan')).to(self.device)
            
            sorted_indices = torch.cat((sorted_indices,nan),dim=0)
            
            self.indices = torch.cat((self.indices,sorted_indices),dim=1)

        else:
            self.indices = torch.cat((self.indices,sorted_indices),dim=1)

        comb1 = self.indices[:,-1][~torch.isnan(self.indices[:,-1])]
        
        combinations_generated = torch.combinations(comb1,(int(self.indices.shape[1])-1))
        
        y_centered_clone = self.y_centered.unsqueeze(1).repeat(len(combinations_generated.tolist()),1,1).to(self.device)
        
        comb_tensor = self.x_standardized.T[combinations_generated.tolist(),:]
        
        x_p = comb_tensor.permute(0,2,1)
        
        start_c = time.time()

        sol, _, _, _ = torch.linalg.lstsq(x_p, y_centered_clone)
        
        predicted = torch.matmul(x_p,sol)
        
        residuals = y_centered_clone - predicted
        
        square = torch.square(residuals)
        
        mean = torch.mean(square,dim=1,keepdim=True)
        
        min_value, min_index = torch.min(mean, dim=0)

        coefs_min = torch.squeeze(sol[min_index]).unsqueeze(1)
        
        indices_min  = torch.squeeze(combinations_generated[min_index])
        
        non_std_coeff = ((coefs_min.T/self.x_std[indices_min.tolist()]))
        
        non_std_intercept = self.y.mean() - torch.dot(self.x_mean[indices_min.tolist()]/self.x_std[indices_min.tolist()],coefs_min.flatten())
        
        self.residual = self.y_centered - torch.mm(coefs_min.T,self.x_standardized[:,indices_min.tolist()].T)
        
        rmse = float(torch.sqrt(torch.mean(self.residual**2)))

        r2 = 1 - (float(torch.sum(self.residual**2))/float(torch.sum((self.y_centered)**2)))
        
        terms = []

        for i in range(len(non_std_coeff.squeeze())):
            
            term = str(float(non_std_coeff.squeeze()[i])) + "*" + str(self.names[int(indices_min[i])])
            
            terms.append(term)

        return float(rmse),terms,non_std_intercept,non_std_coeff,r2
    
    
    '''
    ###################################################################
    
    Defines the function to get the dimensionality of feature variables and divide them into separate dimensionality...
    
    ####################################################################
    '''
    
    def get_dimensions_list(self):
        
        #get the same dimensions from the list along with their index position.. 
        result ={}
        
        for index, value in enumerate(self.dimensionality):
            
            if value not in result:
                
                result[value] = []
                
            result[value].append(index)
            
        
        
        if self.output_dim in result.keys():
            
            
            print('************************************************ Extraction of target dimension feature variables found.., performing the regression!!.. ************************************************ \n')
            
            self.dimension_less = result[self.output_dim]
            
            del result[self.output_dim]
            
            print(f'************************************************ {len(self.dimension_less)} output dimension feature variables found in the given list!! ************************************************ \n')
        
            self.dimensions_index_dict = result
            
            del result
            
            
            return self.dimensions_index_dict, self.dimension_less
        
        else:
            
            print('No target dimension feature variables found.. exiting the program..')
            sys.exit()

    '''
    ##########################################################################################################################

    Defines the function to model the equation

    ##########################################################################################################################
    '''
    def regressor_fit(self):
        
        if self.x.shape[1] > self.sis_features*self.dimension :
            print('\n')
            print(f"************************************************ Starting SyMANTIC model in {self.device} ************************************************")
            print('\n')
        else:
            print('************************************************ Given Number of features in SIS screening is greater than the feature space created, changing the SIS features to shape of features created ************************************************ \n')
            
            self.sis_features = int(self.x.shape[1])
            
            self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
            
            
        #Looping over the dimensions
        
        for i in range(1,self.dimension+1):
            
            if i ==1:
                
                start_1D = time.time()

                #calculate the scores
                scores = torch.abs(torch.mm(self.y_centered.unsqueeze(1).T,self.x_standardized))

                #Set the NaN values claculation to zero, instead of removing 
                scores[torch.isnan(scores)] = 0
                

                #Sort the top number of scores based on the sis_features 
                sorted_scores, sorted_indices = torch.topk(scores,k=self.sis_features)
                
                sorted_indices = sorted_indices.T
                
                remaining = torch.tensor((self.sis_features*self.dimension) - int(sorted_indices.shape[0])).to(self.device)

                #replace the remaining indices with nan
                nan = torch.full((remaining,1),float('nan')).to(self.device)
                
                sorted_indices = torch.cat((sorted_indices,nan),dim=0)
                
                #store the sorted indices as next column
                
                self.indices = torch.cat((self.indices,sorted_indices),dim=1)
                
                selected_index = self.indices[0,1]
                
                x_in = self.x[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias = torch.cat((torch.ones_like(x_in), x_in), dim=1).to(self.device)

                #Calculate the intercept and coefficient, Non standardized
                coef1, _, _, _ = torch.linalg.lstsq(x_with_bias, self.y) #Change to x_with_bias 

                
                #Calculate the residuals based on the standardized and centered values
                x_in1 = self.x_standardized[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias1 = torch.cat((torch.ones_like(x_in1), x_in1), dim=1)
                
                coef, _, _, _ = torch.linalg.lstsq(x_with_bias1, self.y_centered)
            

                self.residual = (self.y_centered - (coef[1]*self.x_standardized[:, int(selected_index)])).unsqueeze(1).T
                rmse = float(torch.sqrt(torch.mean(self.residual**2)))
                
                r2 = 1 - (float(torch.sum(self.residual**2))/float(torch.sum((self.y_centered)**2)))
                #from torcheval.metrics.functional import r2_score
                rmse = float(torch.sqrt(torch.mean(self.residual**2)))
                #self.residual = residual
                coefficient = coef[1]
                
                intercept = self.y.mean() - torch.dot((self.x_mean[int(selected_index)]/self.x_std[int(selected_index)]).reshape(-1), coef[1].reshape(-1))#coef1[0]
                
                if intercept >= 0:
                    coefficient = coef[1]/self.x_std[int(selected_index)]
                    
                    coefficient = "{:.20f}".format(float(coefficient))
                    
                    equation =  coefficient+ '*' + str(self.names[int(selected_index)]) + '+' + str(float(intercept))
                    
                    print('Equation: ', equation)
                    
                    print('RMSE: ', rmse)
                    
                    print('R2:',r2)
                    
                else:
                    
                    coefficient = coef[1]/self.x_std[int(selected_index)]
                    
                    coefficient = "{:.20f}".format(float(coefficient))
                    
                    equation = coefficient + '*' + str(self.names[int(selected_index)])  + str(float(intercept))
                    
                    print('Equation: ', equation)
                    
                    print('RMSE: ', rmse)
                    
                    print('R2::',r2)
                #print('Time taken to generate one dimensional equation: ', time.time()-start_1D,'seconds')
                if self.device == 'cuda':torch.cuda.empty_cache()
                
                
                
            else:
                start = time.time()

                rmse,terms,intercept,coefs,r2 = self.higher_dimension(i)
                equation =''
                for k in range(len(terms)):
                    
                    print(f'{k+1} term in the equation is {terms[k]}')
                    
                    print('\n')

                    if coefs.flatten()[k] > 0:
                        
                        equation = equation + ' + ' + (str(terms[k]))+'  '
                        
                    else:
                        
                        equation = equation + (str(terms[k])) + '  '

                print('Equation: ',equation[:len(equation)-1])
                print('\n')

                print('Intercept:', float(intercept))
                print('\n')

                print('RMSE:',float(rmse))
                print('\n')
                
                print('R2::',r2)

                print(f'Time taken for {i} dimension is: ', time.time()-start)
                
                if self.device == 'cuda': torch.cuda.empty_cache()
                
                

        return float(rmse),equation,r2
