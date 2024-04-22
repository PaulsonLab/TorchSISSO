



'''
##############################################################################################

Importing the required libraries 

##############################################################################################
'''

import torch
import warnings
warnings.filterwarnings('ignore')
import itertools
import time 
import torch.nn as nn
import torch.optim as optim
import itertools
import pdb

class SISSORegressor:
    
    def __init__(self,x,y,names,dimension=1,sis_features=20,device='cpu'):

        '''
        ###################################################################################################################

        x, y, names - are the outputs of the Feature Expansion class which defines the expanded feature space, target tensor, names of the expanded features to use in the equation

        dimension - defines the number of terms in the linear equation generation 

        sis_features - defines the number of top features needs to be considered for building the equation

        ###################################################################################################################
        '''
        self.device = device
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.names = names
        self.dimension = dimension
        self.sis_features = sis_features

        # Transform the features into standardized format
        self.x_mean = self.x.mean(dim=0)
        self.x_std = self.x.std(dim=0)
        self.y_mean = self.y.mean()

        # Transform the target variable value to mean centered
        self.y_centered = self.y - self.y_mean
        self.x_standardized = ((self.x - self.x_mean)/self.x_std)

        self.scores = []
        self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
        self.residual = torch.empty(self.y_centered.shape).to(self.device)
        self.x_std_clone = torch.clone(self.x_standardized)

    '''
    #######################################################################################################

    Constructs the linear equation based on the number of top sis features and the dimension requested.

    #######################################################################################################
    '''
    def higher_dimension(self,iteration):

        #Indices values that needs to be assinged zero 
        ind = (self.indices[:,-1][~torch.isnan(self.indices[:,-1])]).to(self.device)

        self.x_standardized[:,ind.tolist()] = 0

        scores= torch.abs(torch.mm(self.residual,self.x_standardized))

        scores[torch.isnan(scores)] = 0

        self.x_standardized[:,ind.tolist()] = self.x_std_clone[:,ind.tolist()]

        sorted_scores, sorted_indices = torch.topk(scores, k= self.sis_features)

        sorted_indices = sorted_indices.T
        
        sorted_indices_earlier = self.indices[:((iteration-1)*self.sis_features),(iteration-1)].unsqueeze(1)

        sorted_indices = torch.cat((sorted_indices_earlier,sorted_indices),dim=0)

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
  
        rmse = torch.sqrt(min_value)
        coefs_min = torch.squeeze(sol[min_index]).unsqueeze(1)
        indices_min  = torch.squeeze(combinations_generated[min_index])
        non_std_coeff = ((coefs_min.T/self.x_std[indices_min.tolist()]))
        non_std_intercept = self.y.mean() - torch.dot(self.x_mean[indices_min.tolist()]/self.x_std[indices_min.tolist()],coefs_min.flatten())
        self.residual = self.y_centered - torch.mm(coefs_min.T,self.x_standardized[:,indices_min.tolist()].T)
        
        terms = []

        for i in range(len(non_std_coeff.squeeze())):
            term = str(float(non_std_coeff.squeeze()[i])) + "*" + str(self.names[int(indices_min[i])])
            terms.append(term)

        return float(rmse),terms,non_std_intercept,non_std_coeff

    '''
    ##########################################################################################################################

    Defines the function to model the equation

    ##########################################################################################################################
    '''
    def SISSO(self):
        
        if self.x.shape[1] > self.sis_features*self.dimension :
            print('\n')
            print(f"Starting SISSO in {self.device}")
            print('\n')
        else:
            print('Given Number of features in SIS screening is greater than the feature space created, changing the SIS features to shape of features created')
            self.sis_features = self.x.shape[1]
            self.indices = torch.arange(1, (self.dimension*self.sis_features+1)).view(self.dimension*self.sis_features,1).to(self.device)
            #raise RuntimeError(f'Number of features in SIS screening are greater than total number of features, SISSO cannot be performed. Please input the valid number when product of {self.dimension}*{self.sis_features} that is less than  {self.x.shape[1]}')
        
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
                coef1, _, _, _ = torch.linalg.lstsq(x_with_bias, self.y)

                #Calculate the residuals based on the standardized and centered values
                x_in1 = self.x_standardized[:, int(selected_index)].unsqueeze(1)

                # Add a column of ones to x for the bias term
                x_with_bias1 = torch.cat((torch.ones_like(x_in1), x_in1), dim=1)
                coef, _, _, _ = torch.linalg.lstsq(x_with_bias1, self.y_centered)

                self.residual = (self.y_centered - (coef[1]*self.x_standardized[:, int(selected_index)])).unsqueeze(1).T
                rmse = float(torch.sqrt(torch.mean(self.residual**2)))

                coefficient = coef1[1]
                intercept = coef1[0]
                if intercept > 0:
                    equation = str(float(coefficient)) + '*' + str(self.names[int(selected_index)]) + '+' + str(float(intercept))
                    print('Equation: ', equation)
                    print('\n')
                    print('RMSE: ', rmse)
                else:
                    equation = str(float(coefficient)) + '*' + str(self.names[int(selected_index)])  + str(float(intercept))
                    print('Equation: ', equation)
                    print('\n')
                    print('RMSE: ', rmse)
                print('\n')
                print('Time taken to generate one dimensional equation: ', time.time()-start_1D,' seconds')
                print('\n')
                if self.device == 'cuda':torch.cuda.empty_cache()
            else:
                start = time.time()

                rmse,terms,intercept,coefs = self.higher_dimension(i)
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

                print(f'Time taken for {i} dimension is: ', time.time()-start)
                if self.device == 'cuda': torch.cuda.empty_cache()

        return float(rmse),equation

