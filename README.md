#  <p align="center">TorchSISSO: A PyTorch-Based Implementation of the Sure Independence Screening and Sparsifying Operator (SISSO) for Efficient and Interpretable Model Discovery

![torchsisso3](https://github.com/user-attachments/assets/a8d52ec3-3470-4807-904a-52525dc2b5d0)

## What is SISSO?

The **Sure Independence Screening and Sparsifying Operator (SISSO)** is a symbolic regression (SR) method that searches for interpretable models by exploring large feature spaces. It has proven particularly effective in fields like materials science, where it helps to uncover simple, accurate models that describe complex physical phenomena.

**TorchSISSO [PAPER](https://arxiv.org/abs/2410.01752)** is a native Python implementation of SISSO, built using the PyTorch framework. It addresses the limitations of the original FORTRAN-based SISSO [SISSO](https://github.com/rouyang2017/SISSO) by providing a faster, more flexible, and easier-to-use solution.


## Installation

You can install TorchSISSO via pip:
```
pip install TorchSisso
```

Import your data and use the following code to fit a SISSO model
```python 
# import TorchSisso model class along with other useful packages
from TorchSisso import SissoModel
import numpy as np
import pandas as pd
import sympy
from sympy import symbols

# create dataframe composed of targets "y" and primary features "X"
data = np.column_stack((y, X))
df = pd.DataFrame(data)

#define unary and binary operators of interest
operators = ['+','-','*','/','exp','ln','sin','pow(2)']

# create SISSO model object with relevant user-defined inputs
sm = SissoModel(df= df #Takes the dataframe as input with first variable as target variable
                operators = operators #Takes the user-defined operators to perform the feature engineering
                n_expansion = 3 #Defines the number of feature expansions need to be considered
                n_term = 1 #Defines the number of terms in the final equation
                k = 20 #Defines the number of SIS features to be screened for $L_0$ regularization
                initial_screening = ["mi" or "spearman", quantile value] #Defines the feature screening option for high dimensional and 1-quantile_value defines
                                                                          #the features within this quantile range should be kept for feature expansion.
                use_gpu = True or False #Defines the flag whether to consider GPU or not (For efficient computation we consider using GPU only for $L_0$ Regularization.
                dimensionality = ['u1','u2','u3'] #Defines the units of the feature variables in string representation which later converted into sympy format to do the                                                         meaningful feature construction.
                relational_units = [(symbols('u1')*symbols('u2'),symbols('u3)] #Defines the list of tuples where each tuple represents the relational transformation.
                output_dim = (symbols('u1')*symbols('u1')) #Defines the units of the target variable which helps in narrowing down the space for Regularization.
                custom_unary_functions = [('cos', '-sin', 'exp', 'pow(2)'), ('tanh', 'pow(2)','sin')] #list of tuples (Note only torch supported functions can be customized)
                custom_binary_functions = [
                                          {
                                              'functions1': ['pow(2)'],  # Function to apply on var1 (X1)
                                              'functions2': ['sin'],  # No function to apply on var2 (X2)
                                              'operation': torch.add,  # Operation to apply between var1 and var2
                                              'final_functions': []  # Final function to apply to the result of addition
                                          }
                                      ]
#List of dicts where transformations of variables and operation is provided.. (For example the given code above gives sin(x2) + X^2)

                )

# Run the SISSO algorithm to get the interpretable model with the highest accuracy
rmse, equation, r2,_ = sm.fit()
```


# Usage Examples
Usage of TorchSisso can be found in   <a href="https://colab.research.google.com/drive/1q0TEEALkb1PzJuusGKyHphv7tfod66XA?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

About
------
Created and maintained by Madhav Muthyala. Please feel free to open issues in the Github or contact Madhav  
(muthyala.7@osu.edu) in case of any problems/comments/suggestions in using the code. 

**Known Limitations and Future Work**
-----------------------------------

We are aware of current limitations in the code and are actively working on addressing them in future updates. If you encounter any issues or have suggestions for improvements, please feel free to report them. Your feedback is valuable and helps us enhance the project.
