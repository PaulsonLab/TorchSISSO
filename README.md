
# TorchSISSO

Implementation of Sure Independence Screening and Sparsifying Operator (SISSO) algorithm in Python using PyTorch.

# SISSO

SISSO algorithm mainly consists of two steps, one is Feature Space Construction (FSC) and the other is identifying the correlated features using SIS and fit the linear model using SO. 

With the provided initial feature set (or primary feature set) and a operator set, SISSO constructs feature space with given constriant of complexity of the number of operators in the feature variables created. 

</br>
For example:
Initial Feature set: [x1,x2] </br>
Operator set: [+,exp] </br>
Constructed Feature Space: [x1+x2,exp(x1),exp(x1+x2),x1,x2]  (This is for complexity constraint of 3) 
</br>
</br>
Once the feature space is constructed it will invoke the Sure Independnece Screening (SIS) -- Identifying the important feature variables with the correlation coefficient as metric and ranks the features based on the absolute value of scores (correlation coefficient), and then Sparisfying Operator (L0 Regularization is equipped) to identify the best feature and fit a linear model. 
</br>

The SISSO regression algorithm is detailed by the original authors in R. Ouyang, S. Curtarolo, E. Ahmetcik et al., Phys. Rev. Mater. 2, 083802 (2018), R. Ouyang, E. Ahmetcik, C. Carbogno, M. Scheffler, and L. M. Ghiringhelli, J. Phys.: Mater. 2, 024002 (2019).
</br>

Fortran implementation can be found at https://github.com/rouyang2017/SISSO.

# Installation steps 
```
pip install TorchSisso
```

# Usage Examples
Usage of TorchSisso can be found in   <a href="https://github.com/PaulsonLab/TorchSISSO/blob/main/Test_Regressor_SISSO.ipynb">
  <img src="https://colab.research.google.com/drive/1q0TEEALkb1PzJuusGKyHphv7tfod66XA?usp=sharing" alt="Open In Colab"/>
</a>

