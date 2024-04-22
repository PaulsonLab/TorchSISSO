from . import FeatureSpaceConstruction as fc
from . import SISSO_REGRESSOR_TORCH as sr

class SISSO_model:

  def __init__(self,df,operators,no_of_operators=3,dimension=1,sis_features=20,device='cpu'):

    self.operators = operators
    self.df=df
    self.no_of_operators = no_of_operators
    self.device = device
    self.dimension = dimension
    self.sis_features = sis_features

  def fit(self):

    x,y,names = fc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device).feature_space()
    
    rmse, equation =  sr.SISSORegressor(x,y,names,self.dimension,self.sis_features,self.device).SISSO()

    return rmse, equation



