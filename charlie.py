from sklearn.model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

model = make_pipeline(
  OneHotEncoder(),
  SimpleImputer(),
  Ridge(),
)
