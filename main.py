import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from sklearn.metrics import mean_squared_error, r2_score
import models

filename = r"C:\Users\sahus\Desktop\Python\datasets\New folder\New folder\Test Set.xlsx")

test_df = pd.read_excel(filename)

test_df.drop(['Is Working Day'],axis=1,inplace=True)
test_df.reset_index(drop=True,inplace=True)
test_df.set_index('Date',inplace=True)
test_df.drop(['Sgn0 VolumeDir'],axis=1,inplace=True)
test_df.drop(['SDSH DAP'],axis=1,inplace=True)

test_df.drop(['Hour','Weekday'],axis=1,inplace=True)
scaler = MinMaxScaler()
test_X = scaler.fit_transform(test_df.iloc[:,:-1])

print('RandomForest','XGBoost','ANN',end=' ')
print('choose one')

which_model = input()

def pick_model(which_model,data):
    if which_model == 'RandomForest':
        predictions = model.rgb_predict(data)
    elif which_model == 'XGBoost':
        predictions = model.xgb_predict(data)
    else:
        predictions = model.dnn_predict(Data)


predictions = pd.DataFrame(predictions)

predictions['actual'] = y_test.values

predictions.rename(columns = {0:'prediction'},inplace=True)

predictions['difference'] = np.abs(predictions['actual'] - predictions['prediction'])
predictions['% error'] = ((predictions['difference'] + 0.5) /(predictions['actual'] + 0.5))*100

print(predictions['% error'].mean())

predictions.to_csv(r"C:\Users\sahus\Desktop\Python\datasets\New folder\New folder\dnn_results.csv")