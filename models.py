from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import plot_importance,plot_tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import pickle


def rgb_predict(data):
    loaded_model = pickle.load(open('rf_model.sav', 'rb'))
    result = loaded_model.predict(data)

    return result

def xgb_predict(data):
    loaded_model = pickle.load(open('xgb_model.sav', 'rb'))
    result = loaded_model.predict(data)

    return result

def dnn_predict(data):
    loaded_model = pickle.load(open('dnn_model.sav', 'rb'))
    result = loaded_model.predict(data)

    return result

if __name__ == '__main__':
    pass


