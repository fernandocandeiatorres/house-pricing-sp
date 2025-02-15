import tensorflow as tf
import numpy as np
import pickle
import joblib
import os

class HousingPriceModel:
    
    def __init__(self, model_path="/models/housing_model.keras",
                 scaler_path="/models/scaler.pkl",
                 district_means_path="models/district_means.pkl",
                 y_train_path="models/y_train.pkl"
                 ):
        
        #Load model
        self.model = tf.keras.models.load_model(os.path.abspath("models/housing_model.keras"))
        
        #Load Scaler
        self.scaler = joblib.load(os.path.abspath("models/scaler.pkl"))
        
        # Load Y mean
        with open(os.path.abspath(y_train_path), "rb") as f:
            self.y_train = pickle.load(f)
            
        # Load Dist mean
        with open(os.path.abspath(district_means_path), "rb") as f:
            self.district_means = pickle.load(f)
            
    def preprocess_input(self, area:float, bedrooms:int, garage:int, district: str):
        print("district getting...")
        dist_value = self.district_means.get(district.lower(), self.y_train.mean()) # try to get mean rent price for that dist, if not then default
        
        input_values = np.array([[area, bedrooms, garage, dist_value]])
        
        input_scaled = self.scaler.transform(input_values)
        
        return input_scaled
    
    def predict_price(self, area:float, bedrooms:int, garage:int, dist:str):
        
        input_scaled = self.preprocess_input(area, bedrooms, garage, dist)
        pred = self.model.predict(input_scaled)[0][0]
        return pred
        
        
            