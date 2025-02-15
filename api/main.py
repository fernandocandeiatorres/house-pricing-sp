from fastapi import FastAPI
from pydantic import BaseModel
from models.housemodel import HousingPriceModel

model = HousingPriceModel()

app = FastAPI()

class InputData(BaseModel):
    area: float
    bedrooms: int
    garage: int
    district_name: str
    
@app.post("/predict")
def get_prediction(data: InputData):
    price = model.predict_price(data.area, data.bedrooms, data.garage, data.district_name)
    return {"predicted_price": float(price)}
