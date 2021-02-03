
import logging, random, json, joblib, sklearn
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.neighbors import NearestNeighbors
from pickle import dump

log = logging.getLogger(__name__)
router = APIRouter()

url = 'https://raw.githubusercontent.com/boscolio/spotify_data/main/spotify.csv'
songs = pd.read_csv(url)
list_of_names = songs.name

predictor = pickle.load('model_knn.pkl')
print('pickled model loaded')

class Song(BaseModel):
    """Use this data model to parse the request body JSON."""

    song: str = Field(..., example="Yesterday")

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

@router.post('/predict')
async def predict(song: Song):
    
    predict_song = predictor.predict([song.name])[0]
    
    # making a new df for our predicted value 
    y_pred_df = songs.loc[songs['name'] == predict_song.upper()]
   
    names = y_pred_df.name.to_json(orient='values')
    artists = y_pred_df.artists.to_json(orient='values')
    

    return {
        'names': names,
        'artists': artists
    }
