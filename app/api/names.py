from fastapi import APIRouter
import json
import pandas as pd

router = APIRouter()

url = 'https://raw.githubusercontent.com/boscolio/spotify_data/main/spotify.csv'
songs = pd.read_csv(url)

@router.get('/names')
async def names():
  names = []
  for i in range(songs.shape[0]):
    if  in :
      names.append(songs.name.iloc[i])
names_json = json.dumps(names)

return '[dict of names]', names_json
