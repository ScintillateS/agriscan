import requests
import json

url = 'https://api.windy.com/api/point-forecast/v2'
myobj = {"lat": 49.809,
    "lon": 16.787,
    "model": "gfs",
    "parameters": ["wind"],
    "levels": ["surface"],
    "key": "bzKT7ODJ9WUl8LvNJuvc9EieeW329x91"}
x = requests.post(url, json = myobj)
y = json.loads(x.text)
latestxwind = y['wind_u-surface'][-1] #latest wind data point from the x direction (west to east)
latestywind = y['wind_v-surface'][-1] #latest wind data point from the y direction (south to north)
print(latestxwind, latestywind)