#rapidapi
import http.client
import json 
import pandas as pd
conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "26c98e8b88msh4fd9ec64f0fd81ep16b3d7jsnc46504c0744f",
    'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com" #it tells server which api service we are targeting
}

conn.request("GET", "/product-details?asin=B07ZPKBL9V&country=US", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
json_data = json.loads(data.decode("utf-8"))
json_dataframe = pd.DataFrame(json_data)

print(json_dataframe)  # json to dataframe


# SUMMARY
# http.client  is built in python library which is used to create hhtp and https request,means we can use it to connect with server (API)
# httpconnection :- it will act as a bridge between client and server which will connect them , "real -time blah blah is server's address"
# headers:- it is like an identity card and instructions which we share with requests to server
# get:- it tells that we only need data , we don't want to modify anything ,/product-details blah blah is the endpoint of the api
# headers=headers means we have joined api key with host //
# utf-8 :- converts data into human readable format
# server :- means a room where data is stored , endpoint:- is a specific room where our required information is located