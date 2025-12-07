# import pandas as pd
# import requests 
# import time
# response = requests.get("https://api.themoviedb.org/3/movie/top_rated?api_key=920b966b4fb7df3b98e183e65e7e06b4&language=en-US&page=1")
# print(response.json()['results'])
# df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
# # print(df.head())

# dk = pd.DataFrame()
# for i in range(1,512):
#     response = requests.get("https://api.themoviedb.org/3/movie/top_rated?api_key=920b966b4fb7df3b98e183e65e7e06b4&language=en-US&page={}".format(i))
#     temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
#     dk = pd.concat([dk, temp_df], ignore_index=True)
#     time.sleep(0.5)





# import pandas as pd 
# import requests 
# import time

# response = requests.get("https://api.themoviedb.org/3/movie/top_rated?api_key=920b966b4fb7df3b98e183e65e7e06b4&language=en-US&page=1")
# print(response.json()['results'])

# # total pages nikal lo
# total_pages = response.json()['total_pages']

# dk = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
# print(dk.head())
#   # ab fixed 25 ki jagah total_pages use hoga
# for i in range(2,total_pages+1):   # aadhe pages tak hi chalega

#     response = requests.get("https://api.themoviedb.org/3/movie/top_rated?api_key=920b966b4fb7df3b98e183e65e7e06b4&language=en-US&page={}".format(i))
#     temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','vote_average','vote_count']]
#     dk = pd.concat([dk, temp_df], ignore_index=True)   # append ki jagah concat
    # time.sleep(0.2)  # thoda delay add kar lo





import pandas as pd
import requests
import time

dk = pd.DataFrame()

for i in range(1, 513):  # 1 to 512 inclusive
    try:
        url = f"https://api.themoviedb.org/3/movie/top_rated?api_key=920b966b4fb7df3b98e183e65e7e06b4&language=en-US&page={i}"
        response = requests.get(url)
        response.raise_for_status()  # catch bad status codes

        temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
        dk = pd.concat([dk, temp_df], ignore_index=True)

        print(f"Page {i} done ✅")
        time.sleep(0.3)  # small delay to avoid rate limiting

    except Exception as e:
        print(f"Error on page {i}: {e}")
        time.sleep(2)  # wait longer and retry if error

dk.to_csv("top_rated_movies.csv", index=False)  # save to CSV