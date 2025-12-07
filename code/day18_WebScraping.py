from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
# Start browser (this will open Chrome)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open website
driver.get("https://www.ambitionbox.com/list-of-companies?page=1")

# Get full HTML (after JavaScript executes)
html = driver.page_source
# print(html)


soup = BeautifulSoup(html , 'lxml')
# print(soup.find_all('h1')[0].text)

# print(soup.find_all('h2')[0].text)
# for i in soup.find_all('h2'):
#     print(i.text.strip())  #strip removes extra spaces { \n \t etc}

# for rating in soup.find_all('div', class_='rating_text'):
#     print(rating.text.strip())

for i in soup.find_all('span',class_='companyCardWrapper__companyRatingCount'):
    print(i.text.strip())

# Close browser
driver.quit()
