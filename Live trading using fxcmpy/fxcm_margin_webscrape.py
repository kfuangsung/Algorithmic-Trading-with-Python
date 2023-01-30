import time
import pandas as pd 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def main():

    URL = "https://www.fxcm.com/markets/trading-details/margin-updates/"
    option = webdriver.ChromeOptions()
    option.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)
    driver.implicitly_wait(30)
    driver.get(URL)
    time.sleep(5) # wait for page datails to appear
    page = driver.find_element(By.TAG_NAME, 'html')
    page.send_keys(Keys.END)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.close()
    rows = soup.find('table').find('tbody').find_all('tr')

    # website does not include cryto margin
    margin_dict = {}
    margin_dict['symbol'] = []
    margin_dict['margin'] = []

    for row in rows:
        
        symbol = row.find('td', attrs={'data-title': 'Instrument: '})
        if symbol is None: continue
        symbol = symbol.get_text()
        
        margin = row.find('td', attrs={'data-title': 'Current: '})
        margin = margin.get_text()
        margin = margin.split(' ')[0].strip('$').replace(',','')
        
        margin_dict['symbol'].append(symbol)
        margin_dict['margin'].append(margin)

    try:
        assert len(margin_dict['symbol']) == len(margin_dict['margin'])
    except:
        print("len(symbol) != len(margin)")
        print(f"Number of symbol: {len(margin_dict['symbol'])}")
        print(f"Number of margin: {len(margin_dict['margin'])}")
        
    margin_df = pd.DataFrame(data=margin_dict)
    fname = 'fxcm_margin_requirements.csv'
    margin_df.to_csv(fname, index=False)
    print(f"Saved '{fname}'")
    
    
if __name__ == '__main__':
    main()