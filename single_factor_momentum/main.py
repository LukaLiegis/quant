import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'id': 'constituents'})

sp500_data = []

for row in table.find_all('tr')[1:]:
    cols = row.find_all('td')
    if len(cols) >= 4:
        ticker = cols[0].text.strip()
        company = cols[1].text.strip()
        sector = cols[3].text.strip()
        sub_sector = cols[4].text.strip() if len(cols) > 4 else ''

        ticker = ticker.replace('.', '-')

        sp500_data.append({
            'ticker': ticker,
            'company': company,
            'sector': sector,
            'sub_sector': sub_sector
        })

data = pd.DataFrame(sp500_data)