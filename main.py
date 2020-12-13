from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt


finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['DB', 'MS', 'CS']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent':'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        title = row.a.text
        datetime_data = row.td.text.split(' ')

        if len(datetime_data) == 1:
            time = datetime_data[0]
        else:
            date = datetime_data[0]
            time = datetime_data[1]

        parsed_data.append([ticker,date,time,title])

df = pd.DataFrame(parsed_data, columns=['ticker','date','time','title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date

plt.figure(figsize=(10,8))

mean_df = df.groupby(['ticker','date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound',axis="columns").transpose()
mean_df.plot(kind='bar')
plt.show()