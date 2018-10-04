
# coding: utf-8

# In[7]:


import re
import pandas as pd
import newspaper
from newspaper import Article
import requests
from newsapi import NewsApiClient # Needed for using API Feed
from time import time


# In[8]:


def clean_html(html):
    # First we remove inline JavaScript/CSS:
    pg = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    pg = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", pg)
    # Next we can remove the remaining tags:
    pg = re.sub(r"(?s)<.*?>", " ", pg)
    # Finally, we deal with whitespace
    pg = re.sub(r"&nbsp;", " ", pg)
    pg = re.sub(r"&rsquo;", "'", pg)
    pg = re.sub(r"&ldquo;", '"', pg)
    pg = re.sub(r"&rdquo;", '"', pg)
    pg = re.sub(r"\n", " ", pg)
    pg = re.sub(r"\t", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    return pg.strip()


# In[9]:


def newsapi_get_urls(search_words, agency_urls):
    if len(search_words)==0 or agency_urls==None:
        return None
    print("Searching agencies for pages containing:", search_words)
    # This is my API key, each user must request their own
    # API key from https://newsapi.org/account
    api = NewsApiClient(api_key='b0977f9a29784bf39ed770cb6105c5c6')
    api_urls = []
    # Iterate over agencies and search words to pull more url's
    # Limited to 1,000 requests/day - Likely to be exceeded
    for agency in agency_urls:
        domain = agency_urls[agency].replace("http://", "")
        print(agency, domain)
        for word in search_words:
            # Get articles with q= in them, Limits to 20 URLs
            try:
                articles = api.get_everything(q=word, language='en',sources=agency, domains=domain)
            except:
                print("--->Unable to pull news from:", agency, "for", word)
                continue
            # Pull the URL from these articles (limited to 20)
            d = articles['articles']
            for i in range(len(d)):
                url = d[i]['url']
                api_urls.append([agency, word, url])
    df_urls = pd.DataFrame(api_urls, columns=['agency', 'word', 'url'])
    n_total = len(df_urls)
    # Remove duplicates
    df_urls = df_urls.drop_duplicates('url')
    n_unique = len(df_urls)
    print("\nFound a total of", n_total, " URLs, of which", n_unique," were unique.")
    return df_urls


# In[10]:


def request_pages(df_urls):
    web_pages = []
    for i in range(len(df_urls)):
        u = df_urls.iloc[i]
        url = u[2]
        short_url = url[0:50]
        short_url = short_url.replace("https//", "")
        short_url = short_url.replace("http//", "")
        n = 0
        # Allow for a maximum of 5 download failures
        stop_sec=3 # Initial max wait time in seconds
        while n<3:
            try:
                r = requests.get(url, timeout=(stop_sec))
                if r.status_code == 408:
                    print("-->HTML ERROR 408", short_url)
                    raise ValueError()
                if r.status_code == 200:
                    print("Obtained: "+short_url)
                else:
                    print("-->Web page: "+short_url+" status code:",r.status_code)
                n=99
                continue # Skip this page
            except:
                n += 1
                # Timeout waiting for download
                t0 = time()
                tlapse = 0
                print("Waiting", stop_sec, "sec")
                while tlapse<stop_sec:
                    tlapse = time()-t0
        if n != 99:
            # download failed skip this page
            continue
        # Page obtained successfully
        html_page = r.text
        page_text = clean_html(html_page)
        web_pages.append([url, page_text])
    df_www = pd.DataFrame(web_pages, columns=['url', 'text'])
    n_total = len(df_urls)
    # Remove duplicates
    df_www = df_www.drop_duplicates('url')
    n_unique = len(df_urls)
    print("Found a total of", n_total, " web pages, of which", n_unique," were unique.")
    return df_www


# In[11]:


agency_urls = {
'huffington': 'http://huffingtonpost.com',
'reuters': 'http://www.reuters.com',
'cbs-news': 'http://www.cbsnews.com',
'usa-today': 'http://usatoday.com',
'cnn': 'http://cnn.com',
'npr': 'http://www.npr.org',
'wsj': 'http://wsj.com',
'fox': 'http://www.foxnews.com',
'abc': 'http://abc.com',
'abc-news': 'http://abcnews.com',
'abcgonews': 'http://abcnews.go.com',
'nyt': 'http://nytimes.com',
'washington-post': 'http://washingtonpost.com',
'us-news': 'http://www.usnews.com',
'msn': 'http://msn.com',
'pbs': 'http://www.pbs.org',
'nbc-news': 'http://www.nbcnews.com',
'enquirer': 'http://www.nationalenquirer.com',
'la-times': 'http://www.latimes.com'
}


# In[12]:


search_words = ['TAKATA']
df_urls = newsapi_get_urls(search_words, agency_urls)
print("Total Articles:", df_urls.shape[0])


# In[14]:


df_urls.head()


# In[15]:


# Download Discovered Pages
df_www = request_pages(df_urls)
# Store in Excel File
df_www.to_excel('df_www.xlsx')

