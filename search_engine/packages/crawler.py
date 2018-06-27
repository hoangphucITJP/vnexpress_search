from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
import os

def crawl():
    BASE = os.path.dirname(os.path.abspath(__file__))
    page = requests.get('https://vnexpress.net/rss/')
    contents = page.content
    soup = BeautifulSoup(contents, 'html.parser')
    a_elements = soup.find_all('a', attrs={'class': 'rss_txt'})
    rss_links = [urljoin('https://vnexpress.net', i['href']) for i in a_elements]

    try:
        with open('data/links.txt', mode='r') as links_file:
            crawled_links = links_file.readlines()
            crawled_links = [i.strip() for i in crawled_links]
            links = set(crawled_links)
    except:
        links = set()

    rss_links.pop(0)
    queued_links = []

    for i in rss_links:
        category = i[i.rfind('/') + 1:-4]
        directory = BASE + '/data/' + category + '/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)
            
        rss_page = requests.get(i)
        rss_contents = rss_page.content
        rss_soup = BeautifulSoup(rss_contents, features='xml')
        items = rss_soup.find_all('item')
        item_links = [i.link.contents[0] for i in items if i.link is not None]
        item_links = [i for i in item_links if i not in links]
        links.update(item_links)
        item_links = [(category, i) for i in item_links]    
        queued_links += item_links
        
    for i in queued_links:
        if i[1].find('/infographics/') != -1 or i[1].find('/photo/') != -1:
            links.remove(i[1])
            continue
        
        page = requests.get(i[1])
        article_contents = page.content
        article_soup = BeautifulSoup(article_contents, 'html.parser')
        article_title = article_soup.select('h1[class*=title_news_detail]')
        
        if len(article_title) != 0:
            article_title = article_title[0].contents[0].strip()
        else:
            article_title = article_soup.select('h1[class*=title_detail]')
            if len(article_title) != 0:
                article_title = article_title[0].contents[0].strip()
            else:
                links.remove(i[1])
                continue

        data = article_soup.select('article[class*=fck_detail]')    
        if len(data) == 0: 
            links.remove(i[1])
            continue
            
        tinlienquan = data[0].select('div[class*=block_tinlienquan_temp]')
        
        if len(tinlienquan) != 0: 
            tinlienquan[0].decompose()
            
        data = data[0].get_text()
        data = data.split('\n')
        data = [i for i in data if i != '']
        data = '\n'.join(data)
        if len(data) < 420:
            links.remove(i[1])
            continue
        article_title = re.sub(r'[\\/*?:"<>|]', " ", article_title)
        with open('data/' + i[0] + '/' + article_title + '.txt', mode='w', encoding='utf-8') as text_file:
            text_file.write(data)
        print(i[1])
            
    with open('data/links.txt', mode='w') as links_file:
        links_file.write('\n'.join(links))
