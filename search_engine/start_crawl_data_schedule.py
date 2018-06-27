import schedule
import time
from packages import vector_space_model as vsm
from packages.crawler import crawl

def crawl_data():
    crawl()
    vsm.build('data/*/*')

schedule.every(6).hours.do(crawl_data)

while True:
    schedule.run_pending()
    time.sleep(1)
