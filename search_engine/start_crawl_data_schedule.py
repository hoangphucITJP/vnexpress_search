import schedule
import time
from packages import vector_space_model as vsm
from packages.crawler import crawl

def crawl_data():
    crawl()
    vsm.build('data/*/*')
    with open('data/model_reload.req', 'w') as reload_request_file:
        reload_request_file.write('True')

crawl_data()

schedule.every(6).hours.do(crawl_data)

while True:
    schedule.run_pending()
    time.sleep(1)
