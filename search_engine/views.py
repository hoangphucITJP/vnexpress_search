from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import redirect
import os.path
from prettytable import PrettyTable
import numpy as np
from bs4 import BeautifulSoup, Tag

from .packages import vector_space_model as vsm

BASE = os.path.dirname(os.path.abspath(__file__))

top_k = 5

def index(request):
    query = request.GET.get("query", None)
    if query == None:
        template = loader.get_template('search_engine/index.html')
        return HttpResponse(template.render())
    else:
        result = vsm.query(query, top_k, 50, BASE + "/model", BASE + "/data")
        
        documents = result[0]
        if len(documents) == 0:
            return HttpResponse("")
            
        documents = [doc[doc.rfind('/', 0, doc.rfind('/')) + 1:] for doc in documents]
        similarities = result[1]
        
        table = PrettyTable(float_format="6.4")
        table.add_column('Document', documents)
        table.add_column('Cosine similarity', np.round(similarities,4))
        table.sortby = 'Cosine similarity'
        table.reversesort = True
        
        html_table = table.get_html_string(attributes = {"class": "table row-border cell-hover", "id": "result-table"})
        
        soup = BeautifulSoup(html_table, 'html.parser')
        
        soup.tr.wrap(soup.new_tag('thead'))
        trs = soup.find_all('tr')
        new_th = soup.new_tag('th')
        new_th.string = '#'
        trs[0].insert(1, new_th)
        for i in range(1, len(trs)):
            link = soup.new_tag('a')
            link['href'] = 'document?filename=' + trs[i].td.string
            link['target'] = '_blank'
            trs[i].td.string.wrap(link)
            new_td = soup.new_tag('td')
            new_td.string = str(i)
            trs[i].insert(1, new_td)
            
        ths = soup.find_all('th')
        ths[0]['style'] = 'width:10%'
        ths[1]['style'] = 'width:70%'
        ths[2]['style'] = 'width:20%'
        
        return HttpResponse(soup)
        
def document(request):
    query = request.GET.get("filename", None)
    if query == None:
        return
    
    with open(BASE + '/data/' + query) as doc_file:
        doc_data = doc_file.read()
    return HttpResponse(doc_data)
    
def redirect_to_home():
    return redirect('index')
