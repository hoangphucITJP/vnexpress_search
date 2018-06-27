import numpy as np
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer

def check_relevance(query, doc):
    doc = doc[doc.rfind('/') + 1:doc.rfind('.')]
    with open('data/TEST/RES/' + query + '.txt') as query_result_file:
        data = query_result_file.readlines()
    data = [line.split()[1] for line in data]
    try:
        ind = data.index(doc)
        return True
    except:
        return False
    
def get_all_relates(query):
    with open('data/TEST/RES/' + query + '.txt') as query_result_file:
        data = query_result_file.readlines()
    
    return len(data)
    
def load_queries():   
    with open('data/TEST/query.txt', mode='r') as query_file:
        queries = query_file.readlines()
        
    queries = [query.strip() for query in queries]
    queries = [[query.split()[0], ' '.join(query.split()[1:])] for query in queries]
    return queries

def calculate_precision_recall_by_top_k(query_index, results, queries, k):    
    query_content = queries[query_index][1]
    print('Query:' + query_content)
    
    docs = results[query_index][0]
    distance = np.array(results[query_index][1]).astype(np.float)    
    index = np.argsort(distance)[::-1][:k]
    
    docs = [docs[ind] for ind in index]
    distance = [distance[ind] for ind in index]
    
    rel = [check_relevance(str(query_index + 1), doc) for doc in docs]
    total = len(rel)
    true = rel.count(True)
    
    positives = get_all_relates(str(query_index + 1))
    
    precision = true/total
    recall = true/positives
    
    return [precision, recall]
    
def calculate_precision_recall_by_top_k_all_query(results, queries, k):   
    true = positives = total = 0
    for query_index in range(len(queries)):    
        docs = results[query_index][0]
        distance = np.array(results[query_index][1]).astype(np.float)    
        index = np.argsort(distance)[::-1][:k]
        
        docs = [docs[ind] for ind in index]
        distance = [distance[ind] for ind in index]
        
        rel = [check_relevance(str(query_index + 1), doc) for doc in docs]
        total += len(rel)
        true += rel.count(True)
        
        positives += get_all_relates(str(query_index + 1))
        
    precision = true/total
    recall = true/positives
    
    return [precision, recall]
    
def calculate_precision_recall_by_top_ks_all_query(results, queries, ks):
    precision_recall_pairs = [calculate_precision_recall_by_top_k_all_query(results, queries, k) for k in ks]
    
    print(precision_recall_pairs)
    np.save('precision_recall_of_top_10_20_30.npy', precision_recall_pairs)
    


def calculate_precision_recall_by_top_percents_all_query(results, queries, percents):
    precision_all_queries = []
    recall_all_queries = []
    AP_all_queries = []
    num_of_queries = len(queries)
    for query_index in range(num_of_queries):
        print(str(int(query_index + 1 / num_of_queries * 100)) + '%', end='\r')
        docs = results[query_index][0]
        distance = np.array(results[query_index][1]).astype(np.float)
        distance = np.nan_to_num(distance)
        index = np.argsort(distance)[::-1]            
        docs = [docs[ind] for ind in index]    
        distance = [distance[ind] for ind in index]    
        rel = [check_relevance(str(query_index + 1), doc) for doc in docs]
        
        total = 0
        true_positive = 0
        positives = get_all_relates(str(query_index + 1))
        precisions = [None]*len(percents)
        precision_sum = 0
        recalls = [None]*len(percents)
        percent_index = 0
        while percent_index < len(percents):
            if total == len(docs):
                #while percent_index < len(percents):
                #    recalls[percent_index] = percents[percent_index]/100
                #    print('here')
                #    precisions[percent_index] = 0
                #    percent_index += 1
                break
                
            total += 1
            if rel[total - 1] == True:
                true_positive += 1
        
                precision = true_positive/total
                recall = true_positive/positives
                precision_sum += precision
                
                if recall*100 > percents[percent_index]:
                    while recall*100 - 10 >= percents[percent_index]:   
                        percent_index += 1             
                                
                    precisions[percent_index] = precision
                    recalls[percent_index] = recall                    
        
        AP = precision_sum/positives
        AP_all_queries.append(AP)
        precision_all_queries.append(precisions)
        recall_all_queries.append(recalls)
        
    precision_all_queries = np.array(precision_all_queries, dtype=np.float)
    recall_all_queries = np.array(recall_all_queries, dtype=np.float)
    
    precision_all_queries = np.nanmean(precision_all_queries, axis=0)
    recall_all_queries = np.nanmean(recall_all_queries, axis=0)
    mAP = np.mean(AP_all_queries)
    np.save('model/precision_by_percent_recall.npy', [recall_all_queries, precision_all_queries, AP_all_queries, mAP])


queries = load_queries()
results = np.load('model/vsm_result.npy').tolist()
#calculate_precision_recall_by_similarity_threshold(results, queries, 0.5)   #Threshold: 0.0 < cosine similarity < 0.5

# Calculate precision, recall with top 10, 20, 30 similarity
#ks = [10, 20, 30]
#calculate_precision_recall_by_top_ks_all_query(results, queries, ks)

# Calculate average precision
recalls = list(range(0, 100 + 1, 10))
calculate_precision_recall_by_top_percents_all_query(results, queries, recalls)
