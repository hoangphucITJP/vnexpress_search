import vector_space_model as vsm
import numpy as np


building = input('Build/rebuild model (yes/no)?')
if building == 'yes':
    vsm.build('data/Cranfield/*')    

with open('data/TEST/query.txt', mode='r') as query_file:
    queries = query_file.readlines()

queries = [query.strip() for query in queries]
queries = [[query.split()[0], ' '.join(query.split()[1:])] for query in queries]

top_k = 5
num_of_queries = len(queries)
result = []




print('Querying...')
for query_iter in range(num_of_queries):
    result.append(vsm.query(queries[query_iter][1], top_k, 50, 'model', 'data', None))
    
    print(str(int(query_iter/num_of_queries*100)) + '%', end='\r')

np.save('model/vsm_result.npy', result)
