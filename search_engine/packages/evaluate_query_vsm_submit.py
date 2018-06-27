import vector_space_model as vsm
import numpy as np


building = input('Build/rebuild model (yes/no)?')
if building == 'yes':
    vsm.build('data/Cranfield/*')    

with open('data/query.txt', mode='r') as query_file:
    queries = query_file.readlines()

queries = [query.strip() for query in queries]
queries = [[query.split()[0], ' '.join(query.split()[1:])] for query in queries]

top_k = 5
num_of_queries = len(queries)
result = []




print('Querying...')
for query_iter in range(num_of_queries):
    result.append(vsm.query(queries[query_iter][1], top_k, 50, 'model', 'data', 0))

result = np.array(result).T
docs = result[0].tolist()
docs = [[doc[doc.rfind('/') + 1: -4] + '\n' for doc in q] for q in docs]
distance = result[1].tolist()

for i in range(1, 226):
    with open('data/query_results/' + str(i) + '.txt', mode='w') as result_file:
        docs[i - 1][len(docs[i - 1]) - 1] = docs[i - 1][len(docs[i - 1]) - 1].strip()
        result_file.writelines(docs[i - 1])
        
np.save('model/vsm_result.npy', result)
