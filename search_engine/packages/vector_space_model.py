from glob import glob
import re
import os
import numpy as np
import gzip
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

model_path = ''
data_path = ''
alpha = 0.4
max_df = 1
min_df = 0

BASE = os.path.dirname(os.path.abspath(__file__))

def tokenize(input_string):
    with open('tmp', mode='w') as tmp_file:
        tmp_file.write(input_string)
        
    subprocess.run(['java', '-jar', BASE + '/vitk-tok-5.1.jar', 'tmp', 'tmp'])
    with open('tmp', mode='r') as tmp_file:
        data = tmp_file.read()
    
    replaced = re.sub(r'/\w+', '$', data)
    result = replaced.split('$')
    result = [i.strip() for i in result if i.strip() != '']
    return result
    
def preprocess(text):
    text = re.sub(r'[~_{}+|=\-\"&$,?!''><()*/\[\]^#%@.;“”:]', " ", str(text))
    text = re.sub(r'[`]', " ", text)
    text = re.sub(r'[\\]', " ", text)
    text = re.sub(r"[\']", " ", text)
    text = text.lower()
    return text

def preprocess_text(query):
    q = query.lower()
    q = tokenize(q)
    q = remove_stop_words(q)
    q = remove_number_contained(q)
    q = ' '.join(q)
    print(q)
    return q
    

def remove_stop_words(wordList):
    with open(model_path + '/vietnamese-stopwords.txt', mode='r') as stop_word_file:
        stopwords = stop_word_file.readlines()

    stopwords = [word.strip('\n') for word in stopwords]
    nonStopWordText = [word for word in wordList if (
        not word in stopwords) and word[-2:] != "'s"]
    return nonStopWordText

def remove_number_contained(wordList):
    nonNumber = [item for item in wordList if not any(map(lambda c:c.isdigit(),item))]
    return nonNumber


def update_vocab(text, Vocab):
    splitted_text = tokenize(text)
    nonStopWordText = remove_stop_words(splitted_text)
    numberRemoved = remove_number_contained(nonStopWordText)
    preprocessed_data = ' '.join(numberRemoved)
    doc_words = set(numberRemoved)
    Vocab.update(doc_words)
    return preprocessed_data


def calculate_tf(docs, vocab):
    counter = CountVectorizer(vocabulary=vocab)
    doc_tf = counter.fit_transform(docs).toarray()

    return doc_tf


def calculate_ntf(doc_tf):

    max_tf = np.max(doc_tf, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        #doc_ntf = alpha + (1 - alpha)*doc_tf/max_tf[:, None]
        doc_ntf = doc_tf/max_tf[:, None]

    return doc_ntf


def calculate_idf(doc_tf):
    term_idf = 1 + np.log(doc_tf.shape[0]/(np.count_nonzero(doc_tf, axis=0)))

    return term_idf


def calculate_tf_idf(docs, vocab):
    doc_tf = calculate_tf(docs, vocab)
    doc_ntf = calculate_ntf(doc_tf)
    
    term_idf = calculate_idf(doc_tf)
    tf_idf = np.multiply(doc_ntf, term_idf)

    return tf_idf, term_idf


def generate_vocabulary(data):
    vocabulary = set()
    numOfFiles = len(data)
    preprocessed_data = []
    for k, i in enumerate(data):
        print(str(int(k / numOfFiles * 100)) + '%', end='\r')
        text = preprocess(i)
        preprocessed_data += [update_vocab(text, vocabulary)]

    vocabulary = list(vocabulary)

    # Select vocabulary   
    counter = CountVectorizer(vocabulary=vocabulary)
    doc_tf = counter.fit_transform(preprocessed_data).toarray()
    df = np.count_nonzero(doc_tf, axis=0)
    num_of_doc = doc_tf.shape[0]
    selection = np.where((df/num_of_doc < max_df) & (df/num_of_doc > min_df))[0]
    selected_term = [vocabulary[ind] for ind in selection]
    
    # Save vocabulary file
    with open(model_path + "/vocab_vsm.txt", "w") as vocabFile:
        vocabFile.writelines(i + "\n" for i in selected_term)

    return selected_term, preprocessed_data


def query_to_tfidf(query, idf, vocab):
    lowered_query = query.lower()
    query_tf = calculate_tf([lowered_query], vocab)
    query_ntf = calculate_ntf(query_tf)
    query_tfidf = np.multiply(query_ntf, idf)

    return query_tfidf

def build(dataPattern):
    global model_path, data_path
    model_path = 'model'
    data_path = 'data'
    print('Building model...')
    query.dataFile = glob(dataPattern)
    
    print('Loading dataset...')
    data = [open(f).read() for f in query.dataFile]
    print('Generating vocabulary...')
    query.vocabulary, preprocessed_data = generate_vocabulary(data)
    print('Calculating tf-idf...')
    query.tf_idf, query.idf = calculate_tf_idf(preprocessed_data, query.vocabulary)

    with gzip.GzipFile(model_path + "/tf_idf.npy.gz", "w") as tf_idf_zip:
        np.save(tf_idf_zip, query.tf_idf)

    with gzip.GzipFile(model_path + "/idf.npy.gz", "w") as idf_zip:
        np.save(idf_zip, query.idf)
        
    print('Complete building model...')
    
    

def query(arg, top_k, threshold, pmodel_path, data_path, cosine_threshold=0):
    global model_path
    model_path = pmodel_path
    
    with open(BASE + '/../data/model_reload.req', 'r') as reload_request_file:
        reload_request = reload_request_file.read()
    if not hasattr(query, 'tf_idf') or reload_request=='True':
        print('Loading model...')
        dataPattern = data_path + '/*/*'
        query.dataFile = glob(dataPattern)
        with open(model_path + "/vocab_vsm.txt", "r") as vocabFile: # Use custom's vocabulary
            vocabulary = vocabFile.readlines()
        query.vocabulary = [word.strip() for word in vocabulary]
        print('Loading tf-idf...')
        with gzip.GzipFile(model_path + "/tf_idf.npy.gz", "r") as tf_idf_zip:
            query.tf_idf = np.load(tf_idf_zip)

        with gzip.GzipFile(model_path + "/idf.npy.gz", "r") as idf_zip:
            query.idf = np.load(idf_zip)
            
        print('Complete loading model...')
        with open(BASE + '/../data/model_reload.req', 'w') as reload_request_file:
            reload_request_file.write('False')

    preprocessed_query = preprocess_text(arg)
    query_tfidf = query_to_tfidf(preprocessed_query, query.idf, query.vocabulary)

    # Cosine distance
    cosine_distance = cos_sim(query.tf_idf, query_tfidf.T)
    
    # Euclidean distance
    #euclid_distance = np.sqrt(np.sum(np.square((query.tf_idf - query_tfidf)), axis=1))

    # Used distance
    distance = cosine_distance
    distance = np.nan_to_num(distance)
    # Top k
    #ranking = np.argsort(distance)[:top_k]
    #top_k_distance = [distance[index] for index in ranking]
    #top_k_result = [query.dataFile[index] for index in ranking]
    
    # Thresholding
    # Euclidean thresholding
    #ranking = np.where(distance < threshold)[0].tolist()
    # Cosine thresholding
    if cosine_threshold == None:
        #ranking = np.argsort(distance)[:top_k]
        #top_k_distance = [distance[index] for index in ranking]
        #top_k_result = [query.dataFile[index] for index in ranking]
        top_distance = distance
        top_result = query.dataFile
    else:
        # Thresholding
        ranking = np.where(distance > cosine_threshold)[0].tolist()
        
        top_distance = np.array([distance[index] for index in ranking])
        top_result = [query.dataFile[index] for index in ranking]
        
        # Sorting
        ranking = np.argsort(top_distance)[::-1]
        top_distance = [top_distance[index] for index in ranking]
        top_result = [top_result[index] for index in ranking]

    return top_result, top_distance


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    if not hasattr(cos_sim, 'norm_a'):
         cos_sim.norm_a = np.linalg.norm(a, axis=1)
         
    norm_b = np.linalg.norm(b)
    norm_a_b_product = cos_sim.norm_a*norm_b
    distance = dot_product.reshape((dot_product.shape[0])) / (norm_a_b_product)
    return distance
    
def cos_sim_short(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_a_b_product = norm_a
    distance = dot_product.reshape((dot_product.shape[0])) / (norm_a_b_product)
    return distance 
