from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob
import numpy as np

wordsPerQuery = 3
queriesPerCategory = 5
with open("vocab.txt","r") as vocabFile:
	vocab = vocabFile.read().splitlines()

folderPattern = '20_newsgroups/*/'
folders = glob(folderPattern)

fullData = []
categTermFre = None

# Calculate term frequency in each categories
categCount = 0
fullData = []
for folder in folders:
	categCount += 1
	print('Category ' + str(categCount) + '/20: ' + folder)
	dataPattern = folder + '*'
	dataFile = glob(dataPattern)
	data = [open(f, encoding="latin-1").read() for f in dataFile]
	data = '\n'.join(data)

	fullData.append(data)



vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=0.3)
vectorizer.fit(fullData)
vocabulary = list(vectorizer.vocabulary_.keys())
np.save('vocab.npy', vocabulary)
