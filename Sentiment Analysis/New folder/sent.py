import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

######################################################################################################
######################################################################################################
######################################################################################################

# Tokenize the text using nltk's tokenizer
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)


#This function pre-processes things to remove noisy words such as 'it'
#takes a string
def myTokenizer(s):
    s = s.lower() # downcase so capitals don't differintiate
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordNetLemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens

######################################################################################################
######################################################################################################
######################################################################################################

# Create input matrices
def tokensToVector(tokens, label):
    x = np.zeros(len(wordToIndexMap) + 1) # last element is for the label
    for t in tokens:
        i = wordToIndexMap[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

######################################################################################################
######################################################################################################
######################################################################################################



wordNetLemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html     this are some words we want to ignore
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# Load the reviews

# This paper below is used as a basis when using the .reveiw files
    # John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood,
    # Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.
    # Association of Computational Linguistics (ACL), 2007
# http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
# Our sentiment analysis program relies on the research they did

posReviews = BeautifulSoup(open('positive.review').read())
posReviews = posReviews.findAll('review_text')

negReviews = BeautifulSoup(open('negative.review').read())
negReviews = negReviews.findAll('review_text')

#np.random.shuffle(posReviews)
#posReviews = posReviews[:len(negReviews)]


# Word-to-index map so that we can make our word-frequency vectors later
# Save the tokenized versions so we don't have to tokenize again later
wordToIndexMap = {}
currentIdx = 0
positiveTokenizedArray = []
negativeTokenizedArray = []

for review in posReviews:
    tokens = myTokenizer(review.text)
    positiveTokenizedArray.append(tokens)
    for token in tokens:
        if token not in wordToIndexMap:
            wordToIndexMap[token] = currentIdx
            currentIdx += 1

for review in negReviews:
    tokens = myTokenizer(review.text)
    negativeTokenizedArray.append(tokens)
    for token in tokens:
        if token not in wordToIndexMap:
            wordToIndexMap[token] = currentIdx
            currentIdx += 1


N = len(positiveTokenizedArray) + len(negativeTokenizedArray)

data = np.zeros((N, len(wordToIndexMap) + 1))

i = 0 #index on data

for tokens in positiveTokenizedArray:
    #print(tokens)
    xy = tokensToVector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negativeTokenizedArray:
    xy = tokensToVector(tokens, 0)
    data[i,:] = xy
    i += 1

# shuffle the data and create train/test splits
# try it multiple times!
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
#print "Classification rate:", model.score(Xtest, Ytest)


# look at the weights for each word
threshold = 0.5 #can be modified
for word, index in wordToIndexMap.iteritems():
    weight = model.coef_[0][index]
    #if weight > threshold or weight < -threshold:
    #print(index, word, weight)


print("##########################################################################################\n\n")

line = "This is such a nice place and the waitress was pretty"
line = line.split(' ')
sentNum = 0
for word in line:
    aWordIdx = wordToIndexMap.get(word)
    if(aWordIdx):
        #print(word)
        #print(aWordIdx)
        #print(model.coef_[0][aWordIdx])
        sentNum += model.coef_[0][aWordIdx]
        print("________________________")

print("SentNum:  ")
print(sentNum)