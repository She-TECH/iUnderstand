import pandas as pd
import nltk
import json

# read data
reviews_df = pd.read_csv("D:/SGES.csv")

reviews_df = reviews_df.sample(frac = 0.1, replace = False, random_state=42)

# remove 'No Negative' or 'No Positive' from text
reviews_df["comments"] = reviews_df["comments"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))

# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')

# clean text data
reviews_df["review_clean"] = reviews_df["comments"].apply(lambda x: clean_text(x))

# add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
reviews_df["sentiments"] = reviews_df["comments"].apply(lambda x: sid.polarity_scores(x))

reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)

# add number of characters column
reviews_df["nb_chars"] = reviews_df["comments"].apply(lambda x: len(x))

# add number of words column
reviews_df["nb_words"] = reviews_df["comments"].apply(lambda x: len(x.split(" ")))

# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)

# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 0, max_df = 10)
tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)

reviews_df.head()

posCount = 0
negCount = 0
neuCount = 0
for index, row in reviews_df.iterrows():
    print(row['pos'], row['neg'], row['neu'])
    if row['pos'] > 0.5:
        posCount = posCount + 1
    elif row['neg'] > 0.5:
        negCount = negCount + 1
    elif row['neu'] > 0.5:
        neuCount = neuCount + 1        

print("pos", posCount)
print("neg", negCount)
print("neu", neuCount)

data_set = {"Positive": posCount, "Negative": negCount, "Neutral": neuCount}

json_dump = json.dumps(data_set)

print(json_dump)
