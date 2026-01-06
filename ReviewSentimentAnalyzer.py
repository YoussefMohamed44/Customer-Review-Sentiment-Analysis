import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pickle


data = pd.read_csv('./Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv')
#print(data.head(),'\n')
#print(data.info(), '\n')

# To drop the null values if any
data.dropna(inplace=True)

# Change the values of sentiment for a 3-scale ((-1, 0, 1) -1 --> negative, 0 --> neutral, 1 --> positive) instead of 1-5 scale
#1,2->negative(i.e -1)
data.loc[data['Sentiment'] < 3,'Sentiment'] = -1
#3->neutral(i.e 0)
data.loc[data['Sentiment'] == 3,'Sentiment'] = 0
#4,5->positive(i.e 1)
data.loc[data['Sentiment'] > 3,'Sentiment'] = 1
#print(data.head(),'\n')

# Clean the Review column
stp_words=stopwords.words('english')
def clean_review(review): 
  clean_review=" ".join(word for word in review.
                       split() if word not in stp_words)
  return clean_review 
# update Review column after cleaning
data['Review']=data['Review'].apply(clean_review)
#print(data.head(),'\n')
#print(data['Sentiment'].value_counts())

# uses WordCloud to better visualize the importance of words
def DrawWordCloud(sentiment):
    consolidated = " ".join(word for word in data['Review'][data['Sentiment'] == sentiment].astype(str))
    wordCloud = WordCloud(width = 1600, height = 800, random_state = 21, max_font_size = 110)
    plt.figure(figsize=(15,10))
    plt.imshow(wordCloud.generate(consolidated), interpolation = 'bilinear')
    plt.axis('off')
    plt.show()

#DrawWordCloud(-1)
#DrawWordCloud(0)
#DrawWordCloud(1)

TF_IDF = TfidfVectorizer(max_features=2500)
X = TF_IDF.fit_transform(data['Review'] ).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, data['Sentiment'], test_size=0.25 , random_state=42)

model = LogisticRegression()

#Model fitting
model.fit(x_train,y_train)

#testing the model
pred = model.predict(x_test)

#model accuracy
print(accuracy_score(y_test,pred))

cm = confusion_matrix(y_test,pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Negative', 'Neutral', 'Positive'])

# Save the model and vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'vectorizer': TF_IDF,
        'classes': ['Negative', 'Neutral', 'Positive']
    }, f)

cm_display.plot()
plt.show()