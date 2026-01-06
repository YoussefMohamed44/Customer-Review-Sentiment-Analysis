import pickle
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
import nltk

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text cleaning function (must match training preprocessing)
def clean_review(review, stp_words):
    clean_review = " ".join(word.lower() for word in review.split() 
                          if word.lower() not in stp_words and word.isalpha())
    return clean_review

# Load the saved model
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['vectorizer'], data['classes']

# Main prediction function
def predict_sentiment():
    # Load model and vectorizer
    model, vectorizer, classes = load_model()
    stp_words = set(stopwords.words('english'))
    
    while True:
        print("\nReview Sentiment Analyzer")
        print("Type 'quit' to exit\n")
        review = input("\nEnter your review: ").strip()
        if review.lower() == 'quit':
            break
            
        if not review:
            print("Please enter a valid review\n")
            continue
            
        # Clean and vectorize the review
        cleaned_review = clean_review(review, stp_words)
        review_vec = vectorizer.transform([cleaned_review]).toarray()
        
        # Make prediction
        prediction = model.predict(review_vec)[0]
        confidence = model.predict_proba(review_vec).max()
        
        # Get class name
        sentiment = classes[prediction + 1]  # +1 because our classes are -1,0,1
        
        print(f"\nPrediction: {sentiment} (Confidence: {confidence:.2%})")
        print("------------------------------------")

if __name__ == "__main__":
    predict_sentiment()