import pandas as pd
import pickle
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# Sample data
data = pd.read_csv("data.csv")  # or create sample data

# If no data.csv exists, create sample
try:
    df = pd.read_csv("data.csv")
except:
    data = {
        'title': [
            'How to fix Python error?', 'What is closure?', 'Help me code',
            'React not rendering', 'List comprehension', 'Urgent problem',
            'Best way to learn?', 'CSS centering', 'Code crash', 'useState hook'
        ],
        'body': [
            'Error: NameError. Code: print(x)', 'Explain with examples',
            'Need homework help', 'Component shows nothing',
            'How to use list comprehension?', 'Fix broken code',
            'What resources?', 'Div not centering', 'Segmentation fault',
            'State not updating'
        ],
        'tags': ['python', 'javascript', 'all', 'react', 'python', 'urgent', 
                 'learning', 'css', 'c++', 'react'],
        'closed': [0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    }
    df = pd.DataFrame(data)

df['text'] = df['title'] + ' ' + df['body'] + ' ' + df['tags']
df['clean'] = df['text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean'])
y = df['closed']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Naive Bayes": MultinomialNB()
}

results = {}
best_acc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    if results[name]["accuracy"] > best_acc:
        best_acc = results[name]["accuracy"]
        best_model = model

# Save
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

with open("metrics.json", "w") as f:
    json.dump(results, f)

print(f"✅ Best Model: {best_model.__class__.__name__}")
print(f"✅ Accuracy: {best_acc:.2%}")
print("✅ Files saved: model.pkl, vectorizer.pkl, metrics.json")