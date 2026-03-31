import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re

# Sample data
data = {
    'title': [
        'How to fix Python error?',
        'What is JavaScript closure?',
        'Help me code',
        'Why is React component not rendering?',
        'Python list comprehension',
        'Urgent code problem',
        'Best way to learn programming?',
        'CSS flexbox centering issue',
        'Why does my code crash?',
        'How to use useState hook?'
    ],
    'body': [
        'I get AttributeError. Code: print(x)',
        'Explain closures with examples',
        'Need help with homework',
        'Component shows nothing',
        'How to use list comprehension?',
        'Fix this broken code',
        'What resources to learn coding?',
        'Div not centering with flexbox',
        'Getting segmentation fault',
        'useState not updating state'
    ],
    'closed': [0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['text'] = df['title'] + ' ' + df['body']
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['closed']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
print(f"Accuracy: {model.score(X, y):.2%}")