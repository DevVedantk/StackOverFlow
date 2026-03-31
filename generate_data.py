import pandas as pd
import random

good_titles = [
    "How to fix null pointer exception in Java?",
    "Why is my Python list empty after append?",
    "How to center div using CSS flexbox?",
    "React useState not updating state",
    "Segmentation fault in C program",
    "How to connect to MySQL using Python?",
    "Index out of range error in Python",
    "How to deploy Flask app?",
    "Difference between list and tuple in Python?",
    "How to use async await in JavaScript?"
]

good_bodies = [
    "I am getting an error when running my code. Here is the code and error message.",
    "I tried multiple solutions but still getting error.",
    "Here is my code and expected output vs actual output.",
    "I read documentation but didn't understand this error.",
    "Error: NullPointerException at line 45."
]

bad_titles = [
    "Help me",
    "Urgent homework",
    "Do my assignment",
    "Code not working",
    "Please solve this",
    "I need full code",
    "ASAP help",
    "Anyone help?",
    "Homework problem",
    "Write program for me"
]

bad_bodies = [
    "Please solve this homework for me.",
    "I need full code urgently.",
    "My assignment is due tomorrow.",
    "Write full program for this question.",
    "Anyone solve this question."
]

data = []

# 500 good questions
for _ in range(500):
    data.append([
        random.choice(good_titles),
        random.choice(good_bodies),
        "programming",
        0
    ])

# 500 bad questions
for _ in range(500):
    data.append([
        random.choice(bad_titles),
        random.choice(bad_bodies),
        "homework",
        1
    ])

df = pd.DataFrame(data, columns=["title", "body", "tags", "closed"])
df.to_csv("data.csv", index=False)

print("✅ data.csv with 1000 rows created!")
print(df["closed"].value_counts())