import pandas as pd
import matplotlib.pyplot as plt

# LOAD 
df = pd.read_csv("twitter_training.csv", encoding="latin-1")

# Fix column
df.columns = ["id", "entity", "sentiment", "text"]
df = df[["text", "sentiment"]]

#  CLEAN 
df = df.dropna()

df["sentiment"] = df["sentiment"].str.lower()
df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "neutral": 1,
    "positive": 2
})

df = df.dropna()

print(df["sentiment"].value_counts())

# BALANCE 
df_neg = df[df["sentiment"] == 0].sample(1000, random_state=42)
df_neu = df[df["sentiment"] == 1].sample(1000, random_state=42)
df_pos = df[df["sentiment"] == 2].sample(1000, random_state=42)

df = pd.concat([df_neg, df_neu, df_pos])
df = df.sample(frac=1, random_state=42)

print("Dataset size:", len(df))

# VECTORIZE 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),   
    max_features=5000
)

X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# TRAIN 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# PREDICT 
y_pred = model.predict(X_test)

# BAR CHART 
counts = pd.Series(y_pred).value_counts().sort_index()

labels = ["Negative", "Neutral", "Positive"]
values = counts.reindex([0,1,2], fill_value=0).values

plt.style.use("ggplot")

colors = ["#e74c3c", "#95a5a6", "#2ecc71"]

plt.figure()
bars = plt.bar(labels, values, color=colors)

plt.title("Sentiment Distribution (Prediction)")
plt.xlabel("Sentiment")
plt.ylabel("Count")

# Show values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')

plt.show()

# TEST 
test_text = ["I think this product is okay"]
test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)

mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

print("\nTest:", test_text[0])
print("Prediction:", mapping[prediction[0]])

#ACCURACY
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)