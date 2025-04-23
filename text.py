import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('./datasets/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

y_pred = model.predict(X_test_counts)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

counts = df['label'].value_counts()
plt.bar(['Ham', 'Spam'], counts, color=['green', 'red'])
plt.title('Distribution of Ham and Spam Messages')
plt.ylabel('Count')
plt.show()

word_freq = pd.Series(X_train_counts.toarray().sum(axis=0), index=vectorizer.get_feature_names_out())
top_words = word_freq.nlargest(20)
plt.barh(top_words.index, top_words.values)
plt.xlabel('Frequency')
plt.title('Top 20 Most Frequent Words in SMS Messages')
plt.show()

df.head()