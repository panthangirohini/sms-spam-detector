
import pandas as pd


url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])



print("Dataset shape:", df.shape)
print(df.head())


df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


print("\nMissing values:\n", df.isnull().sum())


df['message'] = df['message'].str.lower()


from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label_num']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


def predict_message(msg):
    msg = msg.lower()
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    return "Spam" if prediction == 1 else "Ham"

print("\n--- Try Your Own SMS ---")
user_message = input("Enter an SMS message: ")
result = predict_message(user_message)
print(f"Prediction: This message is '{result}'")
