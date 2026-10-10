

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', names=['label', 'message'])
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].str.lower()
    return df


@st.cache_resource
def train_model(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return vectorizer, model, y_test, y_pred

df = load_data()
vectorizer, model, y_test, y_pred = train_model(df)




with st.sidebar:
    st.header("📊 Dataset")
    st.metric("Total messages", len(df))
    st.metric("Spam", int((df['label_num'] == 1).sum()))
    st.metric("Ham", int((df['label_num'] == 0).sum()))
    st.caption("Source: UCI SMS Spam Collection")

st.title("📩 SMS Spam Detection")
st.caption("Multinomial Naive Bayes · Bag-of-Words feature extraction")

tab1, tab2 = st.tabs(["🔍 Try a Message", "📈 Model Performance"])


with tab1:
    st.subheader("Enter an SMS message")
    user_input = st.text_area(
        "Message",
        placeholder="e.g. Congratulations! You've won a free prize, claim now!",
        height=100,
    )

    if st.button("Predict", type="primary"):
        if not user_input.strip():
            st.warning("Please type a message first.")
        else:
            vec = vectorizer.transform([user_input.lower()])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            if pred == 1:
                st.error(f"🚨 **SPAM**  —  confidence {proba[1]*100:.1f}%")
            else:
                st.success(f"✅ **HAM** (legitimate)  —  confidence {proba[0]*100:.1f}%")

    st.divider()
    st.caption("Quick examples to try:")
    col1, col2 = st.columns(2)
    col1.code("WINNER!! You have been selected to receive a $900 prize reward!")
    col2.code("Hey, are we still meeting for lunch tomorrow?")


with tab2:
    acc = accuracy_score(y_test, y_pred)
    st.metric("Test Accuracy", f"{acc*100:.2f}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Ham', 'Spam'])
    ax.set_yticklabels(['Ham', 'Spam'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3))
