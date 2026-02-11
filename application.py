import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000 
MAX_LENGTH = 500

# Model Loading
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('simple_rnn_imbd.h5')

def predict_sentiment(text, model):
    encoded = [one_hot(text, VOCAB_SIZE)]
    padded = pad_sequences(encoded, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(padded, verbose=0)
    return prediction[0][0]

st.set_page_config(page_title="IMDb RNN Predictor")
st.title("IMDb Sentiment Predictor")
st.markdown(f"**Settings:** Vocab Size: `{VOCAB_SIZE}` | Max Length: `{MAX_LENGTH}`")

model = load_my_model()

user_review = st.text_area("Enter a movie review to analyze:", height=200)

if st.button("Predict"):
    if not user_review.strip():
        st.warning("Please enter a review.")
    else:
        score = predict_sentiment(user_review, model)
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if score > 0.5:
                st.success("Verdict: POSITIVE")
            else:
                st.error("Verdict: NEGATIVE")
        
        with col2:
            st.metric(label="Confidence Score", value=f"{score:.2%}")
        
        st.progress(float(score))