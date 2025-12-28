# # Step 1: Import Libraries and Load the Model
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model

# # Load the IMDB dataset word index
# word_index = imdb.get_word_index()
# reverse_word_index = {value: key for key, value in word_index.items()}

# # Load the pre-trained model with ReLU activation
# model = load_model('simple_rnn_imdb.h5')

# # Step 2: Helper Functions
# # Function to decode reviews
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# # Function to preprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
#     return padded_review


# import streamlit as st
# ## streamlit app
# # Streamlit app
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# # User input
# user_input = st.text_area('Movie Review')

# if st.button('Classify'):

#     preprocessed_input=preprocess_text(user_input)

#     ## MAke prediction
#     prediction=model.predict(preprocessed_input)
#     sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

#     # Display the result
#     st.write(f'Sentiment: {sentiment}')
#     st.write(f'Prediction Score: {prediction[0][0]}')
# else:
#     st.write('Please enter a movie review.')

# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main {
        background-color: #f9fafb;
    }
    h1 {
        text-align: center;
        color: #1f2937;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
    }
    .positive {
        background-color: #dcfce7;
        color: #166534;
    }
    .negative {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model & Word Index ----------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

# ---------------- Helper Functions ----------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---------------- UI ----------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown(
    "<div class='subtitle'>Analyze movie reviews using a Deep Learning RNN model</div>",
    unsafe_allow_html=True
)

st.write("✍️ **Enter a movie review below:**")
user_input = st.text_area(
    "",
    placeholder="Example: The movie had brilliant acting but a weak storyline...",
    height=150
)

# ---------------- Prediction ----------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if user_input.strip() == "":
        st.warning("Please enter a movie review to analyze.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)

        score = prediction[0][0]
        sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        if score > 0.5:
            st.markdown(
                f"<div class='result-box positive'>Sentiment: {sentiment}<br>Confidence Score: {score:.2f}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box negative'>Sentiment: {sentiment}<br>Confidence Score: {score:.2f}</div>",
                unsafe_allow_html=True
            )

        # Confidence bar
        st.progress(float(score))

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
        "<center>Built using TensorFlow & Streamlit</center>",
    unsafe_allow_html=True
)
