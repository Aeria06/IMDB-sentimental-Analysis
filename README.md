🎬 IMDB Movie Review Sentiment Analysis using RNN

This project is a Deep Learning–based Sentiment Analysis system that classifies IMDB movie reviews as Positive or Negative using a Recurrent Neural Network (RNN).
The trained model is deployed using a Streamlit web application for real-time user interaction.
<img width="1084" height="813" alt="image" src="https://github.com/user-attachments/assets/fd66c9f6-abf3-4285-a206-3a4badde2506" />
<img width="1044" height="836" alt="image" src="https://github.com/user-attachments/assets/f505b888-e8ad-4467-b0e9-a77550649493" />

🚀 Project Overview

Understanding audience sentiment is crucial in the entertainment industry. This project leverages Natural Language Processing (NLP) and Recurrent Neural Networks to analyze textual movie reviews and predict their sentiment.

The model is trained on the IMDB Movie Reviews dataset, learning sequential patterns in text to make accurate sentiment predictions.

🧠 Model Architecture

Embedding Layer – Converts words into dense vector representations

Simple RNN Layer – Captures sequential dependencies in text

Dense Output Layer – Outputs sentiment probability

Activation Function – Sigmoid (binary classification)

🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy & Pandas

Streamlit (Web UI)

IMDB Dataset (Keras built-in)

✨ Features

Classifies movie reviews as Positive or Negative

Displays confidence score for predictions

Clean, professional Streamlit UI

Real-time sentiment analysis

Lightweight and easy to deploy

📂 Project Structure
simple_rnn_imdb/
│
├── main.py                  # Streamlit application
├── simple_rnn_imdb.h5       # Trained RNN model
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
└── venv/                    # Virtual environment (not pushed to GitHub)

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/simple_rnn_imdb.git
cd simple_rnn_imdb

2️⃣ Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run main.py

🧪 Sample Input
The movie had brilliant performances and a powerful storyline. I really enjoyed it.

Output
Sentiment: Positive
Confidence Score: 0.87

📊 Dataset

IMDB Movie Reviews Dataset

50,000 labeled reviews

Binary sentiment classification (positive / negative)

Loaded using tensorflow.keras.datasets.imdb

⚠️ Notes on Compatibility

This project uses TensorFlow 2.12 + Keras 2.12 for compatibility with legacy RNN models

Newer versions of Keras (3.x) may cause loading issues with .h5 models

🎯 Future Enhancements

Replace Simple RNN with LSTM / GRU

Add Explainable AI (LIME / SHAP)

Deploy on Streamlit Cloud

Support batch review analysis

Improve preprocessing with stemming & lemmatization

👩‍💻 Author

Hitanshi Arora
Web Developer & IT Student, VIT Vellore
