import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() # Lowercase
    text = nltk.word_tokenize(text) # Tokenization
    
    y = []
    for x in text: # Remove special characters
        if x.isalnum():
            y.append(x)
            
    text = y[:] # cloning
    y.clear()
    
    for x in text: # Remove stopword 
        if x not in stopwords.words('english') and x not in string.punctuation:
            y.append(x)
            
    text = y[:]
    y.clear()
    
    for x in text: # Remove Punctuation
        y.append(ps.stem(x))
            
    return " ".join(y)

# Load model and vectorizer
tfv = pickle.load(open('vectorizer.pkl', 'rb')) # rb stands for read binary mode
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .stTextInput>div>div>div>input {
            border-radius: 10px;
            border: 2px solid #3498db;
            padding: 10px;
        }
        .stTextInput>div>div>div>input:focus {
            border-color: #2980b9;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            margin-top: 10px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .stSuccess {
            background-color: #27ae60;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
        .stError {
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title('SMS SPAM DETECTION')

# Text input
input_sms = st.text_area('Enter the message', height=150)

# Prediction button
if st.button('Predict'):
    # Preprocessing
    transformed_text = transform_text(input_sms)

    # Vectorization
    vector_input = tfv.transform([transformed_text])

    # Prediction
    result = model.predict(vector_input)[0]

    # Display prediction
    if result == 1:
        st.markdown('<div class="stError">Spam Message &#128680;</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stSuccess">Not Spam Message &#128522;</div>', unsafe_allow_html=True)