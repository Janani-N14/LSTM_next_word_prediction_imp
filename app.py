import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
try:
    model = load_model('next_word_lstm.h5')
    # Load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model_loaded = True
except (FileNotFoundError, IOError) as e:
    model_loaded = False

def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Truncate if necessary
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    
    # Pad sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Get prediction
    predictions = model.predict(token_list, verbose=0)[0]
    
    # Get top 3 predictions
    predicted_indices = np.argsort(predictions)[-3:][::-1]
    predicted_words = []
    
    for index in predicted_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                predicted_words.append((word, float(predictions[index])))
                break
    
    return predicted_words

# Streamlit app
st.title('Next Word Prediction with LSTM')

if not model_loaded:
    st.error("Model or tokenizer file not found. Please check file paths.")
else:
    input_text = st.text_input("Enter sequence of words:", "")
    
    if st.button("Predict Next Word"):
        if input_text:
            # Get the correct max sequence length from model
            max_sequence_len = model.input_shape[1] + 1
            
            # Predict next words
            next_words = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            
            if next_words:
                st.write("Top predictions:")
                for word, prob in next_words:
                    st.write(f"- '{word}' (confidence: {prob:.2f})")
                
                # Show complete sentence with top prediction
                st.write("---")
                st.write(f"Complete sentence: {input_text} **{next_words[0][0]}**")
            else:
                st.warning("Could not predict next word. Try different input.")
        else:
            st.warning("Please enter some text.")