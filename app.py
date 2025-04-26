import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Bypass SSL verification if needed (important before downloads)
ssl._create_default_https_context = ssl._create_unverified_context

# Specify a custom path for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)  # <-- Fixed: No error if folder exists

# Add the custom directory to NLTK's data path
nltk.data.path.append(nltk_data_path)

# Download the 'punkt' tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Train the model
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = input_text.lower().strip()
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I couldn't understand that."

counter = 0

def main():
    global counter
    st.title("Care Bot")
    st.write("**Medicine Information Chatbot**")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("""
Welcome to **CareBot**! I'm your friendly neighborhood chatbot with a heart full of health knowledge. Have questions about your medications? I’m here to help!  

I can assist you with:  

- **Usage** – What’s this medication for?  
- **Side Effects** – What might happen?  
- **Dosage** – How much should you take?  

Just share the exact name of the medication, and I’ll do the rest.
""")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.header("About CareBot")
        st.write("""
**CareBot** provides quick and basic information about medicines.  
Built with **Natural Language Processing (NLP)** and **Logistic Regression**, it identifies user queries and responds appropriately.

**Key Features:**
- Usage of medicines
- Possible side effects
- Dosage information

**Dataset:**  
Contains labeled intents with examples for medicines like Paracetamol, Ibuprofen, Amoxicillin, Metformin, Atorvastatin, etc.

**Important:**  
CareBot is for **informational purposes only** and does not replace professional medical advice.
""")

if __name__ == '__main__':
    main()
