from flask import Flask, render_template, jsonify, request
import pandas as pd
import re
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
import zipfile

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load and preprocess data from ZIP file
with zipfile.ZipFile("News.zip", "r") as z:
    with z.open("News.csv") as f:
        news_data = pd.read_csv(f, encoding="latin1", index_col=0)

news_data = news_data.drop(["title", "subject", "date"], axis=1)
news_data = news_data.sample(frac=1).reset_index(drop=True)

def preprocess_text(text_data):
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    
    for sentence in tqdm(text_data):
        # Remove punctuation and lowercase tokens (skipping stopwords)
        sentence = re.sub(r'[^\w\s]', '', str(sentence))
        preprocessed_text.append(' '.join(token.lower() 
                                  for token in nltk.word_tokenize(sentence)
                                  if token.lower() not in stop_words))
    return preprocessed_text

news_data['processed_text'] = preprocess_text(news_data['text'].values)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_random_article')
def get_random_article():
    random_article = news_data.sample(1).iloc[0]
    return jsonify({
        'text': random_article['text'],
        'actual_class': 'Real' if random_article['class'] == 1 else 'Fake',
        'index': int(random_article.name)  # Ensure it's JSON serializable
    })

@app.route('/check_guess', methods=['POST'])
def check_guess():
    user_input = request.get_json()  # Avoid conflict with `news_data`
    actual_class = 'Real' if news_data.loc[user_input['index'], 'class'] == 1 else 'Fake'
    return jsonify({
        'correct': user_input['guess'] == actual_class,
        'actual_class': actual_class
    })

if __name__ == '__main__':
    app.run(debug=True)
