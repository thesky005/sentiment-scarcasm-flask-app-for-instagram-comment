from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import emoji
import instaloader
from collections import defaultdict

app = Flask(__name__)

# Define emoji mappings
positive_emojis = ["😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😊", "😇", "🙂", "😉", "😌", "😍", "🥰", "😘", "😗", "😙", "😚", "😋", "😛", "😝", "😜", "🤪", "😎", "🤩", "🥳", "😏", "😬", "🤗"]
negative_emojis = ["😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "🥺", "😢", "😭", "😤", "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱", "😨", "😰", "😥", "😓", "😈", "👿", "👹", "👺", "💩", "😿"]
neutral_emojis = ["🥹", "🥲", "☺️", "😐", "😑", "😶", "🙃", "😶‍🌫", "🤔", "🫣", "🤭", "🫡", "🫢", "🫡", "🤫", "🫠", "🤥", "😶", "🫥", "😐", "🫤", "😑", "🫨", "🙄", "😯", "😦", "😧", "😮", "😲", "🥱", "😴", "🤤", "😪", "😵", "🤐", "🥴", "🤢", "🤧", "😷", "🤒", "🤕", "🤑", "🤠"]

# Load components
def load_components(tokenizer_path='tokenizer.pkl', model_path='sentiment_sarcasm_model.h5'):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model(model_path)
    return tokenizer, model

# Define emoji and punctuation sentiment analysis function
def analyze_emoji_punctuation(text):
    emojis = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    punctuations = ''.join(c for c in text if c in string.punctuation)
    
    # Analyze emojis
    emoji_counts = {
        'positive_count': sum(1 for e in emojis if e in positive_emojis),
        'negative_count': sum(1 for e in emojis if e in negative_emojis),
        'neutral_count': sum(1 for e in emojis if e in neutral_emojis)
    }
    
    if emoji_counts['positive_count'] > emoji_counts['negative_count']:
        emoji_sentiment = 'Positive'
    elif emoji_counts['negative_count'] > emoji_counts['positive_count']:
        emoji_sentiment = 'Negative'
    else:
        emoji_sentiment = 'Neutral'
    
    # Analyze punctuation
    exclamation_count = punctuations.count('!')
    question_count = punctuations.count('?')
    ellipsis_count = punctuations.count('...')
    
    if exclamation_count >= 3:
        punctuation_sentiment = 'Strongly Positive'
    elif exclamation_count >= 2:
        punctuation_sentiment = 'Positive'
    elif ellipsis_count > 0:
        punctuation_sentiment = 'Neutral'
    elif question_count >= 2:
        punctuation_sentiment = 'Negative'
    elif question_count > 0:
        punctuation_sentiment = 'Neutral'
    else:
        punctuation_sentiment = 'Neutral'
    
    return emoji_sentiment, punctuation_sentiment

# Define model prediction functions
def predict_sarcasm_sentiment(text, model, tokenizer, max_length):
    tokens = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(tokens, maxlen=max_length, padding='post')
    prediction = model.predict(padded_input)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    sarcasm = 'Sarcastic' if prediction >= 0.5 else 'Not Sarcastic'
    return sentiment, sarcasm

# Define final sentiment analysis function
def get_final_sentiment(text, model, tokenizer, max_length):
    emoji_sentiment, punctuation_sentiment = analyze_emoji_punctuation(text)
    sentence_sentiment, sarcasm = predict_sarcasm_sentiment(text, model, tokenizer, max_length)
    
    if sarcasm == 'Sarcastic':
        if emoji_sentiment == 'Positive' and punctuation_sentiment in ['Positive', 'Strongly Positive']:
            final_sentiment = 'Sarcastic and Positive'
        elif emoji_sentiment == 'Negative' and punctuation_sentiment == 'Negative':
            final_sentiment = 'Sarcastic and Negative'
        elif emoji_sentiment == 'Positive' or punctuation_sentiment in ['Positive', 'Strongly Positive']:
            final_sentiment = 'Sarcastic and Slightly Positive'
        elif emoji_sentiment == 'Negative' or punctuation_sentiment == 'Negative':
            final_sentiment = 'Sarcastic and Slightly Negative'
        else:
            final_sentiment = 'Sarcastic'
    else:
        if emoji_sentiment == 'Positive' and punctuation_sentiment in ['Positive', 'Strongly Positive']:
            final_sentiment = 'Very Positive'
        elif emoji_sentiment == 'Negative' and punctuation_sentiment == 'Negative':
            final_sentiment = 'Very Negative'
        elif emoji_sentiment == 'Positive' or punctuation_sentiment in ['Positive', 'Strongly Positive']:
            final_sentiment = 'Slightly Positive'
        elif emoji_sentiment == 'Negative' or punctuation_sentiment == 'Negative':
            final_sentiment = 'Slightly Negative'
        else:
            final_sentiment = sentence_sentiment
    
    return {
        'sentence_sentiment': sentence_sentiment,
        'sarcasm': sarcasm,
        'emoji_sentiment': emoji_sentiment,
        'punctuation_sentiment': punctuation_sentiment,
        'final_sentiment': final_sentiment
    }


import instaloader

def get_instagram_comments(post_url, username, password):
    L = instaloader.Instaloader()
    
    # Load session if available
    try:
        L.load_session_from_file(username)
    except FileNotFoundError:
        # If session file is not found, perform login
        L.login(username, password)
        L.save_session_to_file()
    
    shortcode = post_url.split("/")[-2]
    
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except instaloader.exceptions.QueryReturnedBadRequestException as e:
        # Handle the checkpoint_required error
        return f"Error: {e}"
    
    comments = [comment.text for comment in post.get_comments()]
    return comments
from collections import defaultdict

def aggregate_results(results):
    aggregated = defaultdict(lambda: defaultdict(int))
    
    for result in results:
        print(f"Processing result: {result}")  # Debug print
        aggregated['sentence_sentiment'][result['sentence_sentiment']] += 1
        aggregated['sarcasm'][result['sarcasm']] += 1
        aggregated['emoji_sentiment'][result['emoji_sentiment']] += 1
        aggregated['punctuation_sentiment'][result['punctuation_sentiment']] += 1
        aggregated['final_sentiment'][result['final_sentiment']] += 1
    
    final_results = {
        'sentence_sentiment': dict(aggregated['sentence_sentiment']),
        'sarcasm': dict(aggregated['sarcasm']),
        'emoji_sentiment': dict(aggregated['emoji_sentiment']),
        'punctuation_sentiment': dict(aggregated['punctuation_sentiment']),
        'final_sentiment': dict(aggregated['final_sentiment']),
    }
    
    if 'Positive' not in final_results['final_sentiment']:
        final_results['final_sentiment']['Positive'] = 0
    
    print(f"Aggregated results: {final_results}")  # Debug print
    return final_results

# Initialize Flask app and load components
tokenizer_obj, sarcasm_model = load_components()
max_length = 25

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        post_url = request.form['url']
        instagram_username = request.form['username']
        instagram_password = request.form['password']
        
        comments = get_instagram_comments(post_url, instagram_username, instagram_password)
        
        results = [get_final_sentiment(comment, sarcasm_model, tokenizer_obj, max_length) for comment in comments]
        final_results = aggregate_results(results)
        
        return render_template('result.html', results=final_results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
