import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import time

# Download NLTK corpora for sentiment analysis
nltk.download('vader_lexicon')

# Load data from Excel file into a pandas DataFrame --- using excel for ease, but data can be loaded from any source, i.e. db, etc. 
# updatepathtofilehere with actual path to your Excel file
df = pd.read_excel('updatepathtofilehere.xlsx')

# Create SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Handle missing values
df['Feedback'].fillna('', inplace=True)

# Apply sentiment analysis to feedback text and store scores in 'Sentiment' column
start_time = time.time()
df['Sentiment'] = df['Feedback'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
print("Sentiment analysis completed. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

# Define function to categorize sentiment scores into positive, negative, or neutral
def categorize_sentiment(score):
    if score >= 0.5:
        return 'Positive'
    elif score <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Apply categorization function to sentiment scores and store categories in 'Sentiment_Category' column
start_time = time.time()
df['Sentiment_Category'] = df['Sentiment'].apply(categorize_sentiment)
print("Sentiment categorization completed. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

# Load pre-trained BERT model and tokenizer ------ TODO train model
start_time = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
print("BERT model loaded. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

# Define the theme labels - TODO need to update to add OTHER
theme_labels = ['COURSE CONTENT', 'LEVEL OF DIFFICULTY', 'PACE', 'TECHNICAL ISSUES', 'INSTRUCTOR/DELIVERY']

# Classify the feedback into themes using BERT model
def classify_theme(feedback):
    if isinstance(feedback, str) and feedback.strip() == '':
        return 'N/A'
    elif not isinstance(feedback, str):
        return 'Invalid'
    inputs = tokenizer.encode_plus(
        feedback,
        None,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = theme_labels[logits.argmax()]
    return predicted_label

# Apply theme classification function to feedback and store themes in 'Feedback_Theme' column
start_time = time.time()
df['Feedback_Theme'] = df['Feedback'].apply(classify_theme)
print("Theme classification completed. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

# Save updated DataFrame to a new Excel file using openpyxl
# Replace 'updatefilenamehere.xlsx' with whatever name you want for your ouptut file 
start_time = time.time()
df.to_excel('updatefilenamehere.xlsx', index=False, engine='openpyxl')
print("Output file created. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

