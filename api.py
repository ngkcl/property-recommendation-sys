
from flask import Flask, render_template, request
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask.json import jsonify
import os

app = Flask(__name__)


# Constants:

CSV_FILE_NAME = './airbnb-london.csv'
# Number of cut rows (Memory issues):
NROWS = 8000


pd.set_option('display.max_columns', 100)
# NOTE: Cutting csv, due to memory errors. Might remove when deployed on server
try:
    print('Reading CSV file...')
    df = pd.read_csv(CSV_FILE_NAME, nrows=NROWS)
except Exception as e:
    print(e)

# Select columns we are going to use for recommendations
df = df[['name','host_name','neighbourhood','room_type']]

print(df.head())

# Drop NaN values
df.dropna(inplace=True)

print('Cleaning Data...')

# putting data in lists of words
df['host_name'] = df['host_name'].map(lambda x: x.lower().split(' '))

df['neighbourhood'] = df['neighbourhood'].map(lambda x: x.split(' '))

df['room_type'] = df['room_type'].map(lambda x: x.split(' '))

for index, row in df.iterrows():
    row['host_name'] = ''.join(row['host_name']).lower()

# initializing the new weighted_words column
df['weighted_words'] = ""

'''
Iterate over rows and use Rake to get weighted keywords
and remove unneeded words
'''
for index, row in df.iterrows():
    property_name = row['name']
    # instantiating Rake
    r = Rake()

    # extracting the words
    r.extract_keywords_from_text(property_name)

    # getting the dictionary with weighted words and their scores
    weighted_words_dict_scores = r.get_word_degrees()
    
    # assigning the weighted words to the new column
    row['weighted_words'] = list(weighted_words_dict_scores.keys())

# Set 'name' as index
df.set_index('name', inplace=True)
print(df.head())

# Instantate a new column
df['key_words'] = ''

# Fill keywords column
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'host_name':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['key_words'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'key_words'], inplace = True)

print('Generating Matrix...')

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['key_words'])

indices = pd.Series(df.index)
print(indices)



# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

# function that takes in property name as input and returns the top 10 recommended properties (by name)
def recommendations(property_name, cosine_sim = cosine_sim):
    recommended_properties = []
    
    # gettin the index of the property that matches the name
    try:
        idx = indices[indices == property_name].index[0]
    except:
        return ['']

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar properties
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the names of the best 10 matching properties
    for i in top_10_indexes:
        recommended_properties.append(list(df.index)[i])
        
    return recommended_properties

@app.route('/recommend', methods=['GET', 'POST'])
def predict():
    data = request.args
    output = recommendations(data['name'])
    return jsonify({'recommended': output})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port=port)
