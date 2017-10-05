import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math

from matplotlib.ticker import FuncFormatter, MultipleLocator
from collections import Counter

#####################################################
#                    Functions                      #
#####################################################

def cleanup (s):
    # Parse any non strings to string
    s = str(s)

    # Convert to unicode
    s = unicode(s,"utf-8")

    # Remove all digits from string
    s = re.sub("\d+", "", s)

    # Remove all emojis and non letter characters
    try:
        # Wide UCS-4 build
        myre = re.compile(u'['
            u'\U0001F300-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u26FF\u2700-\u27BF]+',
            re.UNICODE)
    except re.error:
        # Narrow UCS-2 build
        myre = re.compile(u'('
            u'\ud83c[\udf00-\udfff]|'
            u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
            u'[\u2600-\u26FF\u2700-\u27BF])+',
            re.UNICODE)

    return myre.sub(r'', s).replace(" ", "")


def charCount (s):
    """
    Takes in a unicode string representing a string
    Returns a dict with letters as keys and their number of occurence as values
    """
    d = {}
    for each in s:
        lower = each.lower()

        if lower in d.keys():
            d[lower] = d[lower] + 1
        else:
            d[lower] = 1

    return d


def draw_bars (h, legend):
    lists = sorted(h.items()) # sorted by key, return a list of tuples

    x_ticks_labels, y = zip(*lists) # unpack a list of pairs into two tuples
    x = np.array(np.arange(0,len(x_ticks_labels)))

    plt.figure(figsize=(20,4))
    plt.bar(np.arange(0,len(h)), h.values(), color='g')
    plt.xticks(x, h.keys())
    plt.legend(legend)
    plt.show()

def normalize (d):
    """
    Takes in a dictionary containing all the letters organized by language category
    calculates the frequency of each letter by language category
    Returns a dict containing the normalized freq of letters whose freq > uniform distribution prob
    """
    total_count = sum(d.values())
    add = 0
    h = d.copy()

    for key in h.keys():
        h[key] = float(h[key])/total_count
        add = add + h[key]

    print(add, len(h))
    uniform = add/len(h) # if uniformly distributed
    for key in h.keys():
        # Truncate anything that has a rec value less than uniform
        if (h[key] < uniform):
            del h[key]

    return h

def docFrequency (d):

    df = {}
    for language in d:
        for letter in d[language]:
            if letter in df:
                df[letter] = df[letter] + 1
            else:
                df[letter] = 1

    return df


def inverseDocFrequency (d):
    """
    Takes in a dictionary containing all the letters organized by language category
    Returns a dictionary where the keys are the letters and the values are the idf
    """
    df = docFrequency(d)
    length = float(len(d))

    for letter in df:
        df[letter] = math.log10(length/df[letter])

    return df

def draw_multiple_bars(d, normalized):
    for key in d.keys():
        h = dict(d[key]) # Validating and normalizing just by looking at each language on its own
        if normalized:
            h = normalize(h)

        draw_bars(h, key)

def compute_tf_idf(tf, idf):
    # Get the tf idf Now all we need to do is for every language, multiply the tf by idf
    tfidf = d.copy()
    for language in tfidf:
        temp = tfidf[language]
        for letter in temp:
            temp[letter] = temp[letter] * idf[letter] # Will not throw an error since all letters are contained in idf

    return tfidf


#####################################################
#                       Logic                       #
#####################################################

d = pd.read_csv("train_set_x.csv")
r = pd.read_csv("train_set_y.csv")

# Building joining x and y tables
data = pd.merge(d,r,how='inner',on='Id')
data.drop("Id", axis=1, inplace=True)

# Cleanup the data
data['Text'] = data['Text'].apply(cleanup)

# Get the char count for every string in the table
data['MCL'] = data['Text'].apply(charCount)

# Count the frequency of the letters by language category
d = {}
data.count(numeric_only=True)
length = int (data.count(numeric_only=True))
for i in range(0, length - 1):
    lang = str (data['Category'][i])
    if lang in d.keys():
        d[lang] = dict (Counter(d[lang]) + Counter(data['MCL'][i]))
    else:
        d[lang] = data['MCL'][i]

# Get the inverse document frequency
idf = inverseDocFrequency(d)
tfidf = compute_tf_idf(d, idf)

# Non Normalized TF
draw_multiple_bars(d, False)

# Normalized TF
draw_multiple_bars(d, True)

# Non Normalized TFIDF
draw_multiple_bars(tfidf, False)

# Normalized TFIDF
draw_multiple_bars(tfidf, True)

# data.to_csv("out.csv", encoding="utf-8")
