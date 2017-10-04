import pandas as pd
import re

d = pd.read_csv("train_set_x.csv")
r = pd.read_csv("train_set_y.csv")

# Merge the 2 dataframes in a single table by joining on Id
data = pd.merge(d,r,how='inner',on='Id')
data.drop("Id", axis=1, inplace=True)

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


data['Text'] = data['Text'].apply(cleanup) # Clean the data up!

def charCount (s):
    d = {}

    for c in s:
        # Filter all numbers
        if c > '0' and c < '9':
            continue

        lower = c.lower()

        if lower in d.keys():
            d[lower] = d[lower] + 1
        else:
            d[lower] = 1
    i = ''

    if bool(d): #If not empty
        i = max(d, key=d.get)

    return i

data['MCL'] = data['Text'].apply(charCount)

data.to_csv("out.csv", encoding="utf-8")
