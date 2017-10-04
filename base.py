import pandas as pd

d = pd.read_csv("train_set_x.csv")
r = pd.read_csv("train_set_y.csv")

data = pd.merge(d,r,how='inner',on='Id')
data.drop("Id", axis=1, inplace=True)

data['Text'] = data['Text'].apply(lambda x:str(x).replace(" ", "")) # Remove white spaces

def charCount (s):
    s = unicode(s,"utf-8")
    d = {}

    for c in s:
        # Filter all numbers
        if c > '0' and c < '9':
            continue

        lower = c

        if lower in d.keys():
            d[lower] = d[lower] + 1
        else:
            d[lower] = 1
    i = ''

    if bool(d): #If not empty
        i = max(d, key=d.get)

    return i

data['MCL'] = data['Text'].apply(charCount)

print(data)
