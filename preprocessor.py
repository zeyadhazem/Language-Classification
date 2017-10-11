import re

class Preprocessor:
    """Preprocesses the input"""

    def process (self, orig_df, inplace):
        df = orig_df

        if not inplace:
            df = orig_df.copy()

        column = df.columns[0] # A vector of size m that will be preprocessed
        df[column] = df[column].apply(self.cleanup)

        if not inplace:
            return df

    def cleanup (self, s):
        s = self.convert_to_UTF(s) # Mandatory
        s = self.removeLinksAndSpaces(s)
        # s = self.remove_spaces(s)
        s = self.remove_digits(s)
        # s = self.remove_emojis(s)

        return s

    def convert_to_UTF(self, s):
        s = str(s)
        s = unicode(s,"utf-8")

        return s

    def removeLinksAndSpaces(self, s):
        # Lower case
        string = s.lower().split()
        recreated = ''

        for word in string:
            if 'http' not in word:
                recreated = recreated + word

        return recreated


    def remove_spaces(self, s):
        return s.replace(" ","")

    def remove_digits(self, s):
        return re.sub("\d+", "", s)

    def remove_emojis(self, s):
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

        return myre.sub(r'', s)
