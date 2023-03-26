# --------------------- Importing Libraries ---------------------------

from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd
from collections import Counter
import json

from config import config


# ---------------------------------- Preprocessing Techniques ------------------------

# clean text
def clean_text(text, lower=True, stem=False, stopwords=config.STOPWORDS) -> str:
    """Clean row text"""

    # making input text lower
    if lower:
        text = text.lower()

    # remove stopwords
    if len(stopwords):
        pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non-alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


# out of scope labels
def replace_oss_labels(df, labels, label_col, oss_label="other") -> pd.DataFrame:
    """Replace out of scope labels"""

    oss_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oss_label if x in oss_tags else x)
    return df


# minority labels
def replace_minority_labels(df, label_col, min_freq, new_label="other"):
    """Replace Minority labels with another labels"""

    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)

    return df


# custom label encoding

class LabelEncoder(object):
    """ Encode Labels into unique indices"""

    def __int__(self, class_to_index={}):
        """define class variables"""
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        """learning from target y"""

        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    # encode label
    def encode(self, y):
        """apply learning in fit function to encode labels"""

        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    # decode label
    def decode(self, y):
        """Decode the encoded labels"""

        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    # save encoding values
    def save(self, fp):
        """ Save encoded values in json"""

        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    # load file

    @classmethod
    def load(cls, fp):
        """load json file"""
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
