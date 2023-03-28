# --------------------- Importing Libraries ---------------------------

from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd
from collections import Counter
import json
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


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


# Data splitting
def get_data_splits(X, y, train_size=0.7):
    """Split data into train test and validation set"""

    # train set
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)

    # validation and test set
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)

    return X_train, X_val, X_test, y_train, y_val, y_test


# custom label encoding
class LabelEncoder:
    """Encode labels into unique indices.
    ```python
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)
    ```
    """

    def __init__(self, class_to_index: Dict = {}) -> None:
        """Initialize the label encoder.
        Args:
            class_to_index (Dict, optional): mapping between classes and unique indices. Defaults to {}.
        """
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y: List):
        """Fit a list of labels to the encoder.
        Args:
            y (List): raw labels.
        Returns:
            Fitted LabelEncoder instance.
        """
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y: List) -> np.ndarray:
        """Encode a list of raw labels.
        Args:
            y (List): raw labels.
        Returns:
            np.ndarray: encoded labels as indices.
        """
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y: List) -> List:
        """Decode a list of indices.
        Args:
            y (List): indices.
        Returns:
            List: labels.
        """
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp: str) -> None:
        """Save class instance to JSON file.
        Args:
            fp (str): filepath to save to.
        """
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        """Load instance of LabelEncoder from file.
        Args:
            fp: JSON filepath to load from.
        Returns:
            LabelEncoder instance.
        """
        with open(fp) as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


# preprocessing on data
def preprocess(df, lower, stem, min_freq):
    """ Preprocess the data """
    df["text"] = df.title + " " + df.description  # feature engg.
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text

    # replace OOS labels
    df = replace_oss_labels(df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oss_label="other")

    # replace minority labels
    df = replace_minority_labels(df=df, label_col='tag', min_freq=min_freq, new_label='other')

    return df
