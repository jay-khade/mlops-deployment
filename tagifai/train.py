# ------------------------- Importing Libraries ------------------------

from imblearn.over_sampling import RandomOverSampler
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from tagifai import data, utils, predict, evaluate


# ----------------------------------- Training a Model ----------------------

def train(args, df, trial=None):
    """Train model on data"""

    # setup
    utils.set_seeds()

    # data shuffle
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples

    # preprocess
    df = data.preprocess(df, lower=args.lower, stem=args.stem, min_freq=args.min_freq)

    # label encoding
    label_encoder = data.LabelEncoder().fit(df.tag)

    # data split
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(X=df.text.to_numpy(),
                                                                          y=label_encoder.encode(df.tag))

    # test dataframe
    test_df = pd.DataFrame({'text': X_test, 'tag': label_encoder.decode(y_test)})

    # Tf-Idf
    vectorizer = TfidfVectorizer(analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range))

    # apply vectorization
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # oversampling
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model
    model = SGDClassifier(
        loss="log",
        penalty='l2',
        alpha=args.alpha,
        max_iter=1,
        learning_rate='constant',
        eta0=args.learning_rate,
        power_t=args.power_t,
        warm_start=True
    )

    # Training
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)

        # training loss
        train_loss = log_loss(y_train, model.predict_proba(X_train))

        # validation loss
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        if not epoch % 10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f},"
                f"val_loss: {val_loss:.5f}"
            )

    # threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    args.threshold = np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)

    # evaluation
    other_index = label_encoder.class_to_index["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold, index=other_index)

    performance = evaluate.get_metrics(
        y_true=y_test,
        y_pred=y_pred,
        classes=label_encoder.classes,
        df=test_df
    )

    return{
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }
