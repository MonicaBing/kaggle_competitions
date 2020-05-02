#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:12:35 2020

@author: kathykiutang
"""

import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#vectorise it 
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df['text'])
test_vectors = count_vectorizer.transform(test_df['text'])

clf = linear_model.SGDClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

clf.fit(train_vectors, train_df['target'])

sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv("submission.csv", index=False)