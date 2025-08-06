# custom_transforms.py

import logging
import time
from typing import List, Optional
import spacy
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, model: str):
        self.model = model
        self._load_model_cpu()

    def _load_model_cpu(self):
        # Disable all trainable/pipeline components we don't need
        disable = ["tok2vec", "tagger", "parser", "ner"]
        self.nlp = spacy.load(self.model, disable=disable)
        self.using_gpu = False
        msg = "spaCy loaded on CPU"
        logging.info(msg)
        print(msg)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('nlp', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model_cpu()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        total = len(X)
        start = time.monotonic()
        cleaned = []
        for i, doc in enumerate(self.nlp.pipe(X, batch_size=10000), start=1):
            tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct]
            cleaned.append(" ".join(tokens))
            if i % 10000 == 0 or i == total:
                elapsed = time.monotonic() - start
                avg = elapsed / i
                eta = avg * (total - i)
                msg = f"Cleaned {i}/{total} | elapsed={elapsed:.1f}s ETA={eta:.1f}s"
                logging.info(msg)
                print(msg)
        return cleaned

class VaderScore(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs or cpu_count()
        self._init_analyzer()

    def _init_analyzer(self):
        msg = "Initializing VADER"
        logging.info(msg)
        print(msg)
        self.analyzer = SentimentIntensityAnalyzer()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('analyzer', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_analyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        msg = f"VADER scoring {len(X)} texts with {self.n_jobs} cores..."
        logging.info(msg)
        print(msg)
        with parallel_backend('loky', n_jobs=self.n_jobs):
            start = time.monotonic()
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.analyzer.polarity_scores)(text) for text in X
            )
        arr = np.array([[s['compound']] for s in scores])
        elapsed = time.monotonic() - start
        msg = f"VADER scored {len(X)} texts in {elapsed:.1f}s"
        logging.info(msg)
        print(msg)
        return arr
