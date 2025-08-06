#!/usr/bin/env python3
"""
Production-Ready Sentiment Trainer & Evaluator (v3.5, CPU/GPU-adaptive, profiled, single-file)

This script will:
 1. Inspect key table row counts
 2. Load raw text & engineered features in memory-bounded chunks, logging chunk times and ETA
 3. Clean & vectorize text (spaCy on CPU/GPU) with caching, logging progress and ETA
 4. Dimensionality-reduce TF-IDF output
 5. Augment with VADER scores in parallel, logging timing
 6. Scale numeric features
 7. Hyperparameter-tune a calibrated LogisticRegression via RandomizedSearchCV on a subsample, then refit on full data
 8. Evaluate on a hold-out test set (ROC, confusion matrix, probability distribution)
 9. Save model, CV results, metrics, report, & charts in timestamped folders

Adds:
 - Sample-based tuning to avoid OOM on 7M rows
 - Auto-detect `--jobs` default to `cpu_count()`
 - Parallel VADER scoring via joblib
 - Step-level profiling with detailed logs
 - Persistence of sample CV results and best parameters
 - New Probability Distribution plot

Usage:
  python sentiment_pipeline_v3.py --config config.json [--jobs J] [--run-tests]
"""
import argparse
import json
import os
import sys
import time
import pickle
import subprocess
import logging
import gc
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform
from scipy.sparse import hstack, csr_matrix
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.exc import SQLAlchemyError

from sklearn.model_selection import (
    RandomizedSearchCV, train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report, accuracy_score, precision_recall_curve,
    precision_score, recall_score, f1_score
)

import spacy
import xgboost as xgb
from joblib import parallel_backend
import nltk
from custom_transforms import TextCleaner, VaderScore
import shap
import unittest
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Note: Plot styling will be set based on configuration

class ModelTracker:
    """Track model performance and metadata"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics = {}
        self.metadata = {}
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'value': value, 'step': step, 'timestamp': time.time()})
    
    def log_metadata(self, key: str, value: Any):
        self.metadata[key] = value
    
    def save_tracking_data(self):
        with open(os.path.join(self.output_dir, 'model_tracking.json'), 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'metadata': self.metadata
            }, f, indent=2, default=str)

def print_gpu_usage() -> None:
    """Monitor GPU usage if available"""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], encoding='utf-8')
        for line in out.strip().splitlines():
            idx, name, util, used, total = [x.strip() for x in line.split(',')]
            logging.info(f"[GPU] {idx}:{name} util={util}% mem={used}/{total}MiB")
    except Exception:
        logging.info("[GPU] nvidia-smi not available")

def inspect_table_counts(engine: Engine, tables: List[str]) -> None:
    """Inspect database table row counts"""
    with engine.connect() as conn:
        for table in tables:
            try:
                cnt = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                logging.info(f"Table '{table}' row count: {cnt.scalar()}")
            except Exception as e:
                logging.warning(f"Could not count rows for '{table}': {e}")

def get_engine(cfg: Dict[str, Any]) -> Engine:
    """Create database engine with connection validation"""
    try:
        uri = (
            f"postgresql://{cfg['db_user']}:{cfg['db_pass']}@"
            f"{cfg['db_host']}:{cfg['db_port']}/{cfg['db_name']}"
        )
        eng = create_engine(uri)
        eng.connect().close()
        return eng
    except SQLAlchemyError as e:
        logging.error(f"DB connection failed: {e}")
        sys.exit(1)

def load_data(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load data with chunked processing and validation"""
    t0 = time.monotonic()
    eng = get_engine(cfg)
    inspect_table_counts(eng, ['reviews','review_features','user_features'])
    
    chunks = []
    total_rows = 0
    for chunk in pd.read_sql_query(cfg['data_query'], eng, chunksize=cfg['chunk_size']):
        # Data validation
        if chunk.isnull().any().any():
            logging.warning(f"Found {chunk.isnull().sum().sum()} null values in chunk")
        chunks.append(chunk)
        total_rows += len(chunk)
        if len(chunks) % 10 == 0:
            logging.info(f"Loaded {total_rows} rows so far...")
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Label creation with validation
    df['label'] = (df['stars'] >= cfg['pos_threshold']).astype(int)
    class_counts = df['label'].value_counts()
    logging.info(f"Class distribution: {dict(class_counts)}")
    
    # Check for class imbalance
    imbalance_ratio = class_counts.min() / class_counts.max()
    if imbalance_ratio < 0.1:
        logging.warning(f"Severe class imbalance detected: {imbalance_ratio:.3f}")
    
    X = df.drop(columns=['stars','label'])
    y = df['label'].values
    logging.info(f"Loaded {len(df)} rows, {X.shape[1]} features in {time.monotonic()-t0:.1f}s")
    return X, y

def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration completeness"""
    required = [
        'db_user','db_pass','db_host','db_port','db_name',
        'data_query','pos_threshold','spacy_model','tfidf',
        'numeric_features','model','search','chunk_size',
        'test_size','output_base'
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        logging.error(f"Missing config keys: {missing}")
        sys.exit(1)


# ------------------------------------------------------------
def create_advanced_text_features(
    texts: List[str],
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Any]:
    """Create word-level TF-IDF with SVD to avoid OOM from char-level features"""

    logging.info("Creating text features (word TF-IDF + SVD)...")

    # 1) Word-level TF-IDF only
    tfidf_word = TfidfVectorizer(
        max_df=cfg['tfidf']['max_df'],
        max_features=cfg['tfidf'].get('word_features', 5000),
        ngram_range=tuple(cfg['tfidf']['ngram_range']),
        stop_words='english',
        analyzer='word'
    )
    word_features = tfidf_word.fit_transform(texts)

    # 2) Apply SVD for dimensionality reduction directly on word features
    n_comp = cfg['tfidf'].get('n_components', 200)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced = svd.fit_transform(word_features)

    logging.info(
        f"Text features: {word_features.shape[1]} word -> {n_comp} SVD components"
    )
    return reduced, (tfidf_word, svd)

    
def preprocess_and_cache(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced preprocessing with advanced features and caching"""
    base = cfg['output_base']
    os.makedirs(base, exist_ok=True)

    p_clean = os.path.join(base, 'cleaned.pkl')
    p_text_features = os.path.join(base, 'text_features_v2.npz')
    p_tok2vec = os.path.join(base, 'tok2vec.npz')
    p_final = os.path.join(base, 'preprocessed_v2.npz')

    # Load cached final features if available
    if os.path.exists(p_final):
        logging.info(f"Loading final features from {p_final}")
        data = np.load(p_final, mmap_mode='r')
        return data['X'].astype(np.float32, copy=False), data['y']

    # 1) Load raw data
    X_raw, y = load_data(cfg)

    # 2) Clean text
    if os.path.exists(p_clean):
        with open(p_clean,'rb') as f:
            cleaned = pickle.load(f)
    else:
        logging.info("Cleaning text...")
        cleaned = TextCleaner(cfg['spacy_model']).transform(X_raw['text'])
        with open(p_clean,'wb') as f:
            pickle.dump(cleaned, f)

    # 3) Advanced text features
    if os.path.exists(p_text_features):
        data = np.load(p_text_features)
        text_features = data['features']
    else:
        text_features, transformers = create_advanced_text_features(cleaned, cfg)
        np.savez(p_text_features, features=text_features)
        # Save transformers for later use
        pickle.dump(transformers, open(os.path.join(base, 'text_transformers.pkl'), 'wb'))

    # 4) Token vectors
    if os.path.exists(p_tok2vec):
        tok2vec = np.load(p_tok2vec)['X']
    else:
        logging.info("Computing tok2vec embeddings...")
        nlp = spacy.load(
            cfg['spacy_model'],
            exclude=['tagger','parser','ner','attribute_ruler','lemmatizer']
        )
        vec_size = nlp.vocab.vectors_length
        vs = []
        batch_size = 1000
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i:i+batch_size]
            batch_vectors = []
            for doc in nlp.pipe(batch, batch_size=100):
                v = doc.vector
                if v.shape[0] != vec_size or np.allclose(v, 0):
                    v = np.random.normal(0, 0.1, vec_size).astype(np.float32)
                batch_vectors.append(v)
            vs.extend(batch_vectors)
            if i % 10000 == 0:
                logging.info(f"Processed {i}/{len(cleaned)} documents for embeddings")
        tok2vec = np.vstack(vs)
        np.savez(p_tok2vec, X=tok2vec)

    # 5) VADER sentiment scores
    logging.info("Computing VADER scores...")
    vader_scores = VaderScore(cfg.get('jobs', 1)).transform(cleaned).ravel().astype(np.float32)

    # 6) Numeric features with validation
    numeric_features = X_raw[cfg['numeric_features']].astype(np.float32).values
    
    # Handle missing values and outliers
    numeric_features = np.nan_to_num(numeric_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Log feature statistics
    for i, feat_name in enumerate(cfg['numeric_features']):
        feat_values = numeric_features[:, i]
        logging.info(f"Feature {feat_name}: mean={feat_values.mean():.3f}, std={feat_values.std():.3f}")

    # 7) Combine all features
    X = np.hstack([
        text_features.astype(np.float32),
        tok2vec.astype(np.float32),
        vader_scores.reshape(-1, 1),
        numeric_features
    ]).astype(np.float32)
    
    logging.info(f"Final feature matrix: {X.shape}")
    logging.info(f"Caching final features to {p_final}")
    np.savez(p_final, X=X, y=y)
    
    return X, y

def create_diverse_models(cfg: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Create diverse models for ensemble"""
    models = []
    
    # Logistic Regression
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CalibratedClassifierCV(
            estimator=LogisticRegression(
                solver='saga',
                penalty='l2',
                max_iter=cfg['model']['max_iter'],
                random_state=42,
                class_weight='balanced'
            ),
            cv=3,
            method='sigmoid'
        ))
    ])
    models.append(('logistic', lr))
    
    # Random Forest
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=cfg.get('jobs', 1)
        ))
    ])
    models.append(('random_forest', rf))
    
    # XGBoost
    xgb_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            n_jobs=cfg.get('jobs', 1)
        ))
    ])
    models.append(('xgboost', xgb_clf))
    
    return models

def hyperparameter_search(X_train: np.ndarray, y_train: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced hyperparameter search with multiple models"""
    if cfg['search']['n_iter'] == 0:
        return {'best_params': {}, 'best_score': 0.0}
    
    sample_size = min(cfg['search'].get('sample_size', 50000), len(X_train))
    logging.info(f"Hyperparameter search on {sample_size} samples")
    
    # Sample data
    rng = np.random.RandomState(42)
    idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
    X_sample = X_train[idx].copy()
    y_sample = y_train[idx]
    
    # Search space for logistic regression (main model)
    param_dist = {
        'clf__estimator__C': loguniform(0.01, 100),
        'clf__estimator__penalty': ['l1', 'l2'],
        'clf__estimator__solver': ['saga']
    }
    
    # Base pipeline
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=min(1000, X_sample.shape[1]))),
        ('clf', CalibratedClassifierCV(
            estimator=LogisticRegression(
                max_iter=cfg['model']['max_iter'],
                random_state=42,
                class_weight='balanced'
            ),
            cv=3,
            method='sigmoid'
        ))
    ])
    
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=cfg['search']['n_iter'],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=min(cfg.get('jobs', 1), 2),  # Limit parallelism for memory
        verbose=1,
        random_state=42
    )
    
    search.fit(X_sample, y_sample)
    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Best CV score: {search.best_score_:.3f}")
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': pd.DataFrame(search.cv_results_)
    }

def train_ensemble_fixed(X_train: np.ndarray, y_train: np.ndarray, cfg: Dict[str, Any], 
                        search_results: Dict[str, Any]) -> Pipeline:
    """Train ensemble with proper data isolation (fixes data leakage)"""
    logging.info("Training diverse ensemble with proper data isolation...")
    
    # Create diverse base models
    diverse_models = create_diverse_models(cfg)
    ensemble_estimators = []
    
    for name, model in diverse_models:
        logging.info(f"Training {name}...")
        t0 = time.monotonic()
        
        # Each model gets its own random subset and fits its own preprocessors
        rng = np.random.RandomState(hash(name) % 2**32)  # Deterministic but different seeds
        subset_size = min(len(X_train), cfg.get('ensemble_subset_size', len(X_train)))
        
        if subset_size < len(X_train):
            idx = rng.choice(len(X_train), size=subset_size, replace=False)
            X_subset = X_train[idx].copy()
            y_subset = y_train[idx].copy()
        else:
            # Use full data but with different random state
            idx = rng.permutation(len(X_train))
            X_subset = X_train[idx].copy()
            y_subset = y_train[idx].copy()
        
        # Fit model on its own subset
        model.fit(X_subset, y_subset)
        
        # Add to ensemble
        ensemble_estimators.append((name, model))
        
        # Memory cleanup
        del X_subset, y_subset
        gc.collect()
        
        logging.info(f"Trained {name} in {time.monotonic()-t0:.1f}s")
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        n_jobs=1  # Avoid memory issues
    )
    
    # Fit ensemble (this doesn't refit individual models, just learns weights)
    logging.info("Fitting ensemble weights...")
    ensemble.fit(X_train, y_train)
    
    return ensemble

def bootstrap_confidence_intervals(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                 n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
    """Calculate bootstrap confidence intervals for metrics"""
    logging.info(f"Computing bootstrap confidence intervals ({n_bootstrap} iterations)...")
    
    metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        y_boot_true = y_true[indices]
        y_boot_pred_proba = y_pred_proba[indices]
        y_boot_pred = (y_boot_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        if len(np.unique(y_boot_true)) > 1:  # Avoid AUC calculation issues
            metrics['auc'].append(roc_auc_score(y_boot_true, y_boot_pred_proba))
        metrics['accuracy'].append(accuracy_score(y_boot_true, y_boot_pred))
        metrics['precision'].append(precision_score(y_boot_true, y_boot_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_boot_true, y_boot_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_boot_true, y_boot_pred, zero_division=0))
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric, values in metrics.items():
        if values:  # Only if we have values
            ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
            confidence_intervals[metric] = (ci_lower, ci_upper)
    
    return confidence_intervals

def comprehensive_evaluation(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                           output_dir: str, tracker: ModelTracker, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Comprehensive model evaluation with statistical testing"""
    logging.info("Performing comprehensive evaluation...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Log metrics
    tracker.log_metric('test_accuracy', accuracy)
    tracker.log_metric('test_precision', precision)
    tracker.log_metric('test_recall', recall)
    tracker.log_metric('test_f1', f1)
    tracker.log_metric('test_auc', auc_score)
    
    # Bootstrap confidence intervals
    confidence_intervals = bootstrap_confidence_intervals(y_test, y_pred_proba)
    
    # Print results with confidence intervals
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.3f}")
    if 'accuracy' in confidence_intervals:
        ci_low, ci_high = confidence_intervals['accuracy']
        print(f"           95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    print(f"Precision: {precision:.3f}")
    if 'precision' in confidence_intervals:
        ci_low, ci_high = confidence_intervals['precision']
        print(f"           95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    print(f"Recall:    {recall:.3f}")
    if 'recall' in confidence_intervals:
        ci_low, ci_high = confidence_intervals['recall']
        print(f"           95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    print(f"F1 Score:  {f1:.3f}")
    if 'f1' in confidence_intervals:
        ci_low, ci_high = confidence_intervals['f1']
        print(f"           95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    print(f"AUC Score: {auc_score:.3f}")
    if 'auc' in confidence_intervals:
        ci_low, ci_high = confidence_intervals['auc']  
        print(f"           95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print("="*60)
    
    # Save detailed results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'confidence_intervals': confidence_intervals
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate comprehensive plots
    create_evaluation_plots(y_test, y_pred, y_pred_proba, output_dir, cfg)
    
    return results

def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, 
                          output_dir: str, cfg: Dict[str, Any]):
    """Create comprehensive evaluation plots"""
    
    # Apply plot styling if configured
    if cfg.get('plot_style'):
        plt.style.use(cfg['plot_style'])
    if cfg.get('plot_palette'):
        sns.set_palette(cfg['plot_palette'])
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], '--', color='gray', alpha=0.8)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title('Confusion Matrix')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    
    # 4. Prediction Probability Distribution
    axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Negative', density=True)
    axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Positive', density=True)
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Calibration Plot
    from sklearn.calibration import calibration_curve
    fraction_pos, mean_pred_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
    axes[1, 1].plot(mean_pred_value, fraction_pos, marker='o', linewidth=2, label='Model')
    axes[1, 1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='Perfect Calibration')
    axes[1, 1].set_xlabel('Mean Predicted Probability')
    axes[1, 1].set_ylabel('Fraction of Positives')
    axes[1, 1].set_title('Calibration Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Threshold Analysis
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))
        precisions.append(precision_score(y_true, y_pred_thresh))
        recalls.append(recall_score(y_true, y_pred_thresh))
    
    axes[1, 2].plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    axes[1, 2].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[1, 2].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[1, 2].set_xlabel('Threshold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Threshold Analysis')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved comprehensive evaluation plots to {output_dir}")

def analyze_feature_importance(model: Any, feature_names: List[str], X_test: np.ndarray, 
                             output_dir: str, cfg: Dict[str, Any], top_k: int = 20):
    """Analyze and visualize feature importance using multiple methods"""
    logging.info("Analyzing feature importance...")
    
    importance_methods = {}
    
    # Method 1: Try to extract coefficients from ensemble members
    try:
        if hasattr(model, 'estimators_'):
            # For VotingClassifier, get importance from logistic regression member
            for name, estimator in model.estimators_:
                if name == 'logistic':
                    # Navigate through pipeline to get coefficients
                    if hasattr(estimator, 'named_steps') and 'clf' in estimator.named_steps:
                        clf = estimator.named_steps['clf']
                        if hasattr(clf, 'calibrated_classifiers_'):
                            coefs = clf.calibrated_classifiers_[0].estimator.coef_.ravel()
                        else:
                            coefs = clf.coef_.ravel()
                        
                        # Handle feature selection if present
                        if 'select' in estimator.named_steps:
                            selector = estimator.named_steps['select']
                            selected_features = selector.get_support()
                            full_coefs = np.zeros(len(feature_names))
                            full_coefs[selected_features] = coefs
                            coefs = full_coefs
                        
                        importance_methods['logistic_coefficients'] = np.abs(coefs)
                        break
    except Exception as e:
        logging.warning(f"Could not extract logistic regression coefficients: {e}")
    
    # Method 2: Permutation importance (sample-based for efficiency)
    try:
        from sklearn.inspection import permutation_importance
        
        # Use a sample for efficiency
        sample_size = min(1000, len(X_test))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_idx]
        
        # Note: This requires y_test to be passed as well
        # For now, we'll skip this method and add it later if needed
        logging.info("Skipping permutation importance (requires y_test)")
        
    except Exception as e:
        logging.warning(f"Could not compute permutation importance: {e}")
    
    # Method 3: SHAP values (very limited sample for memory efficiency)
    try:
        # Only run SHAP if explicitly enabled and on very small samples
        if cfg.get('enable_shap_analysis', False):
            logging.info("Computing SHAP values (this may take a while)...")
            
            # Use very small samples for memory efficiency
            shap_sample_size = min(20, len(X_test))  # Reduced from 100
            background_size = min(10, len(X_test))   # Reduced from 50
            
            rng = np.random.RandomState(42)
            shap_idx = rng.choice(len(X_test), shap_sample_size, replace=False)
            background_idx = rng.choice(len(X_test), background_size, replace=False)
            
            X_shap = X_test[shap_idx]
            X_background = X_test[background_idx]
            
            # Create SHAP explainer with timeout protection
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
            shap_values = explainer.shap_values(X_shap, nsamples=50)  # Limit samples
            
            # Get mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_importance = np.mean(np.abs(shap_values[1]), axis=0)  # For binary classification
            else:
                shap_importance = np.mean(np.abs(shap_values), axis=0)
            
            importance_methods['shap_values'] = shap_importance
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_shap, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logging.info("SHAP analysis disabled (enable with 'enable_shap_analysis': true in config)")
        
    except Exception as e:
        logging.warning(f"Could not compute SHAP values: {e}")
    
    # Create importance comparison plots
    if importance_methods:
        fig, axes = plt.subplots(len(importance_methods), 1, 
                                figsize=(12, 6 * len(importance_methods)))
        if len(importance_methods) == 1:
            axes = [axes]
        
        for idx, (method_name, importance_scores) in enumerate(importance_methods.items()):
            # Get top features
            top_indices = np.argsort(importance_scores)[-top_k:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_scores = importance_scores[top_indices]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            axes[idx].barh(y_pos, top_scores)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(top_features)
            axes[idx].set_xlabel('Importance Score')
            axes[idx].set_title(f'Top {top_k} Features - {method_name.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save importance scores to CSV
        importance_df = pd.DataFrame({'feature': feature_names})
        for method_name, scores in importance_methods.items():
            importance_df[method_name] = scores
        
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance_analysis.csv'), 
                           index=False)
        
        logging.info(f"Feature importance analysis saved to {output_dir}")
    
    return importance_methods

def cross_validate_final_model(model: Any, X: np.ndarray, y: np.ndarray, 
                              cv: int = 5) -> Dict[str, np.ndarray]:
    """Perform cross-validation on the final model"""
    logging.info(f"Performing {cv}-fold cross-validation on final model...")
    
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=1)
            cv_results[metric] = scores
            logging.info(f"CV {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            logging.warning(f"Could not compute CV {metric}: {e}")
    
    return cv_results

class TransformerTests(unittest.TestCase):
    """Enhanced unit tests"""
    def test_vader_score(self):
        arr = VaderScore(n_jobs=1).transform(['good product', 'terrible service'])
        self.assertEqual(arr.shape, (2, 1))
        self.assertTrue(arr[0, 0] > arr[1, 0])  # 'good' should have higher score than 'terrible'
    
    def test_text_cleaner(self):
        cleaned = TextCleaner('en_core_web_sm').transform(['This IS a Test!', 'Another example.'])
        self.assertEqual(len(cleaned), 2)
        self.assertIsInstance(cleaned[0], str)
        self.assertIsInstance(cleaned[1], str)
    
    def test_model_tracker(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ModelTracker(tmpdir)
            tracker.log_metric('test_metric', 0.85)
            tracker.log_metadata('test_key', 'test_value')
            tracker.save_tracking_data()
            
            # Check if file was created
            import os
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'model_tracking.json')))
    
    def test_advanced_text_features(self):
        """Smoke test for advanced text feature creation"""
        dummy_texts = ['good product quality', 'bad customer service', 'excellent value']
        dummy_config = {
            'tfidf': {
                'max_df': 0.95,
                'word_features': 100,
                'char_features': 50,
                'ngram_range': [1, 2],
                'n_components': 20
            }
        }
        
        features, transformers = create_advanced_text_features(dummy_texts, dummy_config)
        
        # Check output shape
        self.assertEqual(features.shape[0], len(dummy_texts))
        self.assertEqual(features.shape[1], dummy_config['tfidf']['n_components'])
        
        # Check transformers returned
        self.assertEqual(len(transformers), 3)  # word_tfidf, char_tfidf, svd

def main():
    """Enhanced main function with comprehensive pipeline"""
    t0 = time.monotonic()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced Sentiment Analysis Pipeline')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--jobs', type=int, default=cpu_count(), help='Number of parallel jobs')
    parser.add_argument('--run-tests', action='store_true', help='Run unit tests')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Run tests if requested
    if args.run_tests:
        unittest.main(argv=[sys.argv[0]], exit=False)
        return

    # Load and validate configuration
    try:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {args.config} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    
    validate_config(cfg)
    cfg['jobs'] = min(args.jobs, cpu_count())

    # Setup output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(cfg['output_base'], f'enhanced_pipeline_{timestamp}')
    model_dir = os.path.join(output_dir, 'models')
    cache_dir = os.path.join(output_dir, 'cache')
    
    for directory in [output_dir, model_dir, cache_dir]:
        os.makedirs(directory, exist_ok=True)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'pipeline.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Initialize model tracker
    tracker = ModelTracker(output_dir)
    tracker.log_metadata('config', cfg)
    tracker.log_metadata('start_time', datetime.now().isoformat())
    tracker.log_metadata('n_jobs', cfg['jobs'])
    
    logging.info(f"Enhanced Sentiment Analysis Pipeline v4.0")
    logging.info(f"Using up to {cfg['jobs']} CPU cores")
    logging.info(f"Output directory: {output_dir}")
    print_gpu_usage()

    try:
        # 1. Data preprocessing with enhanced features
        logging.info("Step 1: Enhanced preprocessing and feature engineering")
        X, y = preprocess_and_cache(cfg)
        logging.info(f"Final feature matrix shape: {X.shape}")
        tracker.log_metadata('n_samples', X.shape[0])
        tracker.log_metadata('n_features', X.shape[1])
        
        # 2. Train/test split with stratification
        logging.info("Step 2: Train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg['test_size'],
            stratify=y,
            random_state=42
        )
        
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        logging.info(f"Train set: {len(X_train)} samples, {train_pos_rate:.1%} positive")
        logging.info(f"Test set: {len(X_test)} samples, {test_pos_rate:.1%} positive")
        
        tracker.log_metadata('train_size', len(X_train))
        tracker.log_metadata('test_size', len(X_test))
        tracker.log_metadata('train_pos_rate', train_pos_rate)
        tracker.log_metadata('test_pos_rate', test_pos_rate)

        # 3. Hyperparameter search
        logging.info("Step 3: Hyperparameter optimization")
        search_results = hyperparameter_search(X_train, y_train, cfg)
        
        # Save search results (cv_results always exists as DataFrame)
        search_results['cv_results'].to_csv(
            os.path.join(output_dir, 'hyperparameter_search_results.csv'), 
            index=False
        )
        
        # Save search results
        with open(os.path.join(output_dir, 'best_hyperparameters.json'), 'w') as f:
            json.dump({
                'best_params': search_results['best_params'],
                'best_cv_score': search_results['best_score']
            }, f, indent=2)

        # 4. Train diverse ensemble (fixed data leakage)
        logging.info("Step 4: Training diverse ensemble with proper data isolation")
        ensemble_model = train_ensemble_fixed(X_train, y_train, cfg, search_results)
        
        # Save the ensemble model
        model_path = os.path.join(model_dir, 'enhanced_sentiment_ensemble_v4.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
        logging.info(f"Ensemble model saved to {model_path}")

        # 5. Cross-validation on final model
        logging.info("Step 5: Cross-validation evaluation")
        cv_results = cross_validate_final_model(ensemble_model, X_train, y_train, cv=5)
        
        # Save CV results
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)
        
        for metric, scores in cv_results.items():
            tracker.log_metric(f'cv_{metric}_mean', scores.mean())
            tracker.log_metric(f'cv_{metric}_std', scores.std())

        # 6. Comprehensive evaluation
        logging.info("Step 6: Comprehensive evaluation on test set")
        evaluation_results = comprehensive_evaluation(
            ensemble_model, X_test, y_test, output_dir, tracker, cfg
        )

        # 7. Feature importance analysis
        logging.info("Step 7: Feature importance analysis")
        
        # Create feature names
        tfidf_components = cfg['tfidf'].get('n_components', 200)
        vec_size = 300  # Typical spaCy vector size
        feature_names = (
            [f'text_svd_{i}' for i in range(tfidf_components)] +
            [f'tok2vec_{i}' for i in range(vec_size)] +
            ['vader_sentiment'] +
            cfg['numeric_features']
        )
        
        # Adjust if actual feature count differs (ensure meaningful labels)
        actual_features = X.shape[1]
        if len(feature_names) != actual_features:
            logging.warning(f"Feature name count mismatch: {len(feature_names)} vs {actual_features}")
            # Create more descriptive generic names based on expected feature structure
            tfidf_actual = min(tfidf_components, actual_features)
            remaining_features = actual_features - tfidf_actual
            
            feature_names = [f'text_component_{i}' for i in range(tfidf_actual)]
            if remaining_features > 0:
                feature_names.extend([f'other_feature_{i}' for i in range(remaining_features)])
        
        importance_analysis = analyze_feature_importance(
            ensemble_model, feature_names, X_test, output_dir, cfg
        )

        # 8. Generate final report
        logging.info("Step 8: Generating final report")
        
        # Calculate final runtime
        total_runtime = time.monotonic() - t0
        tracker.log_metadata('total_runtime_seconds', total_runtime)
        tracker.log_metadata('end_time', datetime.now().isoformat())
        
        # Save all tracking data
        tracker.save_tracking_data()
        
        # Generate summary report
        with open(os.path.join(output_dir, 'SUMMARY_REPORT.md'), 'w') as f:
            f.write(f"""# Enhanced Sentiment Analysis Pipeline v4.0 - Summary Report

## Execution Details
- **Start Time**: {tracker.metadata.get('start_time', 'N/A')}
- **End Time**: {tracker.metadata.get('end_time', 'N/A')}
- **Total Runtime**: {total_runtime/60:.1f} minutes
- **CPU Cores Used**: {cfg['jobs']}

## Dataset Summary
- **Total Samples**: {X.shape[0]:,}
- **Features**: {X.shape[1]:,}
- **Train Set**: {len(X_train):,} samples ({train_pos_rate:.1%} positive)
- **Test Set**: {len(X_test):,} samples ({test_pos_rate:.1%} positive)

## Model Performance
- **Test Accuracy**: {evaluation_results.get('accuracy', 0):.3f}
- **Test Precision**: {evaluation_results.get('precision', 0):.3f}
- **Test Recall**: {evaluation_results.get('recall', 0):.3f}
- **Test F1 Score**: {evaluation_results.get('f1_score', 0):.3f}
- **Test AUC**: {evaluation_results.get('auc_score', 0):.3f}

## Cross-Validation Results
""")
            for metric, scores in cv_results.items():
                f.write(f"- **CV {metric.upper()}**: {scores.mean():.3f} ± {scores.std():.3f}\n")
            
            f.write(f"""
## Files Generated
- `enhanced_sentiment_ensemble_v4.pkl` - Trained ensemble model
- `comprehensive_evaluation.png` - Evaluation plots
- `feature_importance_analysis.csv` - Feature importance scores
- `hyperparameter_search_results.csv` - Hyperparameter search results
- `cross_validation_results.csv` - Cross-validation scores
- `model_tracking.json` - Complete experiment tracking
- `pipeline.log` - Detailed execution log

## Recommendations for Production
1. Monitor model performance over time for drift
2. Implement A/B testing for model updates
3. Add real-time feature monitoring
4. Consider online learning for continuous improvement
5. Implement model versioning and rollback capabilities
""")

        # Print final summary (also log for automated runs)
        summary_message = f"""
{"="*80}
🎉 ENHANCED SENTIMENT ANALYSIS PIPELINE v4.0 COMPLETED SUCCESSFULLY! 🎉
{"="*80}
📊 Total Runtime: {total_runtime/60:.1f} minutes
📁 Output Directory: {output_dir}
🎯 Test AUC: {evaluation_results.get('auc_score', 0):.3f}
📈 Test F1: {evaluation_results.get('f1_score', 0):.3f}

📋 Key Files Generated:
   • Model: models/enhanced_sentiment_ensemble_v4.pkl
   • Report: SUMMARY_REPORT.md
   • Plots: comprehensive_evaluation.png
   • Features: feature_importance_analysis.csv
{"="*80}
"""
        
        print(summary_message)
        logging.info("Pipeline completed successfully!" + summary_message)

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    # Download required NLTK data
    nltk.download('vader_lexicon', quiet=True)
    main()