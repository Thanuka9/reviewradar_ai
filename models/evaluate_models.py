#!/usr/bin/env python3
"""
Enhanced Model Comparison & Evaluation Suite

Recursively evaluates saved .pkl models, generates clean visualizations,
and creates an HTML dashboard with improved metrics accuracy.
"""
import argparse
import os
import glob
import pickle
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve,
                            precision_recall_curve, confusion_matrix)
from sklearn.calibration import calibration_curve

# Set global style for consistent, clean visuals
sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

def load_models(models_root):
    """Load all .pkl models from `models/` subdirectories."""
    models = {}
    pattern = os.path.join(models_root, "**", "models", "*.pkl")
    for path in glob.glob(pattern, recursive=True):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            logging.warning(f"  ⚠️ Skipping {path}: {e}")
    return models

def evaluate_model(name, model, X, y):
    """Compute precise predictions and metrics with zero_division handling."""
    try:
        y_pred = model.predict(X)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_proba)
            metrics["avg_proba"] = float(np.mean(y_proba))
        else:
            metrics["roc_auc"] = np.nan
            metrics["avg_proba"] = np.nan
        return metrics, y_pred, y_proba
    except Exception as e:
        logging.warning(f"  ⚠️ Failed to evaluate {name}: {e}")
        return None

def plot_performance_bars(df, out_dir):
    """Create a clean bar chart with rotated labels and proper spacing."""
    plt.figure(figsize=(12, 6))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_plot = df.set_index("model")[metrics]
    df_plot.plot(kind="bar", rot=45, width=0.8, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    plt.title("Model Performance Comparison", pad=15)
    plt.xlabel("Model", labelpad=10)
    plt.ylabel("Score", labelpad=10)
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(range(len(df_plot.index)), df_plot.index, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance_bars.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_metric_boxplots(df, out_dir):
    """Generate a boxplot for metric distribution with enhanced clarity."""
    plt.figure(figsize=(12, 6))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    sns.boxplot(data=df[metrics], palette="muted")
    plt.title("Metric Distribution Across Models", pad=15)
    plt.xlabel("Metrics", labelpad=10)
    plt.ylabel("Score", labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metric_boxplots.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_roc_pr(models_data, y, out_dir):
    """Plot ROC and Precision-Recall curves with improved labeling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.3})
    
    for name, (_, _, y_proba) in models_data.items():
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y, y_proba)
            ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, y_proba):.3f})")
            prec, rec, _ = precision_recall_curve(y, y_proba)
            ax2.plot(rec, prec, label=name)
    
    ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set_title("ROC Curves")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    
    ax2.set_title("Precision-Recall Curves")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_pr_curves.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_confusion_matrices(models_data, y, out_dir):
    """Create clear confusion matrices for each model."""
    for name, (_, y_pred, _) in models_data.items():
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_{name}.png"), bbox_inches="tight", dpi=300)
        plt.close()

def plot_proba_distribution(models_data, out_dir):
    """Plot probability distributions with better separation."""
    plt.figure(figsize=(10, 6))
    for name, (_, _, y_proba) in models_data.items():
        if y_proba is not None:
            sns.kdeplot(y_proba, label=name, fill=True, alpha=0.3, linewidth=1.5)
    plt.title("Predicted Probability Distributions", pad=15)
    plt.xlabel("P(Positive)", labelpad=10)
    plt.ylabel("Density", labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "proba_distributions.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_calibration(models_data, y, out_dir):
    """Generate calibration curves with enhanced visibility."""
    plt.figure(figsize=(10, 6))
    for name, (_, _, y_proba) in models_data.items():
        if y_proba is not None:
            frac_pos, mean_pred = calibration_curve(y, y_proba, n_bins=10)
            plt.plot(mean_pred, frac_pos, marker="o", label=name, markersize=6)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("Calibration Curves", pad=15)
    plt.xlabel("Mean Predicted Probability", labelpad=10)
    plt.ylabel("Fraction Positive", labelpad=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_curves.png"), bbox_inches="tight", dpi=300)
    plt.close()

def write_dashboard(df, out_dir):
    """Create an HTML dashboard with embedded plots and Bootstrap styling."""
    html = [
        '<html>',
        '<head>',
        '<title>Model Comparison Dashboard</title>',
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">',
        '</head>',
        '<body>',
        '<h1>Model Performance Dashboard</h1>',
        '<h2>Metrics Table</h2>',
        df.to_html(index=False, classes="table table-striped"),
        '<h2>Visualizations</h2>',
    ]
    for img in sorted(os.listdir(out_dir)):
        if img.endswith(".png"):
            html.append(f'<h3>{img.replace(".png", "")}</h3><img src="{img}" style="max-width:100%;"><br>')
    html.append('</body></html>')
    with open(os.path.join(out_dir, "dashboard.html"), "w") as f:
        f.write("\n".join(html))

def main():
    parser = argparse.ArgumentParser(description="Evaluate models and generate visualizations.")
    parser.add_argument("--models-dir", default="outputs", help="Root directory for models")
    parser.add_argument("--test-data", default=None, help="Path to test data .npz")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--top-n", type=int, default=None, help="Plot only the top N models by ROC AUC")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info("Starting model evaluation...")

    if args.test_data is None:
        args.test_data = os.path.join(args.models_dir, "preprocessed_v2.npz")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = np.load(args.test_data)
    X_test, y_test = data["X"], data["y"]

    # Load and evaluate models
    logging.info("Loading models...")
    models = load_models(args.models_dir)
    logging.info(f"  Found {len(models)} model(s)")

    logging.info("Evaluating models...")
    results, models_data = [], {}
    for name, model in models.items():
        result = evaluate_model(name, model, X_test, y_test)
        if result is None:
            continue
        metrics, y_pred, y_proba = result
        results.append(metrics)
        models_data[name] = (metrics, y_pred, y_proba)

    # Create and sort DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(by="roc_auc", ascending=False)

    # Save results
    df.to_csv(os.path.join(args.output_dir, "model_metrics.csv"), index=False)
    with open(os.path.join(args.output_dir, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Determine models to plot
    if args.top_n is not None:
        top_models = df.head(args.top_n)["model"].tolist()
        filtered_models_data = {name: models_data[name] for name in top_models}
        df_plot = df.head(args.top_n)
    else:
        filtered_models_data = models_data
        df_plot = df

    # Generate plots
    plot_performance_bars(df_plot, args.output_dir)
    plot_metric_boxplots(df, args.output_dir)  # Use full df for boxplots
    plot_roc_pr(filtered_models_data, y_test, args.output_dir)
    plot_confusion_matrices(models_data, y_test, args.output_dir)  # Use full models_data
    plot_proba_distribution(filtered_models_data, args.output_dir)
    plot_calibration(filtered_models_data, y_test, args.output_dir)
    write_dashboard(df, args.output_dir)

    logging.info(f"Evaluation complete. Outputs in {args.output_dir}")

if __name__ == "__main__":
    main()