## Sentiment Pipeline v3.4 Evaluation Results

**Date:** August 4, 2025

**Model Path:** `outputs/20250804_155334/models/sentiment_pipeline_v3.4.pkl`

### Metrics

- **Accuracy:** 0.927
- **AUC:** 0.978

### Confusion Matrix

```
[[397796   63266]
 [ 38528  898152]]
```

### Classification Report

| Class            | Precision | Recall | F1-score | Support   |
| ---------------- | --------- | ------ | -------- | --------- |
| 0                | 0.91      | 0.86   | 0.89     | 461,062   |
| 1                | 0.93      | 0.96   | 0.95     | 936,680   |
| **Accuracy**     |           |        | **0.93** | 1,397,742 |
| **Macro avg**    | 0.92      | 0.91   | 0.92     | 1,397,742 |
| **Weighted avg** | 0.93      | 0.93   | 0.93     | 1,397,742 |

---

*These results reflect the performance of the calibrated Logistic Regression model (**`C=1`**) trained and evaluated on the hold-out test set.*

