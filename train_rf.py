import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_rf_nested_gkf(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    group_col: str = "subject_id",
    n_splits_outer: int = 5,
    n_splits_inner: int = 3,
    decision_threshold: float = 0.5,
    random_state: int = 42,
):
    """
    Trains a Random Forest classifier using Nested Stratified Group K-Fold Cross-Validation.
    Prevents data leakage by encapsulating scaling inside an sklearn Pipeline for inner CV.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features, target labels, and group IDs.
    feature_cols : list[str]
        List of feature column names.
    label_col : str, optional (default='label')
        Target label column name.
    group_col : str, optional (default='subject_id')
        Group/Subject ID column name for GroupKFold splitting.
    n_splits_outer : int, optional (default=5)
        Number of outer folds for model evaluation.
    n_splits_inner : int, optional (default=3)
        Number of inner folds for hyperparameter tuning.
    decision_threshold : float, optional (default=0.5)
        Probability threshold for positive class classification.
    random_state : int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    --------
    fold_results : list[dict]
        List of metric dictionaries for each outer fold.
    fold_models : list[Pipeline]
        List of fitted Pipeline objects (Scaler + RF) for each outer fold.
    pooled_data : dict
        Dictionary containing overall true labels, predicted labels, and probabilities across folds.
    """
    X = df[feature_cols].values
    y = df[label_col].values
    groups = df[group_col].values

    outer_cv = StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state
    )

    # Pipeline avoids data leakage during inner CV grid search
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=random_state)),
        ]
    )

    param_grid = {
        "rf__n_estimators": [100],
        "rf__max_depth": [5, 10],
        "rf__min_samples_leaf": [5, 10],
        "rf__class_weight": ["balanced"],
    }

    fold_results = []
    fold_models = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    print(f"Starting {n_splits_outer}-Fold Outer / {n_splits_inner}-Fold Inner Nested Stratified Group K-Fold CV...\n")

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        if len(np.unique(y_test)) < 2:
            print(f"Fold {fold_i+1}: skipped (only one class present in test set)")
            continue

        inner_cv = StratifiedGroupKFold(
            n_splits=n_splits_inner, shuffle=True, random_state=random_state
        )
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=1,
        )

        grid.fit(X_train, y_train, groups=groups_train)

        best_pipeline = grid.best_estimator_
        
        # Predict on outer test fold
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= decision_threshold).astype(int)

        # Compute metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Fold {fold_i+1}:")
        print(f"  AUC = {auc:.3f} | Accuracy = {acc:.3f} | Precision = {prec:.3f} | Recall = {rec:.3f} | F1 = {f1:.3f}")
        print(f"  Best params: {grid.best_params_}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

        fold_results.append(
            {
                "auc": auc,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
        fold_models.append(best_pipeline)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_pred_proba)

    if len(fold_results) > 0:
        results_df = pd.DataFrame(fold_results)
        print("=" * 60)
        print("OVERALL MEAN METRICS (across outer folds):")
        print(results_df.mean().round(3))
        print("\nOverall Classification Report (pooled across outer folds):")
        print(
            classification_report(
                all_y_true,
                all_y_pred,
                target_names=["Normotensive", "Hypertensive"],
                zero_division=0,
            )
        )
        print("Overall Confusion Matrix (pooled):")
        print(confusion_matrix(all_y_true, all_y_pred))

    pooled_data = {
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_proba": np.array(all_y_proba),
    }

    return fold_results, fold_models, pooled_data


if __name__ == "__main__":
    # Example usage with synthetic dataset matching PPG / subject structure
    np.random.seed(42)
    n_samples = 500
    n_subjects = 50

    sample_df = pd.DataFrame({
        "subject_id": np.random.choice([f"sub_{i:02d}" for i in range(n_subjects)], size=n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "label": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    })

    feats = ["feature_1", "feature_2", "feature_3"]
    results, models, pooled = train_rf_nested_gkf(sample_df, feats)
