import pandas as pd
import optuna
import joblib
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error


optuna.logging.set_verbosity(optuna.logging.WARNING)


def rename_columns(df: pd.DataFrame, renaming_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Renames columns in a DataFrame based on a provided dictionary.

    Args:
        df: The input DataFrame
        renaming_dict: Dictionary with column names to be renamed as keys and new names as values

    Returns:
        DataFrame with renamed columns

    """
    return df.rename(columns=renaming_dict)
def get_features(df: pd.DataFrame, lag_params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Create lag features for time-series data.

    Generates lagged versions of specified columns, useful for capturing
    temporal patterns in time-series forecasting. Missing values at the
    beginning of each lagged column are backfilled.

    Args:
        df: Input DataFrame containing the original features.
        lag_params: Dictionary mapping feature names to lists of lag values.
            Example: {"temperature": [1, 2, 3]} creates columns
            "temperature_lag_1", "temperature_lag_2", "temperature_lag_3".

    Returns:
        DataFrame with original columns plus new lag feature columns.
    """
    for feature, lags in lag_params.items():
        for lag in lags:
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag).bfill()
    timestamps = pd.to_datetime(df['datetime'])
    df.drop(columns=['datetime'], inplace=True)
    print(df.columns)
    return df, timestamps

def make_target(df: pd.DataFrame, target_params: Dict[str, Any]) -> pd.DataFrame:
    """Create target column by shifting."""
    df[target_params["new_target_name"]] = (
        df[target_params["target_column"]].shift(-target_params["shift_period"]).ffill()
    )
    return df


def split_data(
    df: pd.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets."""
    # Get target column name
    target_name = params["target_params"]["new_target_name"]
    # Get features columns names
    features = [col for col in df.columns if col != target_name]
    # Split data into train/test sets
    x, y = df[features], df[target_name]
    train_size = int(params["train_fraction"] * len(df))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print("X_train_shape", x_train.shape)
    print("X_test_shape", x_test.shape)
    print("y_train_shape", y_train.shape)
    print("y_test_shape", y_test.shape)
    return x_train, x_test, y_train, y_test


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Optuna tuning for CatBoost and RandomForest, return best params and model type."""

    def catboost_objective(trial):
        p = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 5.0),
            'iterations': trial.suggest_int('iterations', 100, 500),
            'loss_function': 'RMSE',
            'verbose': 0,
            'random_seed': 42,
            'allow_writing_files': False,
        }
        model = CatBoostRegressor(**p)
        model.fit(x_train, y_train)
        return np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

    def rf_objective(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42,
        }
        model = RandomForestRegressor(**p)
        model.fit(x_train, y_train)
        return np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

    n_trials = params.get('n_trials', 1)

    catboost_study = optuna.create_study(direction='minimize')
    catboost_study.optimize(catboost_objective, n_trials=n_trials)

    rf_study = optuna.create_study(direction='minimize')
    rf_study.optimize(rf_objective, n_trials=n_trials)

    print(f"CatBoost best RMSE: {catboost_study.best_value:.2f}")
    print(f"RandomForest best RMSE: {rf_study.best_value:.2f}")

    if catboost_study.best_value <= rf_study.best_value:
        print("Winner: CatBoost")
        return {'model_type': 'catboost', 'model_params': catboost_study.best_params}
    else:
        print("Winner: RandomForest")
        return {'model_type': 'random_forest', 'model_params': rf_study.best_params}


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
) -> Any:
    """Train the winning model with the best hyperparameters."""
    model_type = best_params['model_type']
    params = best_params['model_params']

    if model_type == 'catboost':
        model = CatBoostRegressor(**params, loss_function='RMSE', verbose=0,
                                  random_seed=42, allow_writing_files=False)
    else:
        model = RandomForestRegressor(**params, random_state=42)

    model.fit(x_train, y_train)
    print(f"Trained {model_type} with params: {params}")
    return model, model_type

def predict(
    model: Any,
    x: pd.DataFrame,
) -> pd.DataFrame:
    """Predict using a trained model."""
    y_pred = pd.DataFrame(model.predict(x), columns=["prediction"])
    return y_pred


def compute_metrics(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true + 1e-8)) * 100
    
    metrics = {
        'MAE': float(round(mae, 2)),
        'RMSE': float(round(rmse, 2)),
        'MAPE': float(round(mape, 2)),
    }
    print(f"Metrics {metrics}")
    return metrics

def save_model(
    model: Any,
    model_type: str,
    best_params: Dict[str, Any],
    model_storage: Dict[str, Any],
) -> None:
    """Persist the trained model to disk.

    Uses model-specific serialization when available:
    - CatBoost: native .cbm format
    - Other models: joblib .pkl format

    Args:
        model: Trained model instance.
        model_type: Type of model (for determining save format).
        model_storage: Dictionary containing:
            - path: Directory path where to save the model.
            - name: Model file name (without extension).
    """
    model_dir = Path(model_storage["path"])
    model_name = model_storage["name"]

    if model_type in ["catboost", "cb"]:
        # CatBoost has native serialization
        catboost_path = model_dir / f"{model_name}.cbm"
        catboost_path.unlink(missing_ok=True)  # Deletes file if it already exists
        model.save_model(str(catboost_path))
    else:
        # Use joblib for sklearn models
        joblib_path = model_dir / f"{model_name}.pkl"
        joblib_path.unlink(missing_ok=True)  # Deletes file if it already exists
        joblib.dump(model, joblib_path)
    return None