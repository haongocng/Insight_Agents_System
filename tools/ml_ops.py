import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from itertools import combinations
from scipy.special import expit
import warnings

def fill_missing_values(data: pd.DataFrame, columns: Union[str, List[str]], method: str = 'auto', fill_value: Any = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns of a DataFrame.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if method == 'auto':
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == 'mean':
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == 'median':
            data[column].fillna(data[column].median(), inplace=True)
        elif method == 'mode':
            data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == 'constant':
            data[column].fillna(fill_value, inplace=True)
        else:
            raise ValueError("Invalid method. Choose from 'auto', 'mean', 'median', 'mode', or 'constant'.")

    return data

def remove_columns_with_missing_data(data: pd.DataFrame, thresh: float = 0.5, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Remove columns containing missing values from a DataFrame based on a threshold.
    """
    if not 0 <= thresh <= 1:
        raise ValueError("thresh must be between 0 and 1")

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        data_subset = data[columns]
    else:
        data_subset = data

    max_missing = int(thresh * len(data_subset))
    columns_to_keep = data_subset.columns[data_subset.isna().sum() < max_missing]

    if columns is not None:
        columns_to_keep = columns_to_keep.union(data.columns.difference(columns))

    return data[columns_to_keep]

def detect_and_handle_outliers_zscore(data: pd.DataFrame, columns: Union[str, List[str]], threshold: float = 3.0, method: str = 'clip') -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns using the Z-score method.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        mean = data[column].mean()
        std = data[column].std()
        z_scores = (data[column] - mean) / std

        if method == 'clip':
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            data.loc[z_scores > threshold, column] = upper_bound
            data.loc[z_scores < -threshold, column] = lower_bound
        elif method == 'remove':
            data = data[abs(z_scores) <= threshold]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return data

def detect_and_handle_outliers_iqr(data: pd.DataFrame, columns: Union[str, List[str]], factor: float = 1.5, method: str = 'clip') -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns using the Interquartile Range (IQR) method.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        if method == 'clip':
            data[column] = data[column].clip(lower_bound, upper_bound)
        elif method == 'remove':
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return data

def remove_duplicates(data: pd.DataFrame, columns: Union[str, List[str]] = None, keep: str = 'first', inplace: bool = False) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The 'data' argument must be a pandas DataFrame.")
        
        if columns is not None and not isinstance(columns, (str, list)):
            raise TypeError("The 'columns' argument must be a string, list of strings, or None.")
        
        if keep not in ['first', 'last', False]:
            raise ValueError("The 'keep' argument must be 'first', 'last', or False.")
        
        if not isinstance(inplace, bool):
            raise TypeError("The 'inplace' argument must be a boolean.")

        if inplace:
            data.drop_duplicates(subset=columns, keep=keep, inplace=True)
            return data
        else:
            return data.drop_duplicates(subset=columns, keep=keep)
    except Exception as e:
        raise RuntimeError(f"Error occurred while removing duplicates: {e}")

def convert_data_types(data: pd.DataFrame, columns: Union[str, List[str]], target_type: str) -> pd.DataFrame:
    """
    Convert the data type of specified columns in a DataFrame.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        if target_type == 'int':
            data[column] = pd.to_numeric(data[column], errors='coerce').astype('Int64')
        elif target_type == 'float':
            data[column] = pd.to_numeric(data[column], errors='coerce')
        elif target_type == 'str':
            data[column] = data[column].astype(str)
        elif target_type == 'bool':
            data[column] = data[column].astype(bool)
        elif target_type == 'datetime':
            data[column] = pd.to_datetime(data[column], errors='coerce')
        else:
            raise ValueError("Invalid target_type. Choose from 'int', 'float', 'str', 'bool', or 'datetime'.")

    return data

def format_datetime(data: pd.DataFrame, columns: Union[str, List[str]], format: str = '%Y-%m-%d %H:%M:%S', errors: str = 'coerce') -> pd.DataFrame:
    """
    Format datetime columns in a DataFrame to a specified format.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        data[column] = pd.to_datetime(data[column], errors=errors)
        data[column] = data[column].dt.strftime(format)

    return data

def one_hot_encode(data: pd.DataFrame, columns: Union[str, List[str]], drop_original: bool = False, handle_unknown: str = 'error') -> pd.DataFrame:
    """
    Perform one-hot encoding on specified categorical columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    encoded = encoder.fit_transform(data[columns])
    
    new_columns = [f"{col}_{val}" for col, vals in zip(columns, encoder.categories_) for val in vals]
    encoded_df = pd.DataFrame(encoded, columns=new_columns, index=data.index)
    
    result = pd.concat([data, encoded_df], axis=1)
    
    if drop_original:
        result = result.drop(columns, axis=1)
    
    return result

def label_encode(data: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Perform label encoding on specified categorical columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    result = data.copy()
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    for col in columns:
        col_data = data[col]
        if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            encoder = LabelEncoder()
            encoded_col_name = f"{col}_encoded"
            result[encoded_col_name] = encoder.fit_transform(col_data.astype(str))
        else:
            warnings.warn(f"Column '{col}' is {col_data.dtype}, which is not categorical. Skipping encoding.")

    return result

def frequency_encode(data: pd.DataFrame, columns: Union[str, List[str]], drop_original: bool = False) -> pd.DataFrame:
    """
    Perform frequency encoding on specified categorical columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    result = data.copy()
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    for col in columns:
        col_data = data[col]
        frequency = col_data.value_counts(normalize=True)
        encoded_col_name = f"{col}_freq"
        result[encoded_col_name] = col_data.map(frequency)

    return result

def target_encode(data: pd.DataFrame, columns: Union[str, List[str]], target: str, min_samples_leaf: int = 1, smoothing: float = 1.0) -> pd.DataFrame:
    """
    Perform target encoding on specified categorical columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    if min_samples_leaf < 0:
        raise ValueError(f"min_samples_leaf should be non-negative, but got {min_samples_leaf}.")
    
    if smoothing <= 0:
        raise ValueError(f"smoothing should be positive, but got {smoothing}.")

    result = data.copy()
    prior = data[target].mean()
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    for col in columns:
        col_data = data[col]
        averages = data.groupby(col)[target].agg(["count", "mean"])
        smoothing_factor = expit((averages["count"] - min_samples_leaf) / smoothing)
        averages["smooth"] = prior * (1 - smoothing_factor) + averages["mean"] * smoothing_factor
        encoded_col_name = f"{col}_target_enc"
        result[encoded_col_name] = col_data.map(averages["smooth"]).fillna(prior)

    return result

def correlation_feature_selection(data: pd.DataFrame, target: str, method: str = 'pearson', threshold: float = 0.5) -> pd.DataFrame:
    """
    Perform feature selection based on correlation analysis.
    """
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    X = data.drop(columns=[target])
    y = data[target]

    if method == 'spearman':
        corr_matrix, _ = spearmanr(X, y)
        corr_with_target = pd.Series(corr_matrix[-1][:-1], index=X.columns)
    else:
        corr_with_target = X.apply(lambda x: x.corr(y, method=method))

    selected_features = corr_with_target[abs(corr_with_target) > threshold]

    return pd.DataFrame({
        'feature': selected_features.index,
        'correlation': selected_features.values
    }).sort_values('correlation', key=abs, ascending=False)

def variance_feature_selection(data: pd.DataFrame, threshold: float = 0.0, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Perform feature selection based on variance analysis.
    """
    if columns is None:
        columns = data.columns
    elif isinstance(columns, str):
        columns = [columns]

    X = data[columns]
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    feature_mask = selector.get_support()
    variances = selector.variances_

    selected_features = pd.DataFrame({
        'feature': X.columns[feature_mask],
        'variance': variances[feature_mask]
    }).sort_values('variance', ascending=False)

    return selected_features

def scale_features(data: pd.DataFrame, columns: Union[str, List[str]], method: str = 'standard', copy: bool = True) -> pd.DataFrame:
    """
    Scale numerical features in the specified columns of a DataFrame.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    unique_columns = []
    for col in columns:
        if col in unique_columns:
            continue
        col_data = data[col]
        if isinstance(col_data, pd.DataFrame):
            if col_data.nunique().eq(1).all():
                print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be scaled.")
                unique_columns.append(col)
            else:
                raise ValueError(f"Duplicate non-identical columns found for '{col}'. Please resolve this before scaling.")
        else:
            unique_columns.append(col)

    non_numeric_cols = [col for col in unique_columns if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric_cols:
        raise ValueError(f"The following columns are not numerical: {non_numeric_cols}. Please only specify numerical columns for scaling.")

    if method == 'standard':
        scaler = StandardScaler(copy=copy)
    elif method == 'minmax':
        scaler = MinMaxScaler(copy=copy)
    elif method == 'robust':
        scaler = RobustScaler(copy=copy)
    else:
        raise ValueError("Invalid method. Choose 'standard', 'minmax', or 'robust'.")

    if copy:
        data = data.copy()

    scaled_data = scaler.fit_transform(data[unique_columns])
    data[unique_columns] = scaled_data

    return data

def perform_pca(data: pd.DataFrame, n_components: Union[int, float, str] = 0.95, columns: Union[str, List[str]] = None, scale: bool = True) -> pd.DataFrame:
    """
    Perform Principal Component Analysis (PCA) on the specified columns.
    """
    if columns is None:
        columns = data.columns
    elif isinstance(columns, str):
        columns = [columns]

    X = data[columns]
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    if not non_numeric_cols.empty:
        raise ValueError(f"Non-numeric data types detected in columns: {list(non_numeric_cols)}.")

    if (X.std() > 10).any():
        warnings.warn("Some features have high standard deviations. Consider scaling.")

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )

    return pca_df

def perform_rfe(data: pd.DataFrame, target: Union[str, pd.Series], n_features_to_select: Union[int, float] = 0.5, step: int = 1, estimator: str = 'auto', columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Perform Recursive Feature Elimination (RFE).
    """
    if isinstance(target, str):
        y = data[target]
        X = data.drop(columns=[target])
    else:
        y = target
        X = data

    if columns:
        if isinstance(columns, str):
            columns = [columns]
        X = X[columns]

    if isinstance(n_features_to_select, float):
        n_features_to_select = max(1, int(n_features_to_select * X.shape[1]))

    is_continuous = np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10

    if estimator == 'auto':
        estimator = 'linear' if is_continuous else 'logistic'

    if estimator == 'logistic':
        est = LogisticRegression(random_state=42)
    elif estimator == 'rf':
        est = RandomForestClassifier(random_state=42)
    elif estimator == 'linear':
        est = LinearRegression()
    elif estimator == 'rf_regressor':
        est = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Invalid estimator. Choose 'auto', 'logistic', 'rf', 'linear', or 'rf_regressor'.")

    rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_].tolist()

    return data[selected_features]

def create_polynomial_features(data: pd.DataFrame, columns: Union[str, List[str]], degree: int = 2, interaction_only: bool = False, include_bias: bool = False) -> pd.DataFrame:
    """
    Create polynomial features from specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    if degree < 1:
        raise ValueError("Degree must be at least 1.")

    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    unique_columns = []
    for col in columns:
        if col in unique_columns:
            continue
        col_data = data[col]
        if isinstance(col_data, pd.DataFrame):
            if col_data.nunique().eq(1).all():
                print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be used.")
                unique_columns.append(col)
            else:
                raise ValueError(f"Duplicate non-identical columns found for '{col}'.")
        else:
            unique_columns.append(col)

    for col in unique_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' is not numeric.")

    X = data[unique_columns]
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    poly_features = poly.fit_transform(X)

    feature_names = poly.get_feature_names_out(unique_columns)
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
    poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]

    result = pd.concat([data, poly_df], axis=1)

    if result.shape[1] > 1000:
        warnings.warn("Resulting DataFrame has over 1000 columns.")

    return result

def create_feature_combinations(data: pd.DataFrame, columns: Union[str, List[str]], combination_type: str = 'multiplication', max_combination_size: int = 2) -> pd.DataFrame:
    """
    Create feature combinations from specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    unique_columns = []
    for col in columns:
        if col in unique_columns:
            continue
        col_data = data[col]
        if isinstance(col_data, pd.DataFrame):
            if col_data.nunique().eq(1).all():
                print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be used.")
                unique_columns.append(col)
            else:
                raise ValueError(f"Duplicate non-identical columns found for '{col}'.")
        else:
            unique_columns.append(col)

    for col in unique_columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' is not numeric.")

    if max_combination_size < 2:
        raise ValueError("max_combination_size must be at least 2.")

    if combination_type not in ['multiplication', 'addition']:
        raise ValueError("combination_type must be either 'multiplication' or 'addition'.")

    result = data.copy()

    for r in range(2, min(len(unique_columns), max_combination_size) + 1):
        for combo in combinations(unique_columns, r):
            if combination_type == 'multiplication':
                new_col = result[list(combo)].prod(axis=1)
                new_col_name = ' * '.join(combo)
            else:
                new_col = result[list(combo)].sum(axis=1)
                new_col_name = ' + '.join(combo)
            
            result[new_col_name] = new_col

    if result.shape[1] > 1000:
        warnings.warn("Resulting DataFrame has over 1000 columns.")

    return result

# def train_and_validation_and_select_the_best_model(X, y, problem_type='binary', selected_models=['XGBoost', 'SVM', 'random forest']):
#     # Split data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Define models and their hyperparameter grids
#     if problem_type in ['binary', 'multiclass']:
#         models = {
#             'logistic regression': (LogisticRegression(max_iter=1000), {
#                 'C': [0.1, 1, 10],
#                 'solver': ['saga'],
#                 'penalty': ['l1', 'l2', 'elasticnet'],
#                 'l1_ratio': [0.5],
#             }),
#             'decision tree': (DecisionTreeClassifier(), {
#                 'max_depth': [None, 5, 15, 20],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]
#             }),
#             'random forest': (RandomForestClassifier(), {
#                 'n_estimators': [10, 50],
#                 'max_depth': [None, 5, 10],
#                 'min_samples_split': [2, 5, 10]
#             }),
#             'XGBoost': (GradientBoostingClassifier(), {
#                 'n_estimators': [50, 100],
#                 'learning_rate': [0.01, 0.1],
#                 'max_depth': [3, 5, 7]
#             }),
#             'SVM': (SVC(), {
#                 'C': [0.1, 1],
#                 'kernel': ['linear', 'rbf'],
#                 'gamma': ['scale', 'auto']
#             })
#         }
#         scoring = 'accuracy' if problem_type == 'binary' else 'f1_weighted'
#     elif problem_type == 'regression':
#         models = {
#             'linear regression': (LinearRegression(), {
#                 'fit_intercept': [True, False],
#                 'copy_X': [True, False]
#             }),
#             'decision tree': (DecisionTreeRegressor(), {
#                 'max_depth': [None, 5, 10, 15, 20],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]
#             }),
#             'random forest': (RandomForestRegressor(), {
#                 'n_estimators': [10, 50, 100],
#                 'max_depth': [None, 5, 10, 20],
#                 'min_samples_split': [2, 5, 10]
#             }),
#             'XGBoost': (GradientBoostingRegressor(), {
#                 'n_estimators': [50, 100],
#                 'learning_rate': [0.01, 0.1],
#                 'max_depth': [3, 5, 7]
#             }),
#             'SVM': (SVR(), {
#                 'C': [0.1, 1, 10],
#                 'kernel': ['linear', 'rbf'],
#                 'gamma': ['scale', 'auto']
#             })
#         }
#         scoring = 'neg_mean_squared_error'
#     else:
#         raise ValueError("Invalid problem_type. Choose from 'binary', 'multiclass', or 'regression'.")

#     best_model = None
#     best_score = float('-inf') if problem_type in ['binary', 'multiclass'] else float('inf')
#     results = {}

#     models = {model_name: models[model_name] for model_name in selected_models}
#     # Hyperparameter optimization
#     for model_name, (model, param_grid) in models.items():
#         optimizer = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scoring)
#         optimizer.fit(X_train, y_train)
#         print(f"Finished model training: {model_name}")
        
#         # Evaluate the model on the validation set
#         y_pred = optimizer.predict(X_val)
#         if problem_type in ['binary', 'multiclass']:
#             score = accuracy_score(y_val, y_pred) if problem_type == 'binary' else f1_score(y_val, y_pred, average='weighted')
#         else:
#             score = -mean_squared_error(y_val, y_pred)

#         # Store the results
#         results[model_name] = {
#             'best_params': optimizer.best_params_,
#             'score': score
#         }

#         if (problem_type in ['binary', 'multiclass'] and score > best_score) or \
#            (problem_type == 'regression' and score < best_score):
#             best_score = score
#             best_model = optimizer.best_estimator_

#     # Output results
#     for model_name, result in results.items():
#         print(f"Model: {model_name}, Best Params: {result['best_params']}, Score: {result['score']}")

#     return best_model

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV

def train_and_validation_and_select_the_best_model_4Classification(
    X, 
    y, 
    selected_models=['RandomForest', 'XGBoost', 'SVM']
):
    """
    Train and select best model for CLASSIFICATION tasks.
    Returns: (best_model, metrics_df)
    """
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, pd.DataFrame()

    models = {
        'LogisticRegression': (LogisticRegression(max_iter=2000), {'C': [0.1, 1, 10], 'solver': ['saga']}),
        'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [5, 10, 20], 'min_samples_split': [5, 10]}),
        'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [10, 20]}),
        'XGBoost': (GradientBoostingClassifier(), {'n_estimators': [50, 100], 'learning_rate': [0.1], 'max_depth': [3, 5]}),
        'SVM': (SVC(probability=True), {'C': [1, 10], 'kernel': ['rbf']})
    }

    # 3. Filter models
    available_models_lower = {k.lower(): v for k, v in models.items()}
    models_to_run = {}
    for name in selected_models:
        if name.lower() in available_models_lower:
            original_key = [k for k in models.keys() if k.lower() == name.lower()][0]
            models_to_run[original_key] = models[original_key]
    
    if not models_to_run: 
        models_to_run = models

    best_model = None
    best_f1 = float('-inf')
    results_list = []

    print(f"Starting CLASSIFICATION training for {len(models_to_run)} models...")

    for model_name, (model, param_grid) in models_to_run.items():
        try:
            # Optimize based on F1-Weighted
            optimizer = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            optimizer.fit(X_train, y_train)
            
            best_estimator = optimizer.best_estimator_
            y_pred = best_estimator.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'Model': model_name,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
                'Best Params': str(optimizer.best_params_)
            }
            results_list.append(metrics)
            print(f"Finished: {model_name} | F1-Score: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = best_estimator
                
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    if not results_list:
        print("ERROR: No models were trained successfully.")
        return None, pd.DataFrame()

    metrics_df = pd.DataFrame(results_list)
    
    print("\n--- Classification Report ---")
    print(metrics_df)
    
    return best_model, metrics_df


def train_and_validation_and_select_the_best_model_4Regression(
    X, 
    y, 
    selected_models=['RandomForest', 'XGBoost', 'LinearRegression']
):
    """
    Train and select best model for REGRESSION tasks.
    Returns: (best_model, metrics_df)
    """
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, pd.DataFrame()

    models = {
        'LinearRegression': (LinearRegression(), {'fit_intercept': [True, False]}),
        'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [5, 10, 20]}),
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [10, 20]}),
        'XGBoost': (GradientBoostingRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.1]}),
        'SVM': (SVR(), {'C': [1, 10], 'kernel': ['rbf']})
    }

    available_models_lower = {k.lower(): v for k, v in models.items()}
    models_to_run = {}
    for name in selected_models:
        if name.lower() in available_models_lower:
            original_key = [k for k in models.keys() if k.lower() == name.lower()][0]
            models_to_run[original_key] = models[original_key]

    if not models_to_run: 
        models_to_run = models

    best_model = None
    best_mse = float('inf')
    results_list = []

    print(f"Starting REGRESSION training for {len(models_to_run)} models...")

    for model_name, (model, param_grid) in models_to_run.items():
        try:
            optimizer = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            optimizer.fit(X_train, y_train)
            
            best_estimator = optimizer.best_estimator_
            y_pred = best_estimator.predict(X_val)
            
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            metrics = {
                'Model': model_name,
                'MSE': round(mse, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R2': round(r2, 4),
                'Best Params': str(optimizer.best_params_)
            }
            results_list.append(metrics)
            print(f"Finished: {model_name} | RMSE: {rmse:.4f}")
            
            if mse < best_mse:
                best_mse = mse
                best_model = best_estimator
                
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    if not results_list:
        print("ERROR: No models were trained successfully.")
        return None, pd.DataFrame()
        
    metrics_df = pd.DataFrame(results_list)

    print("\n--- Regression Report ---")
    print(metrics_df)
    
    return best_model, metrics_df