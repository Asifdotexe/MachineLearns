import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def check_imbalance(df, class_column='class'):
    no_of_true = len(df.loc[df[class_column] == True])
    no_of_false = len(df.loc[df[class_column] == False])

    true_ratio = (no_of_true / (no_of_true + no_of_false))
    false_ratio = (no_of_false / (no_of_false + no_of_true))

    print(f"Number of true: {no_of_true} ({round(true_ratio, 4) * 100}%)")
    print(f"Number of false: {no_of_false} ({round(false_ratio, 5) * 100}%)")

def random_forest_tuning(x_train, x_test, y_train, y_test, n, d, l, seed):

    """
    Evaluate the performance of a Random Forest model with different hyperparameters.

    Parameters:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - n: List of the number of trees in the forest
    - d: List of the maximum depth of the tree
    - l: List of the minimum samples required to be at a leaf node
    - seed: Random seed for reproducibility

    Returns:
    - model_performance_df: DataFrame containing the model performance metrics
    """
    model_performance = []

    for i in n:
        for j in d:
            for k in l:
                # Create and train the Random Forest model
                random_forest = RandomForestClassifier(
                    n_estimators=i,
                    max_depth=j,
                    min_samples_leaf=k,
                    random_state=seed
                )
                random_forest.fit(x_train, y_train.ravel())

                # Predict probabilities on the training and testing sets
                train_pred = random_forest.predict_proba(x_train)
                test_pred = random_forest.predict_proba(x_test)

                # Create a unique identifier for the current set of hyperparameters
                t1 = f'trees{i}_maxDepth{j}_minLeaf{k}'

                # Calculate and store AUC-ROC scores for training and testing sets
                t2 = [t1, round(roc_auc_score(y_train, train_pred[:, 1]), 4), round(roc_auc_score(y_test, test_pred[:, 1]), 4)]
                model_performance.append(t2)

    # Create a DataFrame from the collected performance metrics
    model_performance_df = pd.DataFrame(model_performance)
    model_performance_df.rename(columns={0: 'parameter', 1: 'train_auc', 2: 'test_auc'}, inplace=True)

    return model_performance_df

def get_feature_importance(X_train, model, top_n=5):
    """
    Get feature importances from a trained model and return the top N features.

    Parameters:
    - X_train: DataFrame containing the training features
    - model: Trained model with a `feature_importances_` attribute
    - top_n: Number of top features to retrieve (default is 5)

    Returns:
    - imp_feat_df: DataFrame with the top N features and their importances
    """
    feature_importances = model.feature_importances_

    t1 = []
    for i in range(len(X_train.columns)):
        t2 = [X_train.columns[i], feature_importances[i]]
        t1.append(t2)

    imp_feat_df = pd.DataFrame(t1, columns=['name', 'importance'])
    imp_feat_df = imp_feat_df.sort_values(by=['importance'], ascending=False).head(top_n)

    return imp_feat_df

def print_categorical_value_counts(df, cat_cols):
    """
    Print value counts for each categorical column in the DataFrame.

    Parameters:
    - df: DataFrame
    - cat_cols: List of categorical column names

    Returns:
    - None
    """
    for col in cat_cols:
        print("=" * 15)
        print(col)
        print("--" * 5)
        print(df[col].value_counts())

import pandas as pd

def perform_eda(file_path, num_sheets=None):
    try:
        if file_path.endswith('.xlsx'):
            # Excel file
            if num_sheets is not None:
                # EDA for each sheet
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                results = {}
                
                for sheet_name in sheet_names[:num_sheets]:
                    df = pd.read_excel(file_path, sheet_name)
                    eda_results = {
                        "Sheet Name": sheet_name,
                        "Shape": df.shape,
                        "Columns": df.columns.tolist(),
                        "Info": df.info(),
                        "Missing Values": df.isnull().sum()
                    }
                    results[sheet_name] = eda_results
                
                return results
            else:
                raise ValueError("Number of sheets not provided for Excel file.")
        elif file_path.endswith(('.csv', '.json')):
            # EDA for CSV or JSON file
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_json(file_path)
            eda_results = {
                "File Name": file_path,
                "Shape": df.shape,
                "Columns": df.columns.tolist(),
                "Info": df.info(),
                "Missing Values": df.isnull().sum()
            }
            return eda_results
        else:
            raise ValueError("Unsupported file format. Please provide an Excel, CSV, or JSON file.")
    except Exception as e:
        return f"Error performing EDA: {str(e)}"

# Example usage:
file_path = 'your_file.xlsx'  # Replace with your file path
num_sheets = 2  # Replace with the desired number of sheets for Excel files
eda_results = perform_eda(file_path, num_sheets)
print(eda_results)