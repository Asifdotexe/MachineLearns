import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
