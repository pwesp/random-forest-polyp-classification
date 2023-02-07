import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from   sklearn.inspection import permutation_importance
from   sklearn.metrics import make_scorer, roc_auc_score



def extract_feature_importance(sklearn_model):
    """
    Extract feature importance from trained model.
    """
    importances     = sklearn_model.feature_importances_
    importances_std = []
    
    # Caluclate uncertainty of feature importance estimates
    importances_estimators = [tree.feature_importances_ for tree in sklearn_model.estimators_]
    importances_estimators = np.asarray(importances_estimators)

    # Loop through single estimators of the model
    for feature_idx in range(importances_estimators.shape[1]):
        importances_est    = importances_estimators[:,feature_idx]
        importances_est    = importances_est[importances_est!=0]
        importance_est_std = 0
        if importances_est.shape[0] != 0:
            importance_est_std = np.std(importances_est)
        importances_std.append(importance_est_std)
    
    importances_std = np.asarray(importances_std)
    
    return importances, importances_std



def rank_feature_importance(importances, feature_names, n=15):
    """
    Rank feauture importances.
    """
    if n==-1:
        n = len(feature_names)
        
    # Turn lists into numpy arrays
    importances     = np.asarray(importances)

    # Sort features by importance
    sorting_indices = np.argsort(importances)[::-1]
    #print("\nFeature ranking across cross-validation folds:")
    for f in range(n):
        print("%2.d. feature %3.d: %s (%f)" % (f + 1, sorting_indices[f], feature_names[sorting_indices[f]], importances[sorting_indices[f]]))
        
    # Create lists for feature importance plot
    feature_ids_ranked   = [sorting_indices[f] for f in range(n)]
    feature_names_ranked = [feature_names[sorting_indices[f]] for f in range(n)]
    importances_ranked   = [importances[sorting_indices[f]] for f in range(n)]

    return feature_ids_ranked, feature_names_ranked, importances_ranked



def plot_feature_importance(features, importances):
    sns.set_theme(style="whitegrid")
    
    # Turn plot data into dataframe
    df_plot = pd.DataFrame(data=np.array([features, importances]).swapaxes(0,1), columns=["feature", "importance"])
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10,10))
    sns.barplot(x="feature", y="importance", data=df_plot, order=features, color='lightgray')
    ax.set_xticks(np.arange(0, len(features)))
    ax.set_xticklabels(features)
    ax.tick_params(axis='x', labelsize=16, size=4)
    ax.tick_params(axis='y', labelsize=16, size=4)
    ax.set_xlabel('feature', fontsize=16)
    ax.set_ylabel('importance', fontsize=16)
    plt.tight_layout()
    plt.show()