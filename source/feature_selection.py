import numpy as np



def remove_correlated_features(df, corr_threshold=0.8):
    """
    Remove correlated features such that only features with a pearson 
    correlation coeffcient of less than 'corr_threshold' remain in the dataframe.
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with features.
    non_feature_columns : list of strings
        List of columns which do not contain features.
    corr_threshold: float
        If two features have a pearson correlation index above the 
        corr_threshold, one of the features will be dropped from the 
        dataframe.
   
    Returns
    -------
    df : pandas dataframe
        Dataframe without heavily correlated feautures (less than 'corr_threshold').
    dropped_features : list of strings
        List of features that have been dropped.
    """
    
    print("Remove correlated features...")

    # Check corr_threshold
    if type(corr_threshold) is float:
        if 0 < corr_threshold and corr_threshold < 1:
            print("Correlation threshold:", corr_threshold)
        else:
            print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
            return -1
    else:
        print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
        return -1
    
    # Get names and number of all features (columns)
    all_features = df.columns.to_list()
    n_features   = df.shape[1]

    # Compute correlation matrix
    corr = df.corr(method='pearson')
    corr = corr.abs()

    # Keep only correlation values in the upper traingle matrix
    triu = np.triu(np.ones(corr.shape), k=1)
    triu = triu.astype(np.bool)
    corr = corr.where(triu)
    
    # Select columns which will be dropped from dataframe
    cols_to_drop = [column for column in corr.columns if any(corr[column] > corr_threshold)]

    n_cols_to_drop = len(cols_to_drop)
    p_cols_to_drop = 100 * (float(n_cols_to_drop) / float(n_features))
    p_cols_to_drop = np.round(p_cols_to_drop, decimals=1)
    print("Drop", n_cols_to_drop, "/", n_features, " features (", p_cols_to_drop, "%).")

    # Drop colums
    df = df.drop(cols_to_drop, axis=1)
    
    # Find names of features which have been dropped
    uncorrelated_features = df.columns.to_list()
    dropped_features      = list(set(all_features) - set(uncorrelated_features))
    
    return df, dropped_features