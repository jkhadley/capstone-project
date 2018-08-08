import keras.backend as K 

def recall(y_true,y_pred):
    """Calculates the recall of the prediction.

    This was copied from keras before they removed it from their default metrics.

    Parameters
    ----------
    y_true : np.array
        The true output for the given input
    y_pred : np.array
        The predicted output for the given input
    
    Returns
    -------
    recall : float
        The recall for the inputs.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true,y_pred):
    """Calculates the precision of the prediction.

    This was copied from keras before they removed it from their default metrics.

    Parameters
    ----------
    y_true : np.array
        The true output for the given input
    y_pred : np.array
        The predicted output for the given input
    
    Returns
    -------
    precision : float
        The precision for the inputs.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1Score(y_true,y_pred):
    """Calculates the f1-score for the prediction.

    This was copied from keras before they removed it from their default metrics.

    Parameters
    ----------
    y_true : np.array
        The true output for the given input
    y_pred : np.array
        The predicted output for the given input
    
    Returns
    -------
    f1-score : float
        The f1-score for the inputs.
    """
    p = precision(y_true,y_pred)
    r = recall(y_true,y_pred)
    return 2*p*r/(p+r)

def RMSE(y_true,y_pred):    
    """Calculates the root mean squared error for the prediction.

    This was copied from keras before they removed it from their default metrics.

    Parameters
    ----------
    y_true : np.array
        The true output for the given input
    y_pred : np.array
        The predicted output for the given input
    
    Returns
    -------
    RSME : float
        The Root Mean Squared Error for the inputs.
    """
    return K.sqrt(K.mean(K.square(y_true-y_pred)))