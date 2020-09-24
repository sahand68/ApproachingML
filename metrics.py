## in general use sklearn.metrics for these

def accuracy(y_true, y_pred):

    correct_counter = 0
    for yt , yp in zip(y_true, y_pred):
        if yt==yp:
            correct_counter+=1

    return correct_counter/len(y_true)

def true_positive(y_true, y_pred):

    tp = 0
    for yt ,yp in zip(y_true, y_pred):
        if yt == 1 and y_pred == 1:
            tp+=1
    return tp

def true_negative(y_true, y_pred):

    tn = 0
    for yt ,yp in zip(y_true, y_pred):
        if yt == 0 and y_pred == 0:
            tn+=1
    return tp

def false_positve(y_true,y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp ==1:
            fp +=1
    return fp

def false_negative(y_true, y_pred):

    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp  == 0 :
            fn +=1
    return fn

def accuracy_v2(y_true, y_pred):

    tp = true_positive(y_true,y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positve(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    accuracy_score = (tp+tn) / (tp +tn + fp + fn )

    return accuracy_score

def percision(y_true, y_pred):

    tp = true_positive(y_true, y_pred)
    fp = false_positve(y_true, y_pred)
    percision = tp /(tp+fp)
    return percision

def recall (y_true, y_pred):

    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

def log_loss(y_true, y_proba):
    for yt , yp in zip(y_true, y_proba):
        #0 gets converted to 1e-15
        #1 gets converted to 1-1e-15

        yp= np.clip(yp, epsilon, 1-epsilon)

        temp_los = -1.0*(yt*np.log(yp)+(1-yt)+ np.log(1-yp))
        #add to loss list
        loss.append(temp_los)

    return np.mean(loss)

