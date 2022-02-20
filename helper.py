from matplotlib import pyplot as plt

def plot_model_evaluation(hist):
    
    '''
    Args:

    hist: history object created during model.fit.

    Plots loss and accuracy curves in separated charts. Assumes that validation sample was used.
    '''
    x = range(len(hist.history['accuracy']))
    plt.figure(figsize = (12,5))
    plt.subplot(1,2,1)
    plt.title('Accuracy')
    plt.plot(x,hist.history['accuracy'],label = 'accuracy')
    plt.plot(x,hist.history['val_accuracy'],label = 'val_accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.title('Loss')
    plt.plot(x,hist.history['loss'],label = 'loss')
    plt.plot(x,hist.history['val_loss'],label = 'val_loss')
    plt.legend()
    plt.show()


import tensorflow as tf
import datetime

def create_tensorboard_callback(dir_name, model_name):
    '''
    Creates callback to save model results to dir_name.

    Returns:
    Callback
    '''
    log_dir = dir_name + '/' + model_name + datetime.datetime.now().strftime("_%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    print(f'Saving tensorboard logfiles to {log_dir}')
    return tensorboard_callback
    

def compare_histories(orignal_hist, new_hist, initial_epochs = 5):
    acc = orignal_hist.history['accuracy']
    loss = orignal_hist.history['loss']
    val_acc = orignal_hist.history['val_accuracy']
    val_loss = orignal_hist.history['val_loss']

    total_acc = acc + new_hist.history['accuracy']
    total_loss = loss + new_hist.history['loss']
    total_val_acc = val_acc + new_hist.history['val_accuracy']
    total_val_loss = val_loss + new_hist.history['val_loss']

    plt.figure(figsize = (8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc, label = "accuracy")
    plt.plot(total_val_acc, label = "val_accuracy")
    plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label = "start fine tuning")
    plt.legend()

    plt.figure(figsize = (8,8))
    plt.subplot(2,1,2)
    plt.plot(total_loss, label = "loss")
    plt.plot(total_val_loss, label = "val_loss")
    plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label = "start fine tuning")
    plt.legend()


from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y,y_pred, label_description = None):
    '''
    Args:
    y: true labels.
    y_pred: predicted labels.
    label_description: label description for confusion matrix plot.
    
    Returns accuracy and confusion matrix plot.
    '''
    acc = sum(y_pred == y)/len(y)
    print(f'Accuracy = {acc}')
    
    cm = confusion_matrix(y, y_pred)
  
    if cm.shape[0] == 2:
      tn, fn, tp, fp = cm[0,0], cm[1,0], cm[1,1], cm[0,1]
      precision = tp/(tp+fp)
      recall = tn/(tn+fn)
      f1_score = 2*precision*recall/(precision+recall)
      print(f'Precision (TP / AllP) = {(precision*100):.2f}%\nRecall (TN / AllN)= {(recall*100):.2f}%\nF1 Score = {(f1_score*100):.2f}%\n\n')

#plot cm
    figsize = (5,5) if cm.shape[0] == 2 else (14,14)
    cm_perc = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    fig, ax = plt.subplots(figsize = figsize)
    cax = ax.matshow(cm, cmap = plt.cm.Blues)
    fig.colorbar(cax)
    labels = label_description if label_description is not None else np.arange(cm.shape[0])
    ax.set(title = 'Confusion Matrix',
           xlabel = 'Predicted',
           ylabel = 'True',
           xticks = np.arange(cm.shape[0]),
           yticks = np.arange(cm.shape[0]),
           xticklabels = labels,
           yticklabels = labels)
  
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.title.set_size(15)

    threshold = (cm.max() + cm.min())/2

    import itertools

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[0])):
      plt.text(j,i, f'{cm[i,j]} ({cm_perc[i,j]*100:.1f}%)',
               horizontalalignment = 'center',
               color = 'white' if cm[i,j] > threshold else 'black',
               size = 10)
