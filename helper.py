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
