def plot_model_evaluation(hist):
    
    '''
    Args:

    hist: history object created during model.fit.

    Plots loss and accuracy curves in separated charts. Assumes that validation sample was used.
    '''
    x = range(len(hist.history['accuracy']))
    matplotlib.pyplot.figure(figsize = (12,5))
    matplotlib.pyplot.subplot(1,2,1)
    matplotlib.pyplot.title('Accuracy')
    matplotlib.pyplot.plot(x,hist.history['accuracy'],label = 'accuracy')
    matplotlib.pyplot.plot(x,hist.history['val_accuracy'],label = 'val_accuracy')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.subplot(1,2,2)
    matplotlib.pyplot.title('Loss')
    matplotlib.pyplot.plot(x,hist.history['loss'],label = 'loss')
    matplotlib.pyplot.plot(x,hist.history['val_loss'],label = 'val_loss')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
