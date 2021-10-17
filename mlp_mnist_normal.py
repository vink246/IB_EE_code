from datetime import datetime
import sys
from keras.datasets import mnist
from matplotlib import pyplot
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.python.keras.backend import batch_normalization

# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
'''
The training dataset consists of a 3 dimensional matrix for training with dimensions 60000,28,28 and a label vector with 60000 elements.
This means that there are 60000 images of size 28 pixels by 28 pixels which will be used to train.
The number of rows of data has to match with the training data and target labels, which in this case are both 60000.
'''
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
'''
This is the test data which is what will be used to check the performance with
'''
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28*28))
testX = testX.reshape((testX.shape[0], 28*28))

# encoding categorical data
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# function to scale pixels, this function will make the value of each pixel between 0 and 1 as opposed to 0-255
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# scaling pixels
trainX, testX = prep_pixels(trainX, testX)
print (trainX.shape[1])
# funtion to define mlp model
def define_model():
    model = Sequential()
    # print(trainX.shape[1])
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', input_dim=784)) # dense layer 1
	# model.add(batch_normalization()) # batch normalization
    model.add(Dense(10, activation='softmax')) # dense layer 2
	# compile model
    opt = SGD(lr=0.01, momentum=0.9) # SGD optimizer taker parameters 'learning rate' and 'momentum'
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# function to evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=10):
    scores, histories = list(), list()
	# prepare cross validation (10 folds mean the avg and standard deviation will be calculated across 10 trials)
    kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
    for train_ix, test_ix in kfold.split(dataX): #splitting train data into train and cross validation
        # define model
        model = define_model()
        print(model.summary())
        model.save('mlpMNIST.h5')
		# select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # note: 'testX' here is not the final test data but is data to be used in cross validation

		# fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # store scores
        scores.append(acc)
        histories.append(history) # history is being used so we can see trend in performance as model trains
    return scores, histories

# funtion to plot diagnostic learning curves (helps us analyze trend in performance)
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss (should ideally be very low)
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy (should ideally be very high)
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))

# start time of execution
startTime = datetime.now()
# evaluate model
scores, histories = evaluate_model(trainX, trainY)

print("execution time:", datetime.now()-startTime)

# learning curves
summarize_diagnostics(histories)
# summarize estimated performance
summarize_performance(scores)
