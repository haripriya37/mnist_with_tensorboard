#import necessary libraries
import tensorflow as tf
import datetime, os
import matplotlib.pyplot as plt
#below imports are for building cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#load dataset from tensorflow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#to view any image in the dataset
image_index = 10
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')

# Converting values into float for division
train_norm = x_train.astype('float32')
test_norm = x_test.astype('float32')

# Normalizing the RGB values by dividing with maximum RGB value.
x_train = train_norm / 255.0
x_test = test_norm / 255.0

# Load the TensorBoard notebook extension Note: The code for tensorboard works only in colabs
%load_ext tensorboard

# Creating a Sequential Model and adding the necessary layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

#compiling the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#initiating tensorboard callbacks
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

#use command below to launch tensorboard
%tensorboard --logdir logs

# Reshaping the train and test attributes to 4-dims, to make them compatible for the Keras API
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

#fitting the model, see tensorboard for visualising the training process as training goes on 
model.fit(x=x_train, 
            y=y_train, 
            epochs=10, 
            validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback])
            
#make predictions for new images
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())            
