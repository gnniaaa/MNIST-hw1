import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1])) #32, 24x24
model.add(Conv2D(64, (5,5), activation='relu')) #64, 20x20
model.add(MaxPooling2D(pool_size=(2,2))) #64, 10x10
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,epochs=NUM_EPOCHS , batch_size=BATCH_SIZE)
train_accuracy = model.evaluate(x_train, y_train, verbose=0)
test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print('Training loss: %.4f, Training accuracy: %.2f%%' % (train_accuracy[0],train_accuracy[1]*100))
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (test_accuracy[0],test_accuracy[1]*100))