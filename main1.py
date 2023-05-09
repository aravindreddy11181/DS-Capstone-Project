from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dropout, BatchNormalization, Dense, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import MaxPooling2D, Dense, Dropout, Conv2D, BatchNormalization,Flatten
from keras.callbacks import EarlyStopping, LearningRateScheduler
import os

train_data = 'data/train/'
validation_data = 'data/test/'

train_datagenerator = ImageDataGenerator(
    rotation_range=30,
    rescale=1./255,
    shear_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,
    fill_mode='nearest')

validation_datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = train_datagenerator.flow_from_directory(
    train_data,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    samples_per_class=100)

validation_generator = validation_datagenerator.flow_from_directory(
    validation_data,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    samples_per_class=50)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), activation='elu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(5, 5), activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(5, 5), activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(5, 5), activation='elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# Define learning rate schedule
def adjust_learning_rate(current_epoch):
    learning_rate = 0.001
    if current_epoch > 30:
        learning_rate *= 0.0005
    elif current_epoch > 20:
        learning_rate *= 0.001
    elif current_epoch > 10:
        learning_rate *= 0.01
    # Print the updated learning rate for debugging purposes
    print('Learning rate adjusted to: ', learning_rate)
    return learning_rate

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Define number of training and validation images
num_train_imgs = sum([len(files) for root, dirs, files in os.walk(train_data)])
num_test_imgs = sum([len(files) for root, dirs, files in os.walk(validation_data)])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs // 64,
                    epochs=3,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs // 64,
                    callbacks=[LearningRateScheduler(adjust_learning_rate), early_stopping])


import matplotlib.pyplot as plt

# Get training and validation loss and accuracy from history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot loss over time
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy over time
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


model.save('model_file_demo.h5')
