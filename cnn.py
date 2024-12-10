#implementing a CNN model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Part 1 - Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Part 2 - Building the CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))) #convolutional layer
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)) #pooling
cnn.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')) #adding another convolutional layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten()) #flattening
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) #full connection
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #output layer

# Part 3 - Training the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25) #training the model


# Part 4 - Making a single prediction
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size = (64, 64)) #loading the image
test_image = image.img_to_array(test_image) #converting the image to an array
test_image = np.expand_dims(test_image, axis = 0) #adding an extra dimension
result = cnn.predict(test_image) #predicting the result
training_set.class_indices #checking the indices of the classes
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction) #printing the prediction