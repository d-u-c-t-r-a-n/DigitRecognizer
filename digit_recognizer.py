import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def showMe(image, label):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(label)
    plt.show()


datasets = tf.keras.datasets
layers = tf.keras.layers
models = tf.keras.models
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Load
(train_images, train_labels), (test_images,
                               test_labels) = datasets.mnist.load_data()


# Normalize
train_images, test_images = train_images / 255.0, test_images / 255.0


# Build convolutional base
model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(56, (3, 3), activation='relu'))


# Build Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(56, activation='relu'))
model.add(layers.Dense(10))

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Fit
model.fit(train_images, train_labels, epochs=2,
          validation_data=(test_images, test_labels))


# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Test
while True:
    try:
        number = int(input("Pick a number between 0 to 9999: "))
    except:
        break

    if number < 0 or number > 9999:
        break

    test_image = test_images[number]
    test_label = test_labels[number]
    predicted_index = model.predict(np.array([test_image]))
    predicted_class = class_names[np.argmax(predicted_index)]
    print("Expected " + str(test_label) +
          " - Got " + str(predicted_class))
    showMe(test_image, test_label)
