import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import Model
from tqdm import tqdm
import wandb
from wandb.keras import WandbCallback
import numpy as np

wandb.init(project="Famous_People")

dataset_path=r"C:\Users\Ariya Rayaneh\Desktop\famous"

width = height = 224

idg = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=(0.9, 1.1),
    validation_split=0.2
)

train_data = idg.flow_from_directory(
    dataset_path,
    target_size=(width, height),
    class_mode='categorical',
    subset='training'
)

val_data = idg.flow_from_directory(
    dataset_path,
    target_size=(width, height),
    class_mode='categorical',
    subset='validation'
)

batch_size = 32
epochs = 10

class RezaNet(Model):
    def __init__(self):
        super().__init__()

        self.Conv2D_1 = Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3))
        self.Conv2D_2 = Conv2D(64, (5, 5), activation='relu')
        self.MaxPooling = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(128, activation='relu')
        self.dense_2 = Dense(3, activation='softmax')
        self.dropout = Dropout(0.5)

    def call(self, x):
        y = self.Conv2D_1(x)
        z = self.MaxPooling(y)
        j = self.Conv2D_2(z)
        k = self.MaxPooling(j)
        m = self.flatten(k)
        n = self.dense_1(m)
        w = self.dropout(n)
        out = self.dense_2(w)
        return out

model = RezaNet()

config = wandb.config
learning_rate = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.MeanAbsoluteError()
test_loss = tf.keras.metrics.MeanAbsoluteError()
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_accuracy = tf.keras.metrics.CategoricalAccuracy()
labelss=[]
for epoch in range(epochs):
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    train_loss.reset_states()
    test_loss.reset_states()
    print("Epoch: ", epoch)

    for i, (images, labels) in enumerate(tqdm(train_data)):
        if len(train_data) <= i:
            break
        with tf.GradientTape() as gTape:
            predictions = model(images,training=True)
            loss = loss_function(labels, predictions)
            train_loss(labels, predictions)
            train_accuracy(labels, predictions)
            labelss.append(labels)

        gradiants = gTape.gradient(loss, model.trainable_variables)


        optimizer.apply_gradients(zip(gradiants, model.trainable_variables))

    for i, (images, labels) in enumerate(tqdm(val_data)):
        if len(train_data) <= i:
            break
        predictions = model(images)
        loss = loss_function(labels, predictions)
        test_accuracy(labels, predictions)
        test_loss(labels, predictions)
    print(len(labelss))
    print("Train Accuracy: ", train_accuracy.result())
    print("Test Accuracy: ", test_accuracy.result())
    print("Train loss : ", train_loss.result())
    print("Test loss : ", test_loss.result())

    wandb.log({'epochs': epoch,
               'Train_loss': np.mean(train_loss.result()),
               'Train_accuracy': float(train_accuracy.result()),
               'val_loss': np.mean(test_loss.result()),
               'val_accuracy': float(test_accuracy.result())})

model.save_weights(r"C:\Users\Ariya Rayaneh\Desktop/Famous_People.h5")

