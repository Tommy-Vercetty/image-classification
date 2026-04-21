

from __future__ import print_function
from sklearn.metrics import classification_report, recall_score, accuracy_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
import time
import keras
import keras_tuner as kt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

#--------------------------
#- CHANGE 5: KERAS TUNING -
#--------------------------
def modelBuilder(hp):
    baseModel = MobileNetV2(
        input_shape = (img_height, img_width, img_channels),
        include_top = False,
        weights = 'imagenet'
    )
    baseModel.trainable = True

    for layer in baseModel.layers[:-30]:
        layer.trainable = False
    #-------------------------------
    #- CHANGE 1: DATA AUGMENTATION -
    #-------------------------------
    #First change we make is data augmentation
    dataAugmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2)
        ])
    
    model = tf.keras.models.Sequential([
        dataAugmentation,
        tf.keras.layers.Lambda(preprocess_input),
        baseModel,
        GlobalAveragePooling2D(),
        Dense(units = hp.Int('dense_units', min_value = 64, max_value = 512, step = 64), activation = 'relu'),
        Dropout(rate = hp.Float('dropout_rate', 0.2, 0.6, step = 0.1)),
        Dense(num_classes, activation = 'softmax')
    ])
    #---------------------------
    #- CHANGE 6: LEARNING RATE -
    #---------------------------
    model.compile(
        optimizer = tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-5, 3e-4, sampling = 'log')),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

batch_size = 12
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again
train_dir = 'D:/chest_xray/train'
test_dir = 'D:/chest_xray/test'
totalTimeStartingPoint = time.time()
with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)

    #- CHANGE 5: KERAS TUNING -
    kerasTuner = kt.RandomSearch(
        modelBuilder,
        objective = 'val_accuracy',
        max_trials = 10,
        directory = 'kt_results',
        project_name = 'chestX-RayTuning'
    )

    print('Class Names: ',class_names)
    num_classes = len(class_names)
    #------------------------------------
    #- CHANGE 2: CLASS WEIGHT BALANCING -
    #------------------------------------
    labels = np.concatenate([y.numpy() for x, y in train_ds])

    classWeights = compute_class_weight(
        class_weight = "balanced",
        classes = np.unique(labels),
        y = labels
    )
    classWeightDictionary = dict(enumerate(classWeights))
    #---------------------
    #- DATA DISTRIBUTION -
    #---------------------
    classCounts = {name : 0 for name in class_names}
    for images, labels in train_ds:
        for label in labels.numpy():
            classCounts[class_names[label]] += 1

    print("<< Class Distribution [Training Portion] >>")
    for k, v in classCounts.items():
        print(f"{k}: {v}")

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    #---------------------------------------------
    #- CHANGE 4: TRANSFER LEARNING - MobileNetV2 -
    #---------------------------------------------
    #baseModel = MobileNetV2(
    #    input_shape = (img_height, img_width, img_channels),
    #    include_top = False,
    #    weights = 'imagenet'
    #)
    #baseModel.trainable = False
    #create model
    #model = tf.keras.models.Sequential([
    #   dataAugmentation,
    #   Rescaling(1.0/255),
    #   baseModel,
    #   Conv2D(16, (3,3), activation = 'relu', input_shape = (img_height,img_width, img_channels)),
    #   MaxPooling2D(2,2),
    #   Conv2D(32, (3,3), activation = 'relu'),
    #   MaxPooling2D(2,2),
    #   Conv2D(32, (3,3), activation = 'relu'),
    #   MaxPooling2D(2,2),
    #----------------------------------------
    #- CHANGE 3: TRRANSLATION LAYER - GAP2D -
    #----------------------------------------
    #   Flatten(), # flatten multidimensional outputs into single dimension for input to dense fully connected layers
    #   Dense(512, activation = 'relu'),
    #   GlobalAveragePooling2D(),
    #   Dense(128, activation = 'relu'),
    #   Dropout(0.2),
    #   Dense(num_classes, activation = 'softmax')
    #])

    #model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=Adam(),
    #              metrics=['accuracy'])
    
    #earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras",save_freq='epoch',save_best_only=True)

    if fit:
        kerasTuner.search(
            train_ds,
            validation_data = val_ds, 
            epochs = 8,
            class_weight = classWeightDictionary
        )
        bestModel = kerasTuner.get_best_models(num_models = 1)[0]
        model = bestModel
        
        history = bestModel.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=classWeightDictionary
        )
        
            #model.fit(
            #train_ds,
            #batch_size=batch_size,
            #validation_data=val_ds,
            #callbacks=[save_callback],
            #epochs=epochs,
            #class_weight = classWeightDictionary
            #)
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    #if shuffle=True when creating the dataset, samples will be chosen randomly   
    score = model.evaluate(test_ds)
    print('Test accuracy:', score[1])

    #----------------------
    #- PEFORMANCE METRICS -
    #----------------------
    outputClassLabelActual = []
    outputClassLabelPredictions = []
    for images, labels in test_ds:
        predictions = model.predict(images)
        outputClassLabelActual.extend(labels.numpy())
        outputClassLabelPredictions.extend(np.argmax(predictions, axis = 1))

    overallAccuracy = accuracy_score(outputClassLabelActual, outputClassLabelPredictions)
    precision = precision_score(outputClassLabelActual, outputClassLabelPredictions, average = None)
    recall = recall_score(outputClassLabelActual, outputClassLabelPredictions, average = None)
    f1Score = f1_score(outputClassLabelActual, outputClassLabelPredictions, average = None)
    print("\n << Classification Report >>\n", classification_report(outputClassLabelActual, outputClassLabelPredictions, target_names = class_names))
    print(" Overall Accuracy:", overallAccuracy)
    for i, class_name in enumerate(class_names):
        print(f"\n Class: [{class_name}]")
        print("  Precision:", precision[i])
        print("  Recall:", recall[i])
        print("  F1 Score:", f1Score[i])

    totalTimeEndingPoint = time.time()
    print("\n << Total Execution Time >>") 
    print(f" {totalTimeEndingPoint - totalTimeStartingPoint:.2f} seconds\n")

    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(),0))#perform a prediction on this image
            plt.title('Actual:' + class_names[labels[i].numpy()]+ '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()