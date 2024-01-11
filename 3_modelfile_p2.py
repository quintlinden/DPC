###########################################################################################################################
### Import genral relevant libraries.
print("Import General Relevant Libraries:")
import os
import time
import numpy as np
print("CHECK: libraries imported")
print()
###########################################################################################################################

###########################################################################################################################
### Sanity Check: using a GPU.
print("Detect GPUs:")
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs available: {len(gpus)}")
else:
    print("No GPU available.")
print()
###########################################################################################################################

###########################################################################################################################
### Import the relevant datasets for this model configuration.
os.chdir('/home/u450639/greedy_matrices/') # Specify Input Path.
trainset = np.load('cc256_ga_train.npy')
trainsetlabel = np.load('cc256_ga_trainlabels.npy')
testset = np.load('cc256_ga_test.npy')
testsetlabel = np.load('cc256_ga_testlabels.npy')
modelname = 'stack_p2_cc256g' # Set the name of the unique model to connect results to.

# Sanity Check: Confirm the dimensions of the datasets.
print(f'Modelname: {modelname}')
print(f'shape of trainset file: {trainset.shape}')
print(f'shape of trainsetlabel file: {trainsetlabel.shape}')
print(f'shape of testset file: {testset.shape}')
print(f'shape of testsetlabel file: {testsetlabel.shape}')    
print()
###########################################################################################################################

###########################################################################################################################
### Create U-Net++ architecture.
print("Create network architecture:")
# Import relevant libraries for network architecture
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def unet_plus_plus(input_shape=(trainset.shape[1], trainset.shape[2], trainset.shape[3]), n_labels=5):
    # input_shape defines the height, width, and channels of each image in an array of images.
    # n_classes (default = 5) specifies number of units in the output layer representing the 5 labels to be classified. 
    
    # Input layer.
    inputs = Input(input_shape)

    # Contracting path.
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)

    # Expanding path.
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)

    up10 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge10 = concatenate([conv2, up10, conv8], axis=3)
    conv10 = Conv2D(128, 3, activation='relu', padding='same')(merge10)

    up11 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv10)
    merge11 = concatenate([conv1, up11, conv9], axis=3)
    conv11 = Conv2D(64, 3, activation='relu', padding='same')(merge11)

    # Output layer.
    output = Conv2D(n_labels, 1, activation='sigmoid')(conv11)

    model = Model(inputs=inputs, outputs=output)
    return model

print("CHECK: network architecture created")
print()
# Create overview of model's architecture.
model = unet_plus_plus
print("Network architecture details:")
model.summary()
print()
###########################################################################################################################

###########################################################################################################################
### Compiling and fitting the model configuration using k-fold cross-validation.
print("Compile & Fit Model using KFold (k=5):")
# Import relevant libraries.
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

start_time = time.time() # Record the start time.
k = 5 # Define number of folds.
kfold = KFold(n_splits=k, shuffle=True, random_state=5)

foldhistory = []

fold = 1
for trainfold, valfold in kfold.split(trainset):
    print(f"Fold {fold}/{k}")
    train, val = trainset[trainfold], trainset[valfold]
    trainlabel, vallabel= trainsetlabel[trainfold], trainsetlabel[valfold]

    model = unet_plus_plus()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics=['accuracy'])

    # Train the model separately per fold.
    history = model.fit(train, trainlabel, batch_size=2, epochs=8, validation_data=(val, vallabel))
    foldhistory.append(history.history)

    # Evaluate the model separately per fold.
    accuracies = model.evaluate(val, vallabel, verbose=0)
    print(f"Validation accuracy: {accuracies[1] * 100:.2f}%")

    fold += 1

# Save training history.
import pickle
print()
os.chdir('/home/u450639/traininghistories/') # Specify Output Path.
with open(f'{modelname}_trainhistory.pkl', 'wb') as file_pi:
    pickle.dump(foldhistory, file_pi)
print("CHECK: History Saved")

# Compute total training time.
print()
end_time = time.time() # Record the end time.
execution_time = end_time - start_time # Compute the execution time.
print("Total training time: {:.2f} hours".format(execution_time/60/60))
print()

# Save the final model
print("Save The Trained Model:")
os.chdir('/home/u450639/prediction_matrices/') # Specify Output Path.
model.save(f'{modelname}.h5') # Save the entire model to a single HDF5 file.
print("CHECK: Model Saved")
print()
###########################################################################################################################

###########################################################################################################################
### Make prediction with the final model on the test set.
print("Make Predictions with Trained Model:")
start_time = time.time() # Record the start time.

# Predict labels for test set.
labelpred = model.predict(testset)
end_time = time.time() # Record the end time.
execution_time = end_time - start_time # Compute the execution time.
print("Prediction time: {:.2f} minutes".format(execution_time/60))

# Sanity Check: find characteristics of model predictions.
print(f'data type of predictions: {labelpred.dtype}')
print(f'shape of predictions: {labelpred.shape}')

# Save Predictions.
os.chdir('/home/u450639/prediction_matrices/') # Specify Output Path.
np.save(f'pred_{modelname}.npy', labelpred) 
print(f'CHECK: Prediction Arrays Saved')
###########################################################################################################################