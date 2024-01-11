###########################################################################################################################################
# Import General Relevant Libraries
###########################################################################################################################################

import os
import time
import numpy as np
import hashlib

###########################################################################################################################################
print('Import Preprocessed Labeled Images:')
###########################################################################################################################################
# Specify Input Path
os.chdir('/home/u450639/preprocessed_matrices/')
# Import the preprocessed labeled datasets (matrices) as a variable. 
labels256 = np.load('labels256.npy')
labels512 = np.load('labels512.npy')
print("CHECK: 512 and 256 labels imported")


###########################################################################################################################################
print('Import Preprocessed Unlabeled Images:')
###########################################################################################################################################

# Specify Input Path
os.chdir('/home/u450639/preprocessed_matrices')
# Import the preprocessed unlabeled datasets (matrices) as a variable. 
justdie256 = np.load('justdie256.npy')
justdie512 = np.load('justdie512.npy')
justdelayered256 = np.load('justdelayered256.npy')
justdelayered512 = np.load('justdelayered512.npy')
concat256 = np.load('concat256.npy')
concat512 = np.load('concat512.npy')
simpavg256 = np.load('simpavg256.npy')
simpavg512 = np.load('simpavg512.npy')
print("CHECK: preprocessings are imported")

# Combine datasets to list for looping
preprocessed = [justdie256, justdie512, justdelayered256, justdelayered512, concat256, concat512, simpavg256, simpavg512]
preprocessed_names = ['jdi256','jdi512','jde256','jde512','cc256','cc512','sa256','sa512']

# Sanity Check on Dataset Shapes
for i in preprocessed:
    print(f'dataset shapes: {i.shape}')

###########################################################################################################################################
print('Import Dissimilarity Matrices:')
###########################################################################################################################################
# For Dissimilarity Matrix Code SEE: dismatrices.py
# The preprocessed unlabeled datasets dissimilarity matrices were executed on the aurometalsaurus server of TiU

# Specify Input Path
os.chdir('/home/u450639/dissimilarity_matrices')
# Load the dissimilarity matrices in variables.
justdie256_dismatrix = np.load('justdie256_dismatrix.npy')
justdie512_dismatrix = np.load('justdie512_dismatrix.npy')
justdelayered256_dismatrix = np.load('justdelayered256_dismatrix.npy')
justdelayered512_dismatrix = np.load('justdelayered512_dismatrix.npy')
concat256_dismatrix = np.load('concat256_dismatrix.npy')
concat512_dismatrix = np.load('concat512_dismatrix.npy')
simpavg256_dismatrix = np.load('simpavg256_dismatrix.npy')
simpavg512_dismatrix = np.load('simpavg512_dismatrix.npy')

# Combine datasets to list for looping
dismatrices = [justdie256_dismatrix,justdie512_dismatrix,justdelayered256_dismatrix,justdelayered512_dismatrix,concat256_dismatrix,concat512_dismatrix,simpavg256_dismatrix,simpavg512_dismatrix]

###########################################################################################################################################
# Info On Splitting Datasets
###########################################################################################################################################

# Split Dataset - Dissimilarity (Full Augmentation)
#2 Approaches
#- Greedy Algorithm (This file performs Greedy Algorithm Only)
#- Kmeans Clustering

#Augmentation on both training-set and test-set:
#- 512 tiles: 0.10 * 165 = 16.5 = 17 indices testset
#- 256 tiles: 0.10 * 609 = 60.9 = 61 indices testset

# (17 * 8) / (165 * 8) = 0.10303030303 % 
# (61 * 8) / (609 * 8) = 0.10016420361 %

###########################################################################################################################################
# Define Functions
###########################################################################################################################################

def greedytest(dismatrix, testset_size): 
    # Using a Greedy algorithm, select the mist dissimilar images to each other. 
    max_dis = np.unravel_index(np.argmax(dismatrix), dismatrix.shape) # Which 2 images have the highest dissimilarity score (e.g. row 24 / column 71)
    disimages = set(max_dis) # By using a set we ensure that the same indices (e.g. transpose positions) don't get added. 2 images (with highest score) are added. 

    # Find an image (in rows) that is is the most dissimilar from the ones in the created set, by calculating for each image the average dissimilarity score between the already selected images.  
    while len(disimages) < testset_size: # Stop when the set contains 17 unique images. 
        avg_disscores = dismatrix[:, list(disimages)].mean(axis=1) # Select columns of indices in current set, which are 2 indices in the first loop, and 1 extra each iteration.
        for i in disimages:
            avg_disscores[i] = 0 # Set the avg score of images already in the set to 0 so they dont get selected for the set and get stuck in an infinity loop.  
        next_image = np.argmax(avg_disscores) # Find the image that has the highest (average) dissimilarity compared to the images already in the set.
        disimages.add(next_image) # Add this image to the set
    return np.array(sorted(disimages))

###########################################################################################################################################

"""
Step by Step Augmentation Process:
1. All tiles in dataset get ccw rotated 0 times (no augmentation) and added to the augmentated array list (AAL).
2. All tiles in dataset get ccw rotated 1 time and added to the AAL.
3. All tiles in dataset get ccw rotated 2 times and added to the AAL.
4. All tiles in dataset get ccw rotated 3 times and added to the AAL.
5. All tiles in the first array of the AAL are flipped vertically and are added to the same AAL.
6. All tiles in the first array of the AAL are flipped horizontally and are added to the same AAL.
7. All tiles in the second array of the AAL are flipped vertically and are added to the same AAL.
8. All tiles in the second array of the AAL are flipped horizontally and are added to the same AAL.
9. The list gets concatenated which combines the original, rotated and flipped arrays into a single array.
10. An augmented dataset is returned.
"""

def augment_tiles(dataset):

    import numpy as np
    # Define Rotation Function
    def rotate_tiles(tiles, rotation_count):
        rotated_tiles = np.zeros_like(tiles)
        for i in range(tiles.shape[0]):
            tile = tiles[i]
            rottile = np.rot90(tile, rotation_count, (0,1))
            rotated_tiles[i] = rottile
        return rotated_tiles
    # Define Flipping Functions
    def verflip_tiles(tiles):
        flipped_tiles = np.zeros_like(tiles)
        for i in range(tiles.shape[0]):
            tile = tiles[i]
            fliptile = np.flipud(tile)
            flipped_tiles[i] = fliptile
        return flipped_tiles
    def horflip_tiles(tiles):
        flipped_tiles = np.zeros_like(tiles)
        for i in range(tiles.shape[0]):
            tile = tiles[i]
            fliptile = np.fliplr(tile)
            flipped_tiles[i] = fliptile
        return flipped_tiles 
    
    augdataset = []

    for i in range(4): # 0, 1, 2, 3 rotations needed
        rot = rotate_tiles(dataset, i)
        augdataset.append(rot)

    for i in range(2): # only 0 rotation and 1 rotation are flipped
        verflip = verflip_tiles(augdataset[i])
        horflip = horflip_tiles(augdataset[i])
        augdataset.append(verflip)
        augdataset.append(horflip)

    augdataset = np.concatenate(augdataset,axis=0)
    return augdataset

###########################################################################################################################################

def unique_augimages(augmented_images):
    # Define function for computing hashes for each image 
    def compute_image_hash(img):
        import hashlib
        return hashlib.sha256(img.tobytes()).hexdigest() #convert image values to bytes & use hash algorithm & convert to hexadecimal string 

    # Compute which hashes are unique and keep track of indices
    unique_hashes = {} # Dictionary to keep track of unique hashes and their count (indices)
    for i in range(augmented_images.shape[0]):
        img = augmented_images[i]
        img_hash = compute_image_hash(img) # compute hashkey
        if img_hash not in unique_hashes:
            unique_hashes[img_hash] = [i] # add indice as list value to the hash-key
        else:
            unique_hashes[img_hash].append(i) # add indice to list value

    # Find the indices of the images that have the same hash value
    duplicate_indices = [] # list that will contain lists that have elements (indices) with same hashes. 
    for img_hash, indices in unique_hashes.items():
        if len(indices) > 1: # when list value contains more than 1 indice to a specific hash
            duplicate_indices.append(indices) # add this list value to the duplicate list

    # Print the indices of the duplicate images
    print(f"Indices with same hashvalues:{duplicate_indices}")

###########################################################################################################################################

def gdissplit_aug(matrix, matrixlabels, dismatrix):
    if len(matrix) == 165:
        testset_size = 17
        totalindices = np.arange(165)
    if len(matrix) == 609:
        testset_size = 61
        totalindices = np.arange(609)
    test = matrix[greedytest(dismatrix, testset_size)]
    testlabels = matrixlabels[greedytest(dismatrix, testset_size)]
    train = matrix[np.setdiff1d(totalindices, greedytest(dismatrix, testset_size))]
    trainlabels = matrixlabels[np.setdiff1d(totalindices, greedytest(dismatrix, testset_size))]

    mintest = np.min(test)
    maxtest = np.max(test)
    mintrain = np.min(train)
    maxtrain = np.max(train)

    print(f"Shapes non-augmentated data respectively: {train.shape}, {trainlabels.shape}, {test.shape}, {testlabels.shape}")
    train = augment_tiles(train)
    trainlabels = augment_tiles(trainlabels)
    test = augment_tiles(test)
    testlabels = augment_tiles(testlabels)
    print(f"Shapes augmentated data respectively: {train.shape}, {trainlabels.shape}, {test.shape}, {testlabels.shape}")
    # normalize with original dataset values
    train = (train - mintrain) / (maxtrain - mintrain) # normalize with original train-set values
    test = (test - mintest) / (maxtest - mintest) # normalize with original test-set values
    print(f"Train-set & Test-set: CHECK Normalized")
    
    return train, trainlabels, test, testlabels

###########################################################################################################################################
# Transform and Save Greedy Prepared Datasets
###########################################################################################################################################

# Specify Output Path
os.chdir('/home/u450639/greedy_matrices/')

for x, y, z in zip(preprocessed, dismatrices, preprocessed_names):
    print(f'NAME OF DATASET: {z}')
    if len(x) == 165:
        labels = labels512
    if len(x) == 609:
        labels = labels256
    train, trainlabels, test, testlabels = gdissplit_aug(x, labels, y)
    unique_augimages(train)
    unique_augimages(trainlabels)
    unique_augimages(test)
    unique_augimages(testlabels)
    np.save(f'{z}_ga_train.npy', train)
    np.save(f'{z}_ga_trainlabels.npy', trainlabels)
    np.save(f'{z}_ga_test.npy', test)
    np.save(f'{z}_ga_testlabels.npy', testlabels)


    
