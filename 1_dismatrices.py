###########################################################################################################################################################

# Import general relevant libraries
import os
import time
import numpy as np

# Specify Input Path
os.chdir('/home/u450639/preprocessed_matrices')
# Import Preprocessed Unlabeled Datasets
justdie256 = np.load('justdie256.npy')
justdie512 = np.load('justdie512.npy')

justdelayered256 = np.load('justdelayered256.npy')
justdelayered512 = np.load('justdelayered512.npy')

concat256 = np.load('concat256.npy')
concat512 = np.load('concat512.npy')

simpavg256 = np.load('simpavg256.npy')
simpavg512 = np.load('simpavg512.npy')

# Combine datasets to list for looping
datasets = [justdie256, justdie512, justdelayered256, justdelayered512, concat256, concat512, simpavg256, simpavg512]
# Sanity Check on Dataset Shapes
for dataset in datasets:
    print(f'dataset shape: {dataset.shape}')

###########################################################################################################################################################

# Import relevant libraries
from skimage.metrics import structural_similarity as ssim

###########################################################################################################################################################

# Categorize datasets by tilesize for custom functions
iterate256matrix = [justdie256, justdelayered256, concat256, simpavg256]
iterate256name = ['justdie256', 'justdelayered256', 'concat256', 'simpavg256']
iterate512matrix = [justdie512, justdelayered512, concat512, simpavg512]
iterate512name = ['justdie512', 'justdelayered512', 'concat512', 'simpavg512']

###########################################################################################################################################################

# Create Dissimilarity Matrix Function for looping over 256x256 datasets
def dismatrices256(iterate_array, iterate_name):
    start_time = time.time() # Record start time
    simmatrix = np.zeros((609, 609)) # Initialize empty 2D Array for saving scores
    for i in range(609):
        for j in range(609):
            simmatrix[i, j] = ssim(iterate_array[i], iterate_array[j],
                                   win_size=11, # Low win_size = more details, suggested win_size11 (img 256x256).
                                   data_range=1, # indicating max value, normalized image max value = 1.
                                   channel_axis=2 # indicating which axis corresponds to the channels of the image.
                                   ) # similarity values = [1, -1]: 1 = perfect similarity, 0 indicates no similarity, -1 less similar than uncorrelated random noise.
    # Convert similarities to distances (measure of dissimilarity) by subtracting from 1, dissimilarity values = [0,2]: 0 = no dissimilarity, 2 = more dissimilary than if than uncorrelated random noise.
    dismatrix = 1 - simmatrix 
    
    end_time = time.time() # Record end time
    dismatrix_ET = end_time - start_time # Compute the execution time
    print(f"{iterate_name}_dismatrix execution time: {dismatrix_ET/60:.2f} minutes")
    # Sanity Check
    print(f"{iterate_name}_dismatrix shape: {dismatrix.shape}")
    print(f"{iterate_name}_dismatrix sample: {np.random.choice(dismatrix.flatten(), size=5)}")
    print(f"{iterate_name}_dismatrix identical images: {np.count_nonzero(dismatrix == 0)} (must be equal to 609)") 
    print(f"{iterate_name}_dismatrix Mean: {round(np.mean(dismatrix),5)}, Standard deviation: {round(np.std(dismatrix),5)}, Minimum: {round(np.min(dismatrix),5)}, Maximum: {round(np.max(dismatrix),5)}")

    os.chdir('/home/u450639/dissimilarity_matrices') # Specify Output Path
    np.save(f'{iterate_name}_dismatrix.npy', dismatrix) # Save Dissimilarity Matrix of Dataset

# Loop the function over the 256 tile datasets
for iname, iarray in enumerate(iterate256matrix): 
    dismatrices256(iarray, iterate256name[iname])

###########################################################################################################################################################

# Create Dissimilarity Matrix Function for looping over 512x512 datasets
def dismatrices512(iterate_array, iterate_name):
    start_time = time.time() # Record start time
    simmatrix = np.zeros((165, 165)) # Initialize empty 2D Array for saving scores
    for i in range(165):
        for j in range(165):
            simmatrix[i, j] = ssim(iterate_array[i], iterate_array[j],
                                   win_size=15, # Low win_size = more details, suggested win_size15 (img 512x512).
                                   data_range=1, # indicating max value, normalized image max value = 1.
                                   channel_axis=2 # indicating which axis corresponds to the channels of the image.
                                   ) # similarity values = [1, -1]: 1 = perfect similarity, 0 indicates no similarity, -1 less similar than uncorrelated random noise.
    # Convert similarities to distances (measure of dissimilarity) by subtracting from 1, dissimilarity values = [0,2]: 0 = no dissimilarity, 2 = more dissimilary than if than uncorrelated random noise.
    dismatrix = 1 - simmatrix 
    
    end_time = time.time() # Record end time
    dismatrix_ET = end_time - start_time # Compute the execution time
    print(f"{iterate_name}_dismatrix execution time: {dismatrix_ET/60:.2f} minutes")
    # Sanity Check
    print(f"{iterate_name}_dismatrix shape: {dismatrix.shape}")
    print(f"{iterate_name}_dismatrix sample: {np.random.choice(dismatrix.flatten(), size=5)}")
    print(f"{iterate_name}_dismatrix identical images: {np.count_nonzero(dismatrix == 0)} (must be equal to 165)") 
    print(f"{iterate_name}_dismatrix Mean: {round(np.mean(dismatrix),5)}, Standard deviation: {round(np.std(dismatrix),5)}, Minimum: {round(np.min(dismatrix),5)}, Maximum: {round(np.max(dismatrix),5)}")

    os.chdir('/home/u450639/dissimilarity_matrices') # Specify Output Path
    np.save(f'{iterate_name}_dismatrix.npy', dismatrix) # Save Dissimilarity Matrix of Dataset

# Loop the function over the 512 tile datasets
for iname, iarray in enumerate(iterate512matrix): 
    dismatrices512(iarray, iterate512name[iname])

###########################################################################################################################################################