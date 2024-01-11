##################################################################################################################################################################################
##################################################################################################################################################################################

def gray_to_binary(img_gray):
    """ 
    Arguments: 
    - img_gray (numpy.ndarray): a grayscale image
    Returns: 
    - img_binary (numpy.ndarray): a binary image
    """
    # Import relevant packages
    import cv2
    import numpy as np
    # cv2.threshold returns an image and a thereshold value (which is ignored: _,)
    #the 'img_gray' turns all values greater than '0' to '255' (white).
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    # this is a 8-bit image, since 'cv2' does not support 1-bit binary images
    return img_binary

##################################################################################################################################################################################
##################################################################################################################################################################################

def bool_label_tiling(img, tile_size):
    """
    Arguments:
    - img (numpy.ndarray): The input binary image to be tiled.
    - tile_size (tuple): A tuple of the tile size in (height, width, depth) format.

    Returns:
        - tiles (numpy.ndarray): A 4D array containing the tiled images.
    Prints:
        - overlap that the images have according to the set tile_size.    
    """
    # Import relevant packages
    import numpy as np

    # Get the dimensions of the image
    img_h, img_w, img_d = img.shape

    # Calculate the number of tiles that could fit in each dimension (np.ceil: rounds up to nearest integer which leads to slight overlapping).
    num_tiles_h = int(np.ceil(img_h / tile_size[0]))
    num_tiles_w = int(np.ceil(img_w / tile_size[1]))

    # Calculate the stride(amount that the kernel/tile_size moves over the image) to ensure that all tiles are the same size.
    stride_h = int(np.ceil((img_h - tile_size[0]) / (num_tiles_h - 1)))
    stride_w = int(np.ceil((img_w - tile_size[1]) / (num_tiles_w - 1)))

    # Print the overlapping (kernel - stride) in percentage. 
    print('overlapping height:', round(((tile_size[0] - stride_h) / tile_size[0] * 100), 1), '%')
    print('overlapping width:', round(((tile_size[1] - stride_w) / tile_size[1] * 100), 1), '%')

    # Initialize an empty array to hold the tiled images (4 dimensions: each tile has 3 dimensions in a 1 larger image) (bool = each element has valid values True or False)
    tiles = np.zeros((num_tiles_h, num_tiles_w, tile_size[0], tile_size[1], img_d), dtype=bool)

    # Loop over the tiled images and extract each tile from the image
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # Calculate the start and end indices for the current tile
            start_h = i * stride_h
            start_w = j * stride_w
            end_h = min(start_h + tile_size[0], img_h)
            end_w = min(start_w + tile_size[1], img_w)

            # Extract the current tile from the image
            tiles[i, j, :end_h - start_h, :end_w - start_w, :] = img[start_h:end_h, start_w:end_w, :]

    # Convert the tiled images to a 4D array
    tiles = np.reshape(tiles, (num_tiles_h * num_tiles_w, tile_size[0], tile_size[1], img_d))

    # Return the tiled images
    return tiles

##################################################################################################################################################################################
##################################################################################################################################################################################

def image_tiling(img, tile_size):
    """
    Arguments:
    - img (numpy.ndarray): The input image to be tiled.
    - tile_size (tuple): A tuple of the tile size in (height, width) format.

    Returns:
        - tiles (numpy.ndarray): A 4D array containing the tiled images.
    Prints:
        - overlap that the images have according to the set tile_size.    
    """
    # Import relevant packages
    import numpy as np

    # Get the dimensions of the image
    img_h, img_w, img_c = img.shape

    # Calculate the number of tiles that could fit in each dimension (np.ceil: rounds up to nearest integer which leads to slight overlapping).
    num_tiles_h = int(np.ceil(img_h / tile_size[0]))
    num_tiles_w = int(np.ceil(img_w / tile_size[1]))

    # Calculate the stride(amount that the kernel/tile_size moves over the image) to ensure that all tiles are the same size.
    stride_h = int(np.ceil((img_h - tile_size[0]) / (num_tiles_h - 1)))
    stride_w = int(np.ceil((img_w - tile_size[1]) / (num_tiles_w - 1)))
    # Print the overlapping (kernel - stride) in percentage. 
    print('overlapping height:',round(((tile_size[0] - stride_h)/tile_size[0]*100),1),'%')
    print('overlapping width:',round(((tile_size[1] - stride_w)/tile_size[1]*100),1),'%')

    # Initialize an empty array to hold the tiled images (4 dimensions: each tile has 3 dimensions in a 1 larger image) (choose dtype of input image, suitable for normalized images)
    tiles = np.zeros((num_tiles_h, num_tiles_w, tile_size[0], tile_size[1], img_c), dtype=img.dtype)


    # Loop over the tiled images and extract each tile from the image
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # Calculate the start and end indices for the current tile
            start_h = i * stride_h
            start_w = j * stride_w
            end_h = min(start_h + tile_size[0], img_h)
            end_w = min(start_w + tile_size[1], img_w)

            # Extract the current tile from the image
            tiles[i, j, :end_h - start_h, :end_w - start_w, :] = img[start_h:end_h, start_w:end_w, :]

    # Convert the tiled images to a 4D array
    tiles = np.reshape(tiles, (num_tiles_h * num_tiles_w, tile_size[0], tile_size[1], img_c))

    # Return the tiled images
    return tiles

##################################################################################################################################################################################
##################################################################################################################################################################################