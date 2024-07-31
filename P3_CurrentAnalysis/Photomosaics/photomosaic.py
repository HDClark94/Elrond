import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

def load_images_from_folder(folder, exclude_list=[]):
    x, y, w, h = 200, 300, 900, 900

    images = []
    for filename in os.listdir(folder):
        include = True
        for exclude_str in exclude_list:
            if exclude_str in filename:
                include = False
        if include:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                # Crop the image
                img = img[y:y + h, x:x + w]
                images.append(img)
    return images

from PIL import Image
# Example usage
target_image = cv2.imread('/mnt/datastore/Harry/plot_viewer/worldmap.png')
target_image = cv2.imread('/mnt/datastore/Harry/plot_viewer/photomosaic/newMap.jpg')

tile_images = load_images_from_folder('/mnt/datastore/Harry/Cohort11_april2024/' 
                                      'derivatives/M21/D16/of/M21_D16_2024-05-16_14-03-05_OF1/'
                                      'processed/kilosort4/Figures/rate_maps', exclude_list =["all_firing_rates",
                                                                                        "370","371","316","306",
                                                                                        "310","312","301","276","260",
                                                                                        "261","264","265","134","128"])

# tile images will be 900x900
#make target image 10X18 tiles
# make the target
# Load the image

tile_size = 900
# Define the new size
nrows = 12
ncols = 20
new_size = (tile_size*ncols, tile_size*nrows)
og_size = np.shape(target_image)[1], np.shape(target_image)[0]

# Resize the image
target_image = cv2.resize(target_image, new_size)

new_image = target_image.copy()
# make new image
for i in range(nrows):
    for j in range(ncols):
        random_tile = tile_images[np.random.randint(low=0, high=len(tile_images), size=1)[0]]
        new_image[int(i*tile_size):(int(i+1)*tile_size),
                  int(j*tile_size):(int(j+1)*tile_size)] = random_tile
cv2.imwrite('/mnt/datastore/Harry/plot_viewer/photomosaic/photomosaic.jpg', new_image)

tmp_image = new_image.copy()
for n in range(100000):
    i = np.random.randint(low=0, high=nrows, size=1)[0]
    j = np.random.randint(low=0, high=ncols, size=1)[0]
    random_tile = tile_images[np.random.randint(low=0, high=len(tile_images), size=1)[0]]
    current_tile = tmp_image[int(i*tile_size):(int(i+1)*tile_size), int(j*tile_size):(int(j+1)*tile_size)]
    target_tile = target_image[int(i*tile_size):(int(i+1)*tile_size), int(j*tile_size):(int(j+1)*tile_size)]

    if (stats.pearsonr(random_tile.flatten(), target_tile.flatten())[0] >\
            stats.pearsonr(current_tile.flatten(), target_tile.flatten())[0]):
        tmp_image[int(i * tile_size):(int(i + 1) * tile_size), int(j * tile_size):(int(j + 1) * tile_size)] = random_tile

    if n % 500 == 0:
        resized_new_image = cv2.resize(tmp_image, og_size)
        cv2.imwrite('/mnt/datastore/Harry/plot_viewer/photomosaic/it_'+str(n+1)+'.jpg', resized_new_image)
print("")

