import cv2
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize

# Read GT label (BMP format)
label_path = '/home/yongjun/diffusion-seg/data/OCTA500_3M/test/label/10454.bmp'
image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f'Error: Could not read {label_path}')
else:
    # Binarize
    binary = (image > 127).astype(np.uint8)
    
    # Betti-0: Connected components
    labeled_image = measure.label(binary, connectivity=2)
    betti_0 = labeled_image.max()
    
    # Betti-1: Loops
    skeleton = skeletonize(binary)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) * skeleton
    endpoints = np.sum(neighbors == 2)
    junctions = np.sum(neighbors >= 4)
    betti_1 = max(0, junctions - endpoints // 2)
    
    print(f'GT Label for patient 10454:')
    print(f'  Betti-0 (connected components): {betti_0}')
    print(f'  Betti-1 (loops): {betti_1}')
    print(f'  Endpoints: {endpoints}')
    print(f'  Junctions: {junctions}')