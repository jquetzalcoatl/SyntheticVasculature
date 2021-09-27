from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",
                    type=str)
args = parser.parse_args()
#print(args.square**2)

filename = args.square
# with Image.open(filename) as image:
#     width, height = image.size

image=Image.open(filename)
 # image.show()

data = np.asarray( image, dtype="int32" )


# Invert the horse image
#image = invert(data.horse())

# perform skeletonization
skeleton = skeletonize(data)

im = Image.fromarray(skeleton)
im.save(filename[-14:-4]+"_Sket.tif")

# display results
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
#                          sharex=True, sharey=True)
#
# ax = axes.ravel()
#
# ax[0].imshow(image, cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('original', fontsize=20)
#
# ax[1].imshow(skeleton, cmap=plt.cm.gray)
# ax[1].axis('off')
# ax[1].set_title('skeleton', fontsize=20)
#
# fig.tight_layout()
# plt.show()
