import numpy as np
from PIL import Image

img = Image.open(r'G:\database\0007_base.png')
img = np.array(img)
print(np.unique(img))