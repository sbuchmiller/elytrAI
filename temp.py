from PIL import Image, ImageOps
import numpy as np
test = ImageOps.grayscale(Image.open(r"C:\Users\scott\Desktop\CS175\dump\crop0.png"))
columnWidth = test.width
image_height = test.height
test = np.array(test)
accu = 0
print()
print("maxpix", max(test[0]))
for w in range(0,columnWidth):
    for h in range(0,image_height):
        accu += test[h][w]
print(accu / (columnWidth * image_height))