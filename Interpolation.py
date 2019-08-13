import cv2
import matplotlib.pyplot as plt
path = "../apple.jpg"
img = cv2.imread(path)

# Raw Image
plt.imshow(img)

#NN
near_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
plt.imshow(near_img)
# Inter Linear
bilinear_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
plt.imshow(bilinear_img)
# Inter Cubic
bicubic_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
plt.imshow(bicubic_img)
