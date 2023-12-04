import cv2

img_path = "output/flow_img_0.png"
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY)
cv2.imshow("image", img)
cv2.waitKey(0)