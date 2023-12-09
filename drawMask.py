import numpy as np
import cv2


import cv2 
  
# img = None  
# img = np.zeros((500, 500, 3), dtype='float64')
  
# while True: 
#     cv2.imshow("Title of Popup Window", img) 
      
#     if cv2.waitKey(10) & 0xFF == 27: 
#         break
          
# cv2.destroyAllWindows() 

# image = None

# def draw(event, mouseX, mouseY, flag, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("YO")
#         cv2.circle(image, (mouseX, mouseY), 5, (0, 255, 0), 2)


mouseDown = False

def createWindow(img):
    # global image
    image = np.copy(img)

    def draw_circle(event, x, y, flags, param): 
        global mouseDown
        if event == cv2.EVENT_LBUTTONDOWN: 
            print("hello") 
            mouseDown = True
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1) #bgr
        elif event == cv2.EVENT_LBUTTONUP:
            print("up") 
            mouseDown = False
        
        if mouseDown and event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1) #bgr

    
    winname = "Draw Mask Window"
    cv2.namedWindow(winname = winname) 
    cv2.setMouseCallback(winname, draw_circle) 
    while True:
        cv2.imshow(winname, image) 
        if cv2.waitKey(10) == 27: 
            break
    cv2.destroyAllWindows() 
    return image

def getMask(baseImg):
    drawnImage = createWindow(baseImg)
    # Debug code below
    # cv2.imshow("drawnImage", drawnImage) 
    # cv2.waitKey(0) 
    # cv2.imshow("orig", baseImg) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    maskColored = drawnImage - baseImg
    mask = np.sum(maskColored, axis=-1)
    mask = np.clip(mask, 0, 1)
    # print(np.max(mask))
    return mask


# img_orig = np.zeros((500, 500, 3), dtype='float64')
# image = getMask(img_orig)
# cv2.imshow("result", image) 
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

