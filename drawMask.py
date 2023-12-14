import numpy as np
import cv2
  
mouseDown = False

def createWindow(img):
    # global image
    image = np.copy(img)

    def draw_circle(event, x, y, flags, param): 
        global mouseDown
        if event == cv2.EVENT_LBUTTONDOWN: 
            # print("hello") 
            mouseDown = True
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1) #bgr
        elif event == cv2.EVENT_LBUTTONUP:
            # print("up") 
            mouseDown = False
        
        if mouseDown and event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1) #bgr

    
    winname = "Draw Mask Window"
    cv2.namedWindow(winname = winname) 
    cv2.setMouseCallback(winname, draw_circle) 
    while True:
        cv2.imshow(winname, image) 
        if cv2.waitKey(10) == 27: # press esc to end drawing mask
            break
    cv2.destroyAllWindows() 
    return image

def getMask(baseImg):
    drawnImage = createWindow(baseImg)
    maskColored = drawnImage - baseImg
    mask = np.sum(maskColored, axis=-1)
    mask = np.clip(mask, 0, 1)
    # mask = cv2.GaussianBlur(mask,(3,3),0)
    return mask