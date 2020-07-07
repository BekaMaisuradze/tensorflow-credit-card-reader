import os

######## Create our dataset directories ########

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None, 0
    
for i in range(0,10):
    directory_name = "data/credit_card/train/"+str(i)
    print(directory_name)
    makedir(directory_name) 

for i in range(0,10):
    directory_name = "data/credit_card/test/"+str(i)
    print(directory_name)
    makedir(directory_name)


################################################


######### Data Augmentation functions ##########
# add_noise() - This function introduces some noise elements to the image
# pixelate() - This function re-sizes the image then upscales/upsamples it. This degrades the quality and is meant to simulate blur to the image from either a shakey or poor quality camera.
# stretch() - This simulates some variation in re-sizing. It stretches the image to a small random amount
# pre_process() - This is a simple function that applies OTSU Binarization to the image and re-sizes it. I use this on the extracted digits. To create a clean dataset akin to the MNIST style format.
# digit_augmentation() - This one simply uses the other image manipulating functions by calling them randomly
################################################

import cv2
import numpy as np 
import random
import cv2
from scipy.ndimage import convolve



def digit_augmentation(frame, dim = 32):
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    random_num = np.random.randint(0,9)

    if (random_num % 2 == 0):
        frame = add_noise(frame)
    if(random_num % 3 == 0):
        frame = pixelate(frame)
    if(random_num % 2 == 0):
        frame = stretch(frame)
    frame = cv2.resize(frame, (dim, dim), interpolation = cv2.INTER_AREA)

    return frame 

def add_noise(image):
    prob = random.uniform(0.01, 0.05)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy

def pixelate(image):
    dim = np.random.randint(8,12)
    image = cv2.resize(image, (dim, dim), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (16, 16), interpolation = cv2.INTER_AREA)
    return image

def stretch(image):
    ran = np.random.randint(0,3)*2
    if np.random.randint(0,2) == 0:
        frame = cv2.resize(image, (32, ran+32), interpolation = cv2.INTER_AREA)
        return frame[int(ran/2):int(ran+32)-int(ran/2), 0:32]
    else:
        frame = cv2.resize(image, (ran+32, 32), interpolation = cv2.INTER_AREA)
        return frame[0:32, int(ran/2):int(ran+32)-int(ran/2)]
    
def pre_process(image, inv = False):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
        pass
    
    if inv == False:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(th2, (32,32), interpolation = cv2.INTER_AREA)
    return resized
	
	
######## Generate Train Data #########################
	
# Creating 1000 Images for each digit in creditcard_digits1 - TRAINING DATA

cc1 = cv2.imread('data/creditcard_digits1.jpg', 0)

''' uncomment this code to display credit card digit image
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("cc1", th2)
cv2.imshow("creditcard_digits1", cc1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

region = [(2, 19), (50, 72)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0,10):
    if i > 0:
        top_left_x = top_left_x + 59
        bottom_right_x = bottom_right_x + 59

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
	
    for j in range(0,1000):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv = True)
        cv2.imwrite("data/credit_card/train/"+str(i)+"./_1_"+str(j)+".jpg", roi_otsu)
cv2.destroyAllWindows()


# Creating 1000 Images for each digit in creditcard_digits2 - TRAINING DATA

cc1 = cv2.imread('data/creditcard_digits2.jpg', 0)

''' uncomment this code to display credit card digit image
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("cc1", th2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

region = [(0, 0), (35, 48)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0,10):   
    if i > 0:
        top_left_x = top_left_x + 35
        bottom_right_x = bottom_right_x + 35

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
	
    for j in range(0,1000):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv = False)
        cv2.imwrite("data/credit_card/train/"+str(i)+"./_2_"+str(j)+".jpg", roi_otsu)
        # cv2.imshow("otsu", roi_otsu)
        # print("-")
        # cv2.waitKey(0)

cv2.destroyAllWindows()

################################################


######## Generate Test Data ######################

cc1 = cv2.imread('data/creditcard_digits1.jpg', 0)

''' uncomment this code to display credit card digit image
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("cc1", th2)
cv2.imshow("creditcard_digits1", cc1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

region = [(2, 19), (50, 72)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0,10):
    if i > 0:
        top_left_x = top_left_x + 59
        bottom_right_x = bottom_right_x + 59

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
	
    for j in range(0,200):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv = True)
        cv2.imwrite("data/credit_card/test/"+str(i)+"./_1_"+str(j)+".jpg", roi_otsu)
cv2.destroyAllWindows()
# Creating 200 Images for each digit in creditcard_digits2 - TEST DATA

cc1 = cv2.imread('data/creditcard_digits2.jpg', 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("cc1", th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

region = [(0, 0), (35, 48)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0,10):   
    if i > 0:
        top_left_x = top_left_x + 35
        bottom_right_x = bottom_right_x + 35

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
	
    for j in range(0,200):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv = False)
        cv2.imwrite("data/credit_card/test/"+str(i)+"./_2_"+str(j)+".jpg", roi_otsu)
        # cv2.imshow("otsu", roi_otsu)
        # print("-")
        # cv2.waitKey(0)
cv2.destroyAllWindows()


