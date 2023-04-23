import math
import cv2
import matplotlib.pyplot as plt
import csv
import os
import numpy as np


def find_all_images_paths():
    image_paths = []
    current_path = os.getcwd()
    image_dir = current_path + r"\Images"
    if not os.path.isdir(image_dir):
        raise ValueError('cant find:', image_dir)

    for root, dirs, files in os.walk(image_dir): #walk throgh all the images file in th Images directory
        for file in files:
            image_paths.append(os.path.join(root, file))
    return image_paths


def find_stars(path, index):
    img = cv2.imread(path)
    if img is None:
        raise ValueError('cant find:', path)
        return False

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    # find all the points that creates the edge-line of each of each star
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    numOfStars = len(contours)
    if numOfStars == 0:
        print('didnt found any isars in:', path)
        return False

    print("the number of stars in picture " + str(index) + " is " + str(numOfStars))
    idx=1
    stars=[]
    for star in contours:   #iterate over all the stars we found
        (x, y), radius = cv2.minEnclosingCircle(star) #find the radius of the star
        brightness = np.mean(img_gray[int(y) - math.floor(radius):int(y) + math.floor(radius), int(x) - math.floor(radius):int(x) + math.floor(radius)])
        stars.append({"idx":idx,"x":x,"y":y,"radius":radius,"brightness":brightness})
        idx+=1
    print("\n")
    return stars



if __name__ == '__main__':
    paths = find_all_images_paths()
    # creating csv file for every image that contain all the stars we found in the image
    # every row in the csv file representing a star - idx, x, y, radius, brightness
    index = 1
    for path in paths:
        stars=find_stars(path=path, index=index)
        if stars!=False:
            with open(f"stars_of_img_{index}.csv", mode="w") as csvfile:
                f = stars[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=f)
                for row in stars:
                    writer.writerow(row)
        index += 1
