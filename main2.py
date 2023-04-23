import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from random import sample



images_paths=["Images/IMG_3046.jpg","Images/IMG_3047.jpg","Images/IMG_3048.jpg","Images/IMG_3049.jpg","Images/IMG_3050.jpg","Images/IMG_3051.jpg","Images/IMG_3053.jpg","Images/IMG_3054.jpg","Images/IMG_3055.jpg","Images/IMG_3056.jpg","Images/IMG_3057.jpg","Images/IMG_3058.jpg","Images/IMG_3059.jpg","Images/IMG_3060.jpg","Images/IMG_3061.jpg","Images/IMG_3062.jpg","Images/IMG_3063.jpg"]

testing=False

if testing:
    img1_path = "../EX1_Star_Tracking/test_images/ST_db1.png"
    img2_path = "../EX1_Star_Tracking/test_images/ST_db2.png"
else:
    # get the path of img1 and img2 by choosing random
    s = sample(images_paths, 2)
    img1_path = s[0]
    img2_path = s[1]

img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # query image
img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # train image

# check for valid images
if img1 is None:
    raise ValueError('img1 is None')
if img2 is None:
    raise ValueError('img2 is None')

# Initiate ORB detector (feature detector)
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object (the matcher)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

for match in matches:
    img1_index = match.queryIdx
    img2_index = match.trainIdx
    (img1_x, img1_y) = kp1[img1_index].pt
    (img2_x, img2_y) = kp2[img2_index].pt
    print(f"keypoint ({img1_x}, {img1_y}) in image 1 is matched to keypoint ({img2_x}, {img2_y}) in image 2")

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first n matches.
n = len(matches)
if n > len(matches):
    raise ValueError('cannot show more matches than found')
img_matches = cv.drawMatches(img1,kp1,img2,kp2,matches[:n],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# img_matches = cv.drawMatches(img1,kp1,img2,kp2,matches[:n],None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_matches),plt.show()