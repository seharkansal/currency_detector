"""
Coin recognition, real life application
task: calculate the value of coins on picture
"""

import cv2
import numpy as np

coins = cv2.imread('./input_image/koruny_black3.jpg')

def image_resize(coins,width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = coins.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return coins

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(coins, dim, interpolation = inter)

    # return the resized image
    return resized

def detect_coins():
   
    image= image_resize(coins,height=620)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        img,  # source image
        cv2.HOUGH_GRADIENT,  # type of detection
        1,
        100,
        param1=180,
        param2=30,
        minRadius=10,  # minimal radius
        maxRadius=500,  # max radius
    )

    coins_copy = image.copy()

    print(circles[0])
    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        coins_detected = cv2.circle(
            coins_copy,
            (int(x_coor), int(y_coor)),
            int(detected_radius),
            (0, 255, 0),
            4,
        )
    # cv2.imshow('circled',coins_detected)
    # cv2.waitKey(0)
    try:
        cv2.imwrite("output_image/coin_amount/koruny_test_Hough.jpg", coins_detected)
        print("Image written")
    except:
        print("Problem")
   
    return circles

def calculate_amount():
    koruny = {
        "1 CZK": {
            "value": 1,
            "radius": 20,
            "ratio": 1,
            "count": 0,
        },
        "2 CZK": {
            "value": 2,
            "radius": 21.5,
            "ratio": 1.075,
            "count": 0,
        },
        "5 CZK": {
            "value": 5,
            "radius": 23,
            "ratio": 1.15,
            "count": 0,
        },
        "10 CZK": {
            "value": 10,
            "radius": 24.5,
            "ratio": 1.225,
            "count": 0,
        },
        "20 CZK": {
            "value": 20,
            "radius": 26,
            "ratio": 1.3,
            "count": 0,
        },
        "50 CZK": {
            "value": 50,
            "radius": 27.5,
            "ratio": 1.375,
            "count": 0,
        },
    }

    circles = detect_coins()
    radius = []
    coordinates = []

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        radius.append(detected_radius)
        coordinates.append([x_coor, y_coor])

    smallest = min(radius)
    tolerance = 0.0375
    total_amount = 0
    try:
        coins_circled = cv2.imread('output_image/coin_amount/koruny_test_Hough.jpg', 1)
        print("image read")
    except:
        print("image not written")
    font = cv2.FONT_HERSHEY_SIMPLEX

    for coin in circles[0]:
        ratio_to_check = coin[2] / smallest
        coor_x = coin[0]
        coor_y = coin[1]

        for koruna in koruny:
            value = koruny[koruna]['value']
            print(koruna)
            print(ratio_to_check)
            print(koruny[koruna]['ratio'])
            print(abs(ratio_to_check-koruny[koruna]['ratio']))
            print("----------------")
            if abs(ratio_to_check - koruny[koruna]['ratio']) <= tolerance:
                print("yes it is less")
                print(koruny[koruna]['value'])
                koruny[koruna]['count'] += 1
                total_amount += koruny[koruna]['value']
                coins_circled = cv2.putText(coins_circled, str(value), (int(coor_x), int(coor_y)), font, 1,
                            (0, 0, 0), 4)

    print(f"The total amount is {total_amount} CZK")
    for koruna in koruny:
        pieces = koruny[koruna]['count']
        print(f"{koruna} = {pieces}x")

    cv2.imwrite("output_image/coin_amount/koruny_hodnota.jpg", coins_circled)
    cv2.imshow('output',coins_circled)
    cv2.waitKey(0)

if __name__ == "__main__":
    calculate_amount()