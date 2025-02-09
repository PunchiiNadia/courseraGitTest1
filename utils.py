import cv2
import numpy as np

import cv2
import numpy as np

def stackImages(scale, imgArray, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor

    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                if labels[d][c] != "":
                    cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(ver, labels[d][c],
                                (eachImgWidth * c + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 2)

    return ver


def recContour(contours):
    recCon = []
    for i in contours:
        area = cv2.contourArea(i)
        # print(f"area: {area}")
        if area > 50:
            ##---- i = given contour ----##
            peri = cv2.arcLength(i, True) ##---- Total length ----##
            approx = cv2.approxPolyDP(i, 0.02*peri,True)
            # print(f"Corner Points: {approx}")
            if len(approx) == 4:
                recCon.append(i)

    # print(f"Rectangle Corner Contour Area: {recCon}")
    recCon = sorted(recCon, key=cv2.contourArea, reverse=True) ##---- Sorting from the biggest to smallest rectangle ----##

    return recCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)  ##---- Total length ----##
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)

    return approx

def reorder(myPoints):
    # Reordering the matrix because it does not be in the proper position
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)] # [0, 0]
    myPointsNew[3] = myPoints[np.argmax(add)] # [w, h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # [w, 0]
    myPointsNew[2] = myPoints[np.argmax(diff)] # [0, h]

    return myPointsNew

# EACH BUBBLES
def splitBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 4)
        for box in cols:
            boxes.append(box)
    # cv2.imshow("img", boxes[1])
    return boxes
# Need clarify
def showAnswers(img, myIndex, grading, ans, questions, choices):
    secWidth = int(img.shape[1]/choices)
    secHeight = int(img.shape[0]/questions)

    for x in range(0, questions):
        myAns = myIndex[x]
        centerX = (myAns * secWidth) + secWidth//2
        centerY = (x * secHeight) + secHeight//2

        if grading[x] == 1:
            myColor = (0, 255, 0)
        else:
            myColor = (0, 0, 255)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secWidth)+secWidth//2, (x * secHeight) + secHeight//2), 20, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (centerX, centerY), 50, myColor, cv2.FILLED)
    return img