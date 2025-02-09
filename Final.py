import cv2
import numpy as np

################
path = "1 (5).png"
widthImg = 700
heightImg = 700
questions = 5
choices = 4
ans = [1, 2, 0, 2, 1]
webcamFeed = True
camNum = 0
################

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
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    ## Needs Clarify
    ## The similarities and the common things between: Biggest, Smallest, Max, Min, Width, Height ?
    ## Why does this step is neccessary? Are we resizing the matrix to be the same shape? # What is the purpose of these following code?
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

cap = cv2.VideoCapture(camNum)
cap.set(10, 150) # Brightness setting

while True:
    if webcamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # PREPROCESSING
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
        # FIND ALL CONTOURS
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

        # FIND RECTANGLES
        recCon = recContour(contours)
        biggestContour = getCornerPoints(recCon[0])
        gradePoints = getCornerPoints(recCon[2])

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 255), 10)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (0, 0, 255), 10)

            biggestContour = reorder(biggestContour)
            gradePoints = reorder(gradePoints)

            # What is the purpose of this ? Is it for transformation ? For what ?
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThre = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

            boxes =  splitBoxes(imgThre)

            # ---- GETTING NO ZERO PIXEL VALUES OF EACH BOX ----#
            # Free space to store and be ready for the information
            myPixelValue = np.zeros((questions, choices))
            countCol = 0
            countRow = 0
            # print(f"Original list of the matrix: {myPixelValue}")

            for image in boxes:
                totalPixels = cv2.countNonZero(
                    image)  # Image = Matrices in numerical numbers of black and white - totalPixels = Excluded parts (White)
                myPixelValue[countRow][countCol] = totalPixels  # Location; start with myPixelValue[0][0] = 2199
                # print(f"totalPixels: {totalPixels},  myPixelValue: {myPixelValue}")
                countCol += 1  # After you place the number inside 1st row and column = increment countCol as 1
                if (countCol == choices): countRow += 1;countCol = 0
                # print(myPixelValue)

            # FINDING INDEX VALUES OF THE MARKINGS
            myIndex = []
            for x in range(0, questions):
                array = myPixelValue[x]
                # print(f"Array: {array}")
                myIndexValue = np.where(array == np.amax(array))
                # print(myIndexValue[0])
                myIndex.append(myIndexValue[0][0])
            # print(myIndex)

            # GRADING
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            # print(grading)
            score = (sum(grading) / questions) * 100  # FINAL GRADE
            print(score)

            # DISPLAYING ANSWERS
            imgResult = imgWarpColored.copy()
            imgResult = showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade, str(int(score)) + '%', (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3,
                        (0, 255, 255), 3)

            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGrade = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGrade, 1, 1)

        imgBlank = np.zeros_like(img)
        imgArray = ([img, imgGray, imgBlur, imgCanny], [imgContours, imgBiggestContours, imgWarpColored, imgThre],
                    [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        imgArray = ([img, imgGray, imgBlur, imgCanny], [imgBlank, imgBlank, imgBlank, imgBlank],
                    [imgBlank, imgBlank, imgBlank, imgBlank])

    labels = [["Original", "Gray", "Blur", "Canny"], ["Contours", "Biggest Con", "Warp", "Threshold"],
              ["Result", "Raw Drawing", "Inv Warp", "Final"]]

    imgStacked = stackImages(0.3, imgArray, labels)

    cv2.imshow("Stacked Images", imgStacked)
    if cv2.waitKey(1) & ord('s'):
        cv2.imwrite("FinalResult.jpg", imgFinal)
        cv2.waitKey(300)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

