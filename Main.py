import cv2
import numpy as np
import utils

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
        recCon = utils.recContour(contours)
        biggestContour = utils.getCornerPoints(recCon[0])
        gradePoints = utils.getCornerPoints(recCon[2])

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 255), 10)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (0, 0, 255), 10)

            biggestContour = utils.reorder(biggestContour)
            gradePoints = utils.reorder(gradePoints)

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

            boxes = utils.splitBoxes(imgThre)

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
            imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
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

    imgStacked = utils.stackImages(0.3, imgArray, labels)

    cv2.imshow("Stacked Images", imgStacked)
    if cv2.waitKey(1) & ord('s'):
        cv2.imwrite("FinalResult.jpg", imgFinal)
        cv2.waitKey(300)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

