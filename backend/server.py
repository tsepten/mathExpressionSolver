from flask import Flask, request
from flask_cors import CORS
import urllib
import base64
import json
from PIL import Image
# Import the necessary packages

# image segmentation
import cv2
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

matplotlib.use('agg')

# image classification
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import img_to_array
import pickle

# symbol parser and equation solver
from sympy import *
from sympy import parse_expr
from sympy import parse_expr, symbols, sqrt
from sympy.parsing.sympy_parser import transformations
from sympy.parsing.sympy_parser import standard_transformations
from sympy.parsing.sympy_parser import implicit_multiplication
from sympy.parsing.sympy_parser import function_exponentiation
from sympy import symbols
from sympy import Eq
from sympy.solvers import solve

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers',
#                          'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods',
#                          'GET,PUT,POST,DELETE,OPTIONS')
#     return response

# we create the image segment class with various functions


class imageSegment:

    def __init__(self):
        return

    def plotBeforeAfter(self, oldImage, newImage, boundingBoxes):
        # find max height and min height of bounding boxes
        maxHeight = 0
        minHeight = 9999
        for bbox in boundingBoxes:
            if (bbox[1] + bbox[3] > maxHeight):
                maxHeight = bbox[1] + bbox[3]
            if (bbox[1] > maxHeight):
                maxHeight = bbox[1]
            if (bbox[1] + bbox[3] < minHeight):
                minHeight = bbox[3] + bbox[1]
            if (bbox[1] < minHeight):
                minHeight = bbox[1]
        print("Max height:", maxHeight)
        print("Min height:", minHeight)

        fig, ax = plt.subplots(1, 1)  # creates a grid of 1 row and 2 columns

        ax.imshow(newImage, cmap="gray")  # plots on the first subplot
        # ax.set_title("Contoured")  # sets the title of the first subplot

        for bbox in boundingBoxes:
            # Draw rectangle on the first subplot (contoured image)
            # bbox is a tuple with format (x, y, width, height)
            rect = Rectangle((bbox[0], bbox[1]),
                             bbox[2],
                             bbox[3],
                             fill=False,
                             edgecolor='red',
                             linewidth=2)
            ax.add_patch(rect)

        ax.axis('off')  # hide the axis on first subplot
        plt.ylim(maxHeight + 10, minHeight - 10)
        plt.savefig('./contours.png', transparent=True)
        fig, ax = plt.subplots(1, 1)

        ax.imshow(oldImage, cmap="gray")  # plots on the second subplot
        ax.axis('off')  # hide the axis on second subplot
        plt.ylim(maxHeight + 10, minHeight - 10)
        plt.savefig('./old.png', transparent=True)

        # plt.tight_layout()
        return

    def boundingBoxDetails(self, boundingBox):
        print("Bounding boxes")
        for i in range(len(boundingBox)):
            print("Segment", str(i + 1) + ":", str(boundingBox[i]))

    # plot the segmented images
    def plotSegmentedImages(self, segmentedImages, max_images_per_row=5):

        # Calculate the number of rows needed
        num_rows = len(segmentedImages) // max_images_per_row
        if len(segmentedImages) % max_images_per_row:
            num_rows += 1

        # create grid of subplots
        fig, axs = plt.subplots(num_rows,
                                max_images_per_row,
                                figsize=(10, 2 * num_rows))

        # to handle case of single row or single column, we ensure axs is always a 2-D array
        if axs.ndim == 1:
            axs = axs.reshape(num_rows, -1)

        # loop through each image and display it on the subplot
        for i, img in enumerate(segmentedImages):
            row = i // max_images_per_row
            col = i % max_images_per_row
            axs[row, col].imshow(img, cmap='gray')
            axs[row, col].set_title('Symbol {}'.format(i))
            axs[row, col].axis('off')

        # if there are any empty subplots at the end, we hide them
        for j in range(i + 1, num_rows * max_images_per_row):
            row = j // max_images_per_row
            col = j % max_images_per_row
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.savefig('./segmented.png', transparent=False)

    def segmentationModel(self, fileName=str()):

        # Read the image and turn to grayscale
        image = cv2.imread(fileName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image to 8-bit
        gray = cv2.convertScaleAbs(gray)

        # Perform thresholding to create a binary image
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]

        # sort cnts and bounding box from left to right
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        boundingBoxes = sorted(boundingBoxes, key=lambda x: x[0])

        # Create an array to store segmented images
        segmentedImages = []
        contourOnImage = image.copy()
        _ = cv2.drawContours(contourOnImage, contours, -1, (0, 255, 0), 3)

        # Iterate over the contours and hierarchy information
        for i, contour in enumerate(contours):
            # Check if contour is an external contour (no parent)
            if hierarchy[0][i][3] == -1:
                # Create a blank mask image
                mask = np.zeros_like(image)

                # Draw the contour on the mask
                cv2.drawContours(mask, contours, i, (255, 255, 255),
                                 cv2.FILLED)

                # Apply the mask to the original image
                segmented_image = cv2.bitwise_and(image, mask)
                segmented_image[mask == 0] = 255

                # Store the segmented image in the array
                segmented_image = cv2.cvtColor(segmented_image,
                                               cv2.COLOR_BGR2GRAY)
                segmented_image[segmented_image >= 150] = 255
                segmented_image[segmented_image < 150] = 0
                segmentedImages.append(
                    segmented_image[boundingBoxes[i][1]:boundingBoxes[i][1] +
                                    boundingBoxes[i][3],
                                    boundingBoxes[i][0]:boundingBoxes[i][0] +
                                    boundingBoxes[i][2]])

        # return old image, contoured image, segmentedImages, and bounding boxes
        return image, contourOnImage, segmentedImages, boundingBoxes

    # def segmentationModel(self, fileName=str()):

    #     # imageName = "test_image/testImage_hard.png"

    #     # Read the image and turn to grayscale
    #     image = cv2.imread(fileName)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Perform thresholding to create a binary image
    #     _, binary = cv2.threshold(gray, 0, 255,
    #                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #     # Find contours with hierarchy
    #     contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
    #                                            cv2.CHAIN_APPROX_SIMPLE)
    #     boundingBoxes = [cv2.boundingRect(c) for c in contours]

    #     # sort cnts and bounding box from left to right
    #     contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    #     boundingBoxes = sorted(boundingBoxes, key=lambda x: x[0])

    #     # Create an array to store segmented images
    #     segmentedImages = []
    #     contourOnImage = image.copy()
    #     _ = cv2.drawContours(contourOnImage, contours, -1, (0, 255, 0), 3)

    #     # Iterate over the contours and hierarchy information
    #     for i, contour in enumerate(contours):
    #         # Check if contour is an external contour (no parent)
    #         if hierarchy[0][i][3] == -1:
    #             # Create a blank mask image
    #             # if(boundingBoxes[i][2]*boundingBoxes[i][3] < 500):
    #             #     continue
    #             mask = np.zeros_like(image)
    #             # print(contour[0]/ratio)

    #             # Draw the contour on the mask
    #             cv2.drawContours(mask, contours, i, (255, 255, 255),
    #                              cv2.FILLED)

    #             # Apply the mask to the original image
    #             segmented_image = cv2.bitwise_and(image, mask)
    #             segmented_image[mask == 0] = 255

    #             # Store the segmented image in the array
    #             segmented_image = cv2.cvtColor(segmented_image,
    #                                            cv2.COLOR_BGR2GRAY)
    #             segmented_image[segmented_image >= 150] = 255
    #             segmented_image[segmented_image < 150] = 0
    #             segmented_image = segmented_image / 255
    #             segmentedImages.append(
    #                 segmented_image[boundingBoxes[i][1]:boundingBoxes[i][1] +
    #                                 boundingBoxes[i][3],
    #                                 boundingBoxes[i][0]:boundingBoxes[i][0] +
    #                                 boundingBoxes[i][2]])

    #     # return old image, contoured image, segmentedImages, and bounding boxes
    #     return image, contourOnImage, segmentedImages, boundingBoxes


# resize images
def resize_and_stretch(img, target_size=(32, 32), stretch_margin=0.20):
    # convert the image to grayscale if it's in color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # first, resize the image while keeping the aspect ratio
    original_height, original_width = img.shape[:2]
    ratio = min(target_size[0] / original_height,
                target_size[1] / original_width)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # if the image is still too small, stretch it up to 20% to reach the target size
    if img.shape[0] < target_size[0] or img.shape[1] < target_size[1]:
        stretch_ratio = min(target_size[0] / img.shape[0],
                            target_size[1] / img.shape[1])
        stretch_ratio = min(stretch_ratio, 1 + stretch_margin)
        new_size = (int(img.shape[1] * stretch_ratio),
                    int(img.shape[0] * stretch_ratio))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # finally, if the image is still too small (due to the stretch limit), pad it to reach the target size
    delta_w = target_size[1] - img.shape[1]
    delta_h = target_size[0] - img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=255)
    img[img < 250] = 0
    img[img >= 250] = 255
    return img


# load our model

cnnModelFilePath = 'cnn_v2/cnn_v2.h5'  # CHANGE ME
classifier = load_model(cnnModelFilePath)
# load in label encoder (to decode)

labelEncoderFilePath = 'cnn_v2/label_encoder.pkl'  # CHANGE ME
pkl_file = open(labelEncoderFilePath, 'rb')
le = pickle.load(pkl_file)
pkl_file.close()


def makePrediction(inputImages):
    predictions = []

    for img in inputImages:
        # add an extra dimension for the channels and normalize the image
        img = np.expand_dims(img, axis=-1)
        img = img.astype('float32') / 255.0
        # add an extra dimension for the batch size
        img = np.expand_dims(img, axis=0)
        predictions.append(classifier.predict(img))

    return predictions


def getPredictedLabels(predictions):
    # get the class with the highest probability for each prediction
    predicted_classes = [np.argmax(pred[0]) for pred in predictions]

    # use the inverse transform of the label encoder to get the original labels
    predicted_labels = le.inverse_transform(predicted_classes)

    return predicted_labels


def parse(predicted_labels, boundingBoxes):
    rootWrappers = {}

    currentSqrtStartIndex = 0
    sqrtBounds = (0, 0)
    currentWrapper = []

    for i in range(len(predicted_labels)):
        # we are currently inside a squareroot
        if (sqrtBounds != (0, 0)):
            currentBound = (boundingBoxes[i][0],
                            boundingBoxes[i][0] + boundingBoxes[i][2])

            # check if current symbol belongs inside square root bounds
            if (currentBound[0] >= sqrtBounds[0]
                    and currentBound[1] <= sqrtBounds[1]):
                print("Added current bound of (" + str(currentBound[0]),
                      str(currentBound[1]) + ")")
                currentWrapper.append(i)
            else:
                rootWrappers[currentSqrtStartIndex] = currentWrapper.copy()
                currentWrapper.clear()
                sqrtBounds = (0, 0)

        if (predicted_labels[i] == "\\sqrt{}"):
            currentSqrtStartIndex = i
            sqrtBounds = (boundingBoxes[i][0],
                          boundingBoxes[i][0] + boundingBoxes[i][2])
            print("A square root found at index", i,
                  "has bounds: (" + str(sqrtBounds[0]),
                  str(sqrtBounds[1]) + ")")

    if (sqrtBounds != (0, 0) and currentWrapper != []):
        rootWrappers[currentSqrtStartIndex] = currentWrapper.copy()

    parsed = list(predicted_labels.copy())

    for i in range(len(predicted_labels)):
        currLabel = predicted_labels[i]
        if (currLabel == "\\ast"):
            parsed[i] = "*"
        if (currLabel == "\\sqrt{}"):
            parsed[i] = "sqrt("
            endIndex = i + len(rootWrappers[i]) + 1
            parsed.insert(endIndex, ')')

    return parsed


# Assumes perfect parsed
def evaluate3(expression):
    # First, convert the list to a string
    str_expr = ''.join(expression)

    # Then, parse the string as a sympy expression
    sympy_expr = sympify(str_expr)

    # Finally, evaluate the sympy expression
    result = sympy_expr.evalf()

    return result


def plotPredictionProbabilities(predictions, inputImages):
    class_names = le.classes_

    # Create a bar chart of the prediction probabilities
    for imageIndex in range(len(predictions)):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        # Plot the predicted probabilities as a bar chart
        axs[0].bar(class_names, predictions[imageIndex][0])
        axs[0].set_title(f"Prediction for Image {imageIndex}")
        axs[0].set_xlabel("Class")
        axs[0].set_ylabel("Probability")

        # Plot the image on the right
        axs[1].imshow(inputImages[imageIndex], cmap='gray')
        axs[1].set_title(f"Image {imageIndex}")
        # axs[1].axis('off')  # Hide the axes on the image plot

        plt.tight_layout()
        plt.show()
    return


# check if img exists
def checkImageExist(filePath):
    image = cv2.imread(filePath)
    if image is None:
        print("Check path again:", filePath)
        print("Image does not exist / failed to load")
        return -1
    return 0


folderPath = "."  #path to input image folder


def runModel(fileName):

    # check if image exists
    if checkImageExist(folderPath + "/" + fileName) == -1:
        return -99999999  #"FAILED"

    # run image segment
    imgSeg = imageSegment()
    oldImage, contouredImage, segmentedImages, boundingBoxes = imgSeg.segmentationModel(
        folderPath + "/" + fileName)
    imgSeg.plotBeforeAfter(oldImage, contouredImage, boundingBoxes)
    imgSeg.plotSegmentedImages(segmentedImages)

    # Resize each image in segmentedImages, getting it ready for predictions
    inputImages = [
        resize_and_stretch(img, target_size=(32, 32))
        for img in segmentedImages
    ]
    # plot input images
    for i in range(len(inputImages)):
        plt.subplot(1, len(inputImages), i + 1)
        plt.imshow(inputImages[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./inputImages.png', transparent=True)

    predictions = makePrediction(inputImages)  # get raw undecoded predictions

    predictedLabels = getPredictedLabels(predictions)  # get predicted labels

    try:
        parsedExpression = parse(predictedLabels,
                                 boundingBoxes)  # get parsed expression
    except:
        parsedExpression = "Error"
    try:
        solvedExpression = evaluate3(parsedExpression)  # solve the expression
    except:
        solvedExpression = "Error"

    return parsedExpression, solvedExpression  # return answer


@app.post("/")
def helloWorld():
    response = urllib.request.urlopen(request.get_json()['dataURL'])

    with open('equation.jpg', 'wb') as f:
        f.write(response.file.read())
        parsedExpression, solvedExpression = runModel('equation.jpg')

    def bbox(im):
        a = np.array(im)[:, :, :3]  # keep RGB only
        m = np.any(a != [255, 255, 255], axis=2)
        coords = np.argwhere(m)
        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
        return (x0 - 10, y0 - 10, x1 + 10, y1 + 10)

    im = Image.open('old.png')
    im2 = im.crop(bbox(im))
    im2.save('old.png')
    im = Image.open('contours.png')
    im2 = im.crop(bbox(im))
    im2.save('contours.png')
    im = Image.open('segmented.png')
    im2 = im.crop(bbox(im))
    im2.save('segmented.png')
    with open('old.png', 'rb') as f:
        binary_fc = f.read()
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')

        oldurl = f'data:image/png;base64,{base64_utf8_str}'
    with open('contours.png', 'rb') as f:
        binary_fc = f.read()
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')

        contoursurl = f'data:image/png;base64,{base64_utf8_str}'
    with open('segmented.png', 'rb') as f:
        binary_fc = f.read()
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')

        segmentedurl = f'data:image/png;base64,{base64_utf8_str}'
    return {
        'parsedExpression': json.dumps(parsedExpression),
        'solvedExpression': str(solvedExpression),
        'oldurl': oldurl,
        'contoursurl': contoursurl,
        'segmentedurl': segmentedurl
    }


app.run(debug=True)
