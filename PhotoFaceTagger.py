import dlib
from skimage import io
import os
from PIL import Image
from pyfacy import face_clust
from pyfacy import utils
import json
import cv2
from loadbar import LoadBar
import time
import glob
from threading import Thread
import subprocess
import sys

def detect_faces(image):
    # we fo a quick haarcascade to speed up to process and use the neural net only if there is a face.
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) > 0:
        # Create a face detector
        face_detector = dlib.get_frontal_face_detector()

        # Run detector and get bounding boxes of the faces on image.
        detected_faces = face_detector(image, 1)
        face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
    else:
        face_frames = []
    return face_frames

def getListOfFiles(dirName):
    # create a list of absolute path to the file in the directory and the following sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def excludeNonPictureFromFileList(fileList):
    ValidFormat = ["rgb", "gif", "pbm", "pgm", "ppm", "tiff", "rast", "xbm", "jpeg", "jpg", "bmp", "png", "webp", "exr"]
    listOfPictures = []
    # exclude everyfile that is not an image
    for file in fileList:
        try:
            img = Image.open(file)
            if str(img.format).lower() in ValidFormat:
                listOfPictures.append(file)
        except:
            print("Not a picture : "+str(file))
    return listOfPictures

def FaceExtraction(listOfPictures, threadIndex):
    faceIndex = 0
    picIndex = 0
    faceDB = {}
    resolution = 0.02
    lastStep = resolution
    for pic in listOfPictures:
        picIndex = picIndex + 1
        if picIndex/len(listOfPictures) > lastStep:
            lastStep = lastStep + resolution
            print(str(picIndex)+" out of " + str(len(listOfPictures)) + " For Thread " + str(threadIndex))
        # Load image
        try:
            image = io.imread(pic)

            # check if there is a face
            detected_faces = detect_faces(image)

            # Iterate over each face, Crop it and save it as jpg in a created Face Folder
            # create a .json with the link between a face and its origin picture
            for n, face_rect in enumerate(detected_faces):
                face = Image.fromarray(image).crop(face_rect)  # crop the picture
                path = os.path.join("Faces", str(faceIndex*100+threadIndex) + ".jpg")  # name its face from 0.jpg to N.jpg in Faces folder
                faceIndex = faceIndex + 1  # keep a counter to name the face.jpg file
                face.save(path)  # write the picture to disk
                face = utils.load_image(path)  # not super efficient but the lib don't accept the picture already loaded
                encodings = utils.img_to_encodings(face)
                faceDB[str(faceIndex*100+threadIndex) + ".jpg"] = [pic, str(encodings)]  # link faces, fingerprint and origin picture
        except Exception as e:
            print(str(e) + " with picture :  " + pic)
    data = {}
    try:
        with open('Faces.json') as json_file:
            data = json.load(json_file)
    except Exception as e:
        print(e)
    print(faceDB)
    with open('Faces.json', 'w') as fp:
        faceDB.update(data)
        json.dump(faceDB, fp)
    print("Thread "+str(threadIndex)+" Done")


def main(path):
    ListOfFile = getListOfFiles(path)
    listOfPictures = excludeNonPictureFromFileList(ListOfFile)
    corpusSize = len(listOfPictures)
    print(str(corpusSize) + " Pictures loaded from folder and subfolder")
    os.makedirs("Faces", exist_ok=True)
    if os.path.exists(os.path.join("Faces", "Faces.json")):  # check if the face extraction was completed before
        skipFaceExtraction = True
    else:
        skipFaceExtraction = False
        files = glob.glob('Faces/*')
        for f in files:
            os.remove(f)

    if not skipFaceExtraction:  # Face extraction from picture corpus
        NumberOfCore = 1
        if sys.platform == 'win32':
            NumberOfCore = (int)(os.environ['NUMBER_OF_PROCESSORS'])
        else:
            NumberOfCore = (int)(os.popen('grep -c cores /proc/cpuinfo').read())
        NumberOfCore = NumberOfCore -1
        print("Starting "+str(NumberOfCore)+" Threads")
        numberOfPicPerCore = int(len(listOfPictures)/NumberOfCore)
        print(str(numberOfPicPerCore) + " pics per core")
        output = [listOfPictures[i:i + numberOfPicPerCore] for i in range(0, len(listOfPictures), numberOfPicPerCore)]
        threadlist = []
        for i in range(0, NumberOfCore):
            try:
                thread = Thread(target=FaceExtraction, args=(output[i], i))  # start as many thread as there are core
                threadlist.append(thread)
            except Exception as e:
                print("Error: unable to start thread "+str(i))  # we resplit the list equally among the remaining thread
                print(e)
                print("------------------------------------------")
                for j in range(i,NumberOfCore): # can fuck up if last thread is not available...
                    listOfPictures = listOfPictures + output[j]
                    numberOfPicPerCore = int(len(listOfPictures) / (NumberOfCore-i))
                    output = [listOfPictures[k:k + numberOfPicPerCore] for k in range(0, len(listOfPictures), numberOfPicPerCore)]
        for x in threadlist:
            x.start()
        for x in threadlist:
            x.join()

    print("creating the face model")
    # Create object for Cluster class with your source path(only contains jpg images)
    mdl = face_clust.Face_Clust_Algorithm("Faces")
    # Load the faces to the algorithm
    print("loading the face model")
    mdl.load_faces()
    # Save the group of images to custom location(if the arg is empty store to current location)
    print("Running the clustering model")
    mdl.save_faces("FaceCluster")


if __name__ == "__main__":
    main("/home/nathann/SortedPictures")