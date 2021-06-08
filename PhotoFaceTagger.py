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

    if not skipFaceExtraction:  # Face extraction from picture corpus
        faceIndex = 0
        picIndex = 1
        faceDB = {}
        resolution = 0.01
        lastStep = resolution
        bar = LoadBar(max=corpusSize)  # display a not too verbose % of the work done
        for pic in listOfPictures:
            bar.update(step=picIndex)
            picIndex = picIndex + 1

            # Load image
            image = io.imread(pic)

            # check if there is a face
            detected_faces = detect_faces(image)

            # Iterate over each face, Crop it and save it as jpg in a created Face Folder
            # create a .json with the link between a face and its origin picture
            for n, face_rect in enumerate(detected_faces):
                face = Image.fromarray(image).crop(face_rect)  # crop the picture
                path = os.path.join("Faces", str(faceIndex)+".jpg") # name its face from 0.jpg to N.jpg in Faces folder
                faceIndex = faceIndex + 1  # keep a counter to name the face.jpg file
                face.save(path)  # write the picture to disk
                face = utils.load_image(path) # not super efficient but the lib don't accept the picture already loaded
                encodings = utils.img_to_encodings(face)
                faceDB[str(faceIndex)+".jpg"] = [pic, str(encodings)]  # link faces, fingerprint and origin picture

        bar.end()
        with open('Faces.json', 'w') as fp:
            json.dump(faceDB, fp)

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