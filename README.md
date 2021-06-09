# PhotoFaceTagger
more than a decade of unsorted  family photos. Here is a simple tool I built to scan all your picture, detect the face and cluster them to find the same persons accros multiple picture and complete the picture tag so that you can easily find who you're looking for.

## capability and caveat
So far the script parse a given directory and its sub directory and extract all face. it save them as png in a folder called "Faces"
run the clustering algorithm and store the clustered picture in the folder called FaceCluster.

The edditing of the exif data must be implemented. (and the random segfault that arrise when the number of picture to tag get over a few thousand must be fixed)

currently there is no CLI parsing. that will also come.

## Installing requirements
```
pip3 install -r requirements.txt
```

## Running
currently to have to edit the hardcoded path in the script. I will add a correct argument parsing mechanism once the program does what its supposed to.
```
python3 PhotoFaceTagger.py
```
