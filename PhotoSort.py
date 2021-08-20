"""
Take a folder as input
complete metadata using file creation if needed (metadata over file creation expect if date in futur or older than 1970)
Sort picture by date in folder date
sort picture by name in folder people



https://medium.com/analytics-vidhya/ai-saved-my-family-photos-521ce6fa5877

https://cloudinary.com/blog/automatic_image_categorization_and_tagging_with_imagga


"""

import time
from PIL import Image
from PIL.ExifTags import TAGS
from os.path import splitext
from tqdm import tqdm
import piexif
import shutil
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import imagehash
from collections import defaultdict
import hashlib
import os
import sys

def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(filename, first_chunk_only=False, hash=hashlib.sha1):
    hashobj = hash()
    file_object = open(filename, 'rb')

    if first_chunk_only:
        hashobj.update(file_object.read(1024))
    else:
        for chunk in chunk_reader(file_object):
            hashobj.update(chunk)
    hashed = hashobj.digest()

    file_object.close()
    return hashed


def check_for_duplicates(paths, hash=hashlib.sha1):
    hashes_by_size = defaultdict(list)  # dict of size_in_bytes: [full_path_to_file1, full_path_to_file2, ]
    hashes_on_1k = defaultdict(list)  # dict of (hash1k, size_in_bytes): [full_path_to_file1, full_path_to_file2, ]
    hashes_full = {}   # dict of full_file_hash: full_path_to_file_string

    print("checking duplicates")
    for path in tqdm(paths):
        for dirpath, dirnames, filenames in tqdm(os.walk(path)):
            # get all files that have the same size - they are the collision candidates
            for filename in filenames:

                full_path = os.path.join(dirpath, filename)
                try:
                    # if the target is a symlink (soft one), this will
                    # dereference it - change the value to the actual target file
                    full_path = os.path.realpath(full_path)
                    file_size = os.path.getsize(full_path)
                    hashes_by_size[file_size].append(full_path)
                except (OSError,):
                    # not accessible (permissions, etc) - pass on
                    continue
    print("still checking duplicates")
    # For all files with the same file size, get their hash on the 1st 1024 bytes only
    for size_in_bytes, files in tqdm(hashes_by_size.items()):
        if len(files) < 2:
            continue    # this file size is unique, no need to spend CPU cycles on it

        for filename in files:
            try:
                small_hash = get_hash(filename, first_chunk_only=True)
                # the key is the hash on the first 1024 bytes plus the size - to
                # avoid collisions on equal hashes in the first part of the file
                # credits to @Futal for the optimization
                hashes_on_1k[(small_hash, size_in_bytes)].append(filename)
            except (OSError,):
                # the file access might've changed till the exec point got here
                continue
    print("Listing duplicates")
    # For all files with the hash on the 1st 1024 bytes, get their hash on the full file - collisions will be duplicates
    for __, files_list in tqdm(hashes_on_1k.items()):
        if len(files_list) < 2:
            continue    # this hash of fist 1k file bytes is unique, no need to spend cpy cycles on it

        for filename in files_list:
            try:
                full_hash = get_hash(filename, first_chunk_only=False)
                duplicate = hashes_full.get(full_hash)
                if duplicate:
                    print("Duplicate found: {} and {}".format(filename, duplicate))
                else:
                    hashes_full[full_hash] = filename
            except (OSError,):
                # the file access might've changed till the exec point got here
                continue

def convert_picture_extension(dumppath, img_path):
    """
    Convert picture to jpg and try to fix date metadata by extracting it from file creation time or folder with date
    :param dumppath:   path to dump the converted picture
    :param img_path: path of the picture to convert
    :return: a dictionnary with the status
    """
    rename = False
    status = {"datesource":"exif", "sucess":"true", "file":img_path}
    target = '.jpg'
    filename, extension = splitext(os.path.basename(img_path))
    metadate = ImageDate(img_path)
    if metadate is None:
        status["datesource"] = "filemetadata"
        # metadata > foldername > file creation
        words = img_path.split(os.pathsep)
        t = time.localtime(os.path.getctime(img_path))
        ctime = time.strftime('%Y:%m:%d %H:%M:%S', t)
        metadate = ctime
        timeSeparator = ["-", ".", " ", ":", ",", ";", "_"]
        success = False
        for separatorA in timeSeparator:
            try:
                pathtime = datetime.datetime.strptime(str(words[-2]+" 12:00:00"), "%Y"+separatorA+"%m"+separatorA+"%d %H:%M:%S").date()
                success = True
                break
            except:
                pass
        #pathtime = datetime.datetime.strptime(str(words[-2]+" 12:00:00"), "%Y-%m-%d %H:%M:%S")
        if success:
            if pathtime < datetime.datetime.strptime("1990-12-12 12:00:00", "%Y-%m-%d %H:%M:%S"):
                if pathtime > datetime.datetime.now():
                    # must be a real date so not in futur nor too old.
                    metadate = time.strftime('%Y:%m:%d %H:%M:%S', pathtime)
                    status["datesource"] = "folder"
    if rename:
        #adding useful keyword in picture name from path (folder name, categories)
        speratorList = ["-", "_", ".", ";", ",", ":", "/", "/", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        for separator in speratorList:
            img_path = img_path.replace(separator, " ")
        string = img_path.split(" ")
        keywords = []
        accept = ["nathann", "ulysse", "lulu", "n&a" "n&u", "n+a" "n+u", "booboo", "john", "jc", "jn", "jc+b", "jc+m+boubou", "jcn"]
        for word in string:
            key = word.lower()
            if key in accept:
                keywords.append(keywords)
        newpath = os.path.join(dumppath, filename)
        targetname = newpath + target
    else:
        words = img_path.split(os.pathsep)
        newpath = os.path.join(dumppath, str(words[-2]+"-"+filename))
        targetname = newpath + target

    if (extension not in [target]):
        try:
            # convert picture to jpg
            im = Image.open(img_path)
            im.save(targetname)
        except Exception as e:
            status["sucess"] = "false"
            print("might be that the pic was edited and has some transparent")
            print(e)
            try:
                os.remove(targetname)
            except OSError:
                pass
            return status
    else:
        try:
            im = Image.open(img_path) # keep this line so metadata can still be used
            # copy the jpg as it aldready exist
            shutil.copy(img_path, targetname)
        except Exception as e:
            status["sucess"] = "false"
            print(e)
            try:
                os.remove(targetname)
            except OSError:
                pass
            return status
    try:
        # update the metadata
        if im.info is not None:
            exif_dict = piexif.load(targetname)
        exif_dict['0th'][piexif.ImageIFD.DateTime] = metadate
        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = metadate
        exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized] = metadate
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, targetname)
    except Exception as e:
        print(e)
        status["sucess"] = "partial"
    try:
        # edit creation time
        date = datetime.datetime.strptime(metadate, "%Y:%m:%d %H:%M:%S")
        modTime = time.mktime(date.timetuple())
        os.utime(targetname, (modTime, modTime))
    except Exception as e:
        print(e)
        print("unable to set file creation time to picture shot time")
        status["sucess"] = "partial"

    status["sucess"] = "true"
    return status

def get_exif(fn):
    #https://stackoverflow.com/questions/765396/exif-manipulation-library-for-python/765403#765403
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    if info == None: return  # found no tags
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret

def GetImageDateDistribution(listOfPicturesPath):
    timeSeparator = ["-", ".", " ", ":", ",", ";", "_"]
    # convert everything to png and fill metadata + creation time
    result = []
    for pic in tqdm(listOfPicturesPath):
        datetimestring = ImageDate(pic)
        if datetimestring is not None:
            for separatorA in timeSeparator:
                for separatorB in timeSeparator:
                    try:
                        result.append(datetime.datetime.strptime(datetimestring, "%Y"+separatorA+"%m"+separatorA+"%d %H"+separatorB+"%M"+separatorB+"%S").date())
                    except:
                        pass
        else:
            pass
    plt.rcParams["figure.figsize"] = [12.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    df = pd.DataFrame(np.random.choice(result, 500), columns=['dates'])
    df['ordinal'] = [x.toordinal() for x in df.dates]

    ax = df['ordinal'].plot(kind='kde')

    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks[::2])

    xlabels = [datetime.datetime.fromordinal(int(x)).strftime('%Y-%m-%d') for x in x_ticks[::2]]
    ax.set_xticklabels(xlabels)

    plt.show()



def ImageDate(fn):
    "Returns the date and time from image(if available) "
    # https://orthallelous.wordpress.com/2015/04/19/extracting-date-and-time-from-images-with-python/
    TTags = [('DateTimeOriginal', 'SubsecTimeOriginal'),  # when img taken
             ('DateTimeDigitized', 'SubsecTimeDigitized'),  # when img stored digitally
             ('DateTime', 'SubsecTime')]  # when img file was changed
    # for subsecond prec, see doi.org/10.3189/2013JoG12J126 , sect. 2.2, 2.3
    try:
        exif = get_exif(fn)
    except Exception as e:
        print("could not extract exif data for : "+str(fn))
        print(e)
        return
    if exif == None: return  # found no tags

    for i in TTags:
        dat, sub = exif.get(i[0]), exif.get(i[1], 0)
        dat = dat[0] if type(dat) == tuple else dat  # PILLOW 3.0 returns tuples now
        sub = sub[0] if type(sub) == tuple else sub
        if dat != None: break  # got valid time
    if dat == None: return  # found no time tags
    return dat

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

def pathKeywordAnalysis(img_paths):
    keywords = []
    speratorList = ["-", "_", ".", ";", ",", ":", "/", "/", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    for img_path in img_paths:
        for separator in speratorList:
            img_path = img_path.replace(separator, " ")
        string = img_path.split(" ")
        for sentence in string:
            keywords.append(sentence.lower())

    keywords = set(keywords)
    print("priting keywors extracted from path")
    for k in sorted(keywords):
        print(k)

def main(path):
    """
    ListOfFile = getListOfFiles(path)
    listOfPicturesPath = excludeNonPictureFromFileList(ListOfFile)
    corpusSize = len(listOfPicturesPath)
    print(corpusSize)

    #list keyword found in path
    #pathKeywordAnalysis(listOfPicturesPath)

    # get historgram with picture distribution
    GetImageDateDistribution(listOfPicturesPath)

    # convert everything to png and fill metadata + creation time
    result = []
    for pic in tqdm(listOfPicturesPath):
        result.append(convert_picture_extension("/home/nathann/SortedPictures/pngdump", pic))

    #display stat of what picture failed etc
    datesourceEXIF = 0
    datesourcePATH = 0
    datesourceFILE = 0
    sucess = []
    failure = []
    partial = []

    for r in result:
        if r["datesource"] == "exif":
            datesourceEXIF = datesourceEXIF + 1
        if r["datesource"] == "filemetadata":
            datesourceFILE = datesourceFILE + 1
        if r["datesource"] == "folder":
            datesourcePATH = datesourcePATH + 1
        if r["sucess"] == "true":
            sucess.append(r["file"])
        if r["sucess"] == "partial":
            partial.append(r["file"])
        if r["sucess"] == "false":
            failure.append(r["file"])
    print("partial convert : ")
    print(partial)
    print("failure convert : ")
    print(failure)
    print("Stats")
    print("----------------------------------")
    print("Sucess: "+str(len(sucess)))
    print("partial: " + str(len(partial)))
    print("failure: " + str(len(failure)))
    print("----------------------------------")
    print("EXIF source : "+str(datesourceEXIF))
    print("file creation date source : " + str(datesourceFILE))
    print("folder name source : " + str(datesourcePATH))
    """
    # delete duplicated picture in dump
    """
    ListOfFile = getListOfFiles("/home/nathann/SortedPictures/pngdump")
    listOfPicturesPath = excludeNonPictureFromFileList(ListOfFile)
    corpusSize = len(listOfPicturesPath)
    print(corpusSize)

    hashdict = {}
    hashlist = []
    pathlist = []
    for pic in tqdm(listOfPicturesPath):
        try:
            i = Image.open(pic)
            hashpic = imagehash.whash(i)
            hashlist.append(int(hashpic))
            pathlist.append(pic)
            hashdict["pic"] = int(hashpic)
        except:
            pass

    hashlist.sort()
    print(hashlist)
    #print(max(filter(lambda gr: gr[0] == 0,groupby(hashlist)), key=lambda gr: len(list(gr[1]))))
    #print(len(max(filter(lambda gr: gr[0] == 0, groupby(hashlist)), key=lambda gr: len(list(gr[1])))))
    """
    # delete duplicate
    check_for_duplicates(["/home/nathann/SortedPictures/pngdump/"])

    # detect face in picture and tag them

    # create folder with all picture having a face

    # sort the rest of the picture
if __name__ == "__main__":
    main("/home/nathann/Images")
