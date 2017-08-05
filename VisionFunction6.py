##########
19/07/2017
##########

#########################################################
##Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

##Global Variables
#QueryImage = 'C:/PyProjects/VisionSystem/VisionPass/templates_resized/sample13.facebook.jpg'
QueryImage = 'C:/PyProjects/VisionSystem/VisionPass/templates_resized/sample14.netflix.jpg'

#files = glob.glob ("C:/PyProjects/VisionSystem/VisionPass/templates_resized/*.jpg")
files = glob.glob ("C:/PyProjects/VisionSystem/VisionPass/bad_templates/*.jpg")

CV_LOAD_IMAGE_COLOR = 1
X_data = []
desIndex = []
desQuery = []
kpIndex = []
kpQuery = []
lenIndex = []
desNum = []
idsIndex = []
matches = []
desHolder = []
desIxHold = []
numHold = []
idMatch = []
matchesArray = []
orb = cv2.ORB_create()



matchesThreshold = 0.5



desTemplate = np.zeros((10,500,32))




#########################################################
##Functions
 
def showImage(timg):
    plt.imshow(timg)
    plt.show()

def processQuery (Qimg):
    imgSourceBGR = cv2.imread(Qimg, CV_LOAD_IMAGE_COLOR)
    Qimg = cv2.cvtColor(imgSourceBGR, cv2.COLOR_BGR2GRAY)
    return (Qimg)


def computeQuery (Qimg):
    kp = orb.detect(Qimg, None)
    kp, des = orb.compute(Qimg, kp)
    QdesNum = des.shape
    arraynew = [kp, des, QdesNum]
    return (arraynew)

def processIndex (images):
    i = 0
    for myFile in images:
        print 'Processed image {} of {}'.format(i+1,len(images))
        print(myFile)
        i = i + 1
        idsIndex.append(myFile)
        image = cv2.imread (myFile)
        imggry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_data.append (imggry)
    array = [idsIndex, X_data]
    print('')
    return (array)

def computeIndex (X_data):
        for i,img in enumerate(X_data):
            kp, des = orb.detectAndCompute(img,None)
            desIndex.append(des)
            kpIndex.append(kp)
            lenIndex.append(des[:,0].shape)
            desNum.append(des.shape)
            descriptors = np.zeros((500,32)) #Matrix to hold the descriptors
            descriptors = np.concatenate((descriptors,des),axis=0)
            print 'Computed image {} of {}'.format(i+1,len(X_data))
            print "No. of Descriptors %f" % des[:,0].shape
            values = descriptors[1:,:]
            array = [desIndex, kpIndex, lenIndex, desNum]
        print('')
        return (array)
    
def BruteForceMatcher (desQuery,desIndex,desNum):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    imgnum = 0
    for i in range (0, len(desIndex)):
        matches = bf.match(desQuery,desIndex[i])
        matchesArray.append (sorted(matches, key = lambda x:x.distance))
    return (matchesArray)

def Comparator (matches, QdesNum):
    matchCheck = 0
    imgnum = 0
    for i in range (0, len(desIndex)):
        numtmp, trash = QdesNum
        tmpthresh = numtmp*0.4
        print 'Compared image {} of {}'.format(i+1,len(desIndex))
        print 'Number of matches = {}, Threshold = {}'.format(len(matches[i]),tmpthresh)
        if len(matches[i])>= (tmpthresh):
            matchCheck = 1
            imgnum = i
            print ("          Match, Image = %s" % idsIndex[i])   
        else:
            print ('No Match, Image = Unknown')
    return(matches[imgnum], matchCheck, imgnum)

def DrawMatches (gryimage, kpHolder, dataHolder, kpIxHold, matches, imgnum, matchCheck):
    if matchCheck == 1:
        img3 = cv2.drawMatches(gryimage,kpHolder,dataHolder[imgnum],kpIxHold[imgnum],matches[:10],None, flags=2)
        plt.imshow(img3)
        plt.show()
    return()

def main():
    orb = cv2.ORB_create()
    gryimage = processQuery(QueryImage)
    #showImage(gryimage)
    
    # Compute query () = array holder stores query image kp and des
    arrayHolder = computeQuery(gryimage)
    kpHolder = arrayHolder[0]
    desHolder = arrayHolder[1]
    QdesNum = arrayHolder[2]
    # End Compute query ()  method

    # Process Index () = converts query image colours
    PiArrayHolder = processIndex(files)
    idsHolder = PiArrayHolder[0]
    dataHolder = PiArrayHolder[1]
    # End Compute query ()  method

    # Compute Index () = array holder stores index images kp's and des's
    CiArrayHolder = computeIndex(X_data)
    kpIxHold = CiArrayHolder[1]
    desIxHold = CiArrayHolder[0]
    # End Compute Index ()  method

    lenHold = CiArrayHolder[2]
    numHold = CiArrayHolder[3]
    BfArrayHolder = BruteForceMatcher(desHolder, desIxHold, numHold)
    CompArrayHolder = Comparator(BfArrayHolder,QdesNum)
    matches = CompArrayHolder[0]
    matchCheck = CompArrayHolder[1]
    imgnum = CompArrayHolder[2]
    DrawMatches(gryimage, kpHolder, dataHolder, kpIxHold, matches, imgnum, matchCheck) 

if __name__ == '__main__':
    main()
