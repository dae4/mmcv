import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

dataDir='/home/daehan/project/mmcv/mmdetection/data/'
dataType='val'
coco =COCO(dataDir+'{}.json'.format(dataType))
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

if not os.path.exists(dataDir+'mask'):
    os.mkdir(dataDir+'mask')
    os.mkdir(dataDir+'mask/{}'.format(dataType))

filterClasses = ['food']
# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))
for i in imgIds:
# load and display a random image
    img = coco.loadImgs(imgIds[i-1])[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    plt.imsave(dataDir+'masks/'+'{}/'.format(dataType)+img['file_name'],mask)
