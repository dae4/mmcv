#%%
from PIL import Image, ImageDraw
import json
def polygon2mask(image_dir_path,json_path, save_dir_path):
    with open(json_path,'r') as json_file:
        json_read = json.load(json_file)
    # print(json_read.keys())
    print(len(json_read['images']))

    for id in json_read['images']:
        img = Image.open(image_dir_path+id['file_name'])
        (width, height) = img.size
        print(width,height)
        for itr in json_read['annotation']:
            if itr['image_id']==id['id']:
                polygon=itr['segmentation']

if __name__ == '__main__':
    data_root = '/home/daehan/project/mmcv/mmdetection/data/'
    image_dir_path = data_root+'images/val/'
    json_path = data_root+'val.json'
    save_dir_path = ''
 
    polygon2mask(image_dir_path,json_path, save_dir_path)
# %%
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

dataDir='/home/daehan/project/mmcv/mmdetection/data/'
dataType='val'

coco =COCO('/home/daehan/project/mmcv/mmdetection/data/yolo_val.json')
# %%
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)
# %%
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

filterClasses = ['food']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)
print(catIds) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))

# load and display a random image
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))/255.0

plt.axis('off')
plt.imshow(I)
plt.show()
# %%
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
# %%
classes = ['food']
images = []
if classes!=None:
    # iterate for each individual class in the list
    for className in classes:
        # get all images containing given class
        catIds = coco.getCatIds(catNms=className)
        imgIds = coco.getImgIds(catIds=catIds)
        images += coco.loadImgs(imgIds)
else:
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    
# Now, filter out the repeated images    
unique_images = []
for i in range(len(images)):
    if images[i] not in unique_images:
        unique_images.append(images[i])

dataset_size = len(unique_images)

print("Number of images containing the filter classes:", dataset_size)