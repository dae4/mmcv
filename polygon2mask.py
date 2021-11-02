#%%
import numpy
from PIL import Image, ImageDraw
import json
# polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# width = ?
# height = ?
def polygon2mask(image_dir_path,json_path, save_dir_path):
    with open(json_path,'r') as json_file:
        json_read = json.load(json_file)
    # print(json_read.keys())
    print(len(json_read['images']))

    for id in json_read['images']:
        img = Image.open(image_dir_path+id['file_name'])
        (width, height) = img.size
        print(width,height)
        
        mask_img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        # mask = numpy.array(mask_img)


if __name__ == '__main__':
    data_root = '/home/daehan/project/mmcv/mmdetection/data/'
    image_dir_path = data_root+'images/val/'
    json_path = data_root+'val.json'
    save_dir_path = ''
 
    polygon2mask(image_dir_path,json_path, save_dir_path)
# %%
