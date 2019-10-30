import os
import glob

root = '/media/yj_data/DataSet/reid/kesci'
with open(root + '/gallery_a_list.txt', 'w') as f:
    for file in glob.glob(os.path.join(root, 'gallery_a', '*.png')):
        img_path = file.split('/')[-1]
        folder = file.split('/')[-2]
        f.write('%s/%s %s\n'%(folder, img_path, 0))