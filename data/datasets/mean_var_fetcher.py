from PIL import Image
import numpy as np
import os
import glob



r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0
root = "/media/yj_data/DataSet/reid/kesci/train"
files = glob.glob(os.path.join(root, '*.png'))
count = len(files)

for i, image_file in enumerate(files):
    print('Process: %d/%d' % (i, count))
    img = Image.open(image_file)
    # img = img.resize((299, 299))
    img = np.asarray(img)
    img = img.astype('float32') / 255.
    total += img.shape[0] * img.shape[1]

    r += img[:, :, 0].sum()
    g += img[:, :, 1].sum()
    b += img[:, :, 2].sum()

    r_2 += (img[:, :, 0] ** 2).sum()
    g_2 += (img[:, :, 1] ** 2).sum()
    b_2 += (img[:, :, 2] ** 2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_std = np.sqrt(r_2 / total - r_mean ** 2)
g_std = np.sqrt(g_2 / total - g_mean ** 2)
b_std = np.sqrt(b_2 / total - b_mean ** 2)

print('Mean is %s' % ([r_mean, g_mean, b_mean]))
print('Std is %s' % ([r_std, g_std, b_std]))

# kesci
# Mean is [0.09661545132935477, 0.18356956955890383, 0.21322472792945668]
# Std is [0.17483382734478947, 0.16510604114266944, 0.22086535365381385]


# market
# PIXEL_MEAN: [0.485, 0.456, 0.406]
# PIXEL_STD: [0.229, 0.224, 0.225]
# Mean is [0.41448362789363885, 0.38874238652644255, 0.383957302868228]
# Std is [0.21722327898281163, 0.20945075025456739, 0.2087847228164872]