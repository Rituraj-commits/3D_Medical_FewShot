from utils import *

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

LIR = -125
HIR = 275
BD_BIAS = 32 # cut irrelavent empty boundary to make roi stands out
SPA_FAC = (512 - 2 * BD_BIAS) / 256 # spacing factor

Path = '/media/ri2raj/External HDD/SABS/RawData/Training/'

for _,file in enumerate(tqdm(os.listdir(Path))):
    if file=='img':
        for _,subfile in enumerate(os.listdir(Path + file)):
            #print(subfile)
            img = sitk.ReadImage(os.path.join(Path + file, subfile))
            filename,extension = os.path.splitext(subfile)
            img_array = sitk.GetArrayFromImage(img)
            img_array[img_array > HIR] = HIR
            img_array[img_array < LIR] = LIR

            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0

            array = img_array[:, BD_BIAS: -BD_BIAS, BD_BIAS: -BD_BIAS]
            cropped_img_o = sitk.GetImageFromArray(array)
            cropped_img_o = copy_spacing_ori(img, cropped_img_o)

            # resampling
            img_spa_ori = img.GetSpacing()
            res_img_o = resample_by_res(cropped_img_o, [img_spa_ori[0] * SPA_FAC, img_spa_ori[1] * SPA_FAC, img_spa_ori[-1]], interpolator = sitk.sitkLinear,
                                            logging = True)

            img_array = sitk.GetArrayFromImage(res_img_o)
            img_array = resize_image(img_array, (64,256,256), mode='symmetric')
            sitk.WriteImage(sitk.GetImageFromArray(img_array), os.path.join(Path + file, subfile))

    elif file=='label':
        for _,subfile in enumerate(os.listdir(Path + file)):
            #print(subfile)
            filename,extension = os.path.splitext(subfile)
            label = sitk.ReadImage(os.path.join(Path + file, subfile))
            label_array = sitk.GetArrayFromImage(label)


            # cropping
            lb_arr = label_array[:,BD_BIAS: -BD_BIAS, BD_BIAS: -BD_BIAS]
            cropped_lb_o = sitk.GetImageFromArray(lb_arr)
            cropped_lb_o = copy_spacing_ori(label, cropped_lb_o)

            lb_spa_ori = label.GetSpacing()

            # resampling
            res_lb_o = resample_lb_by_res(cropped_lb_o,  [lb_spa_ori[0] * SPA_FAC, lb_spa_ori[1] * SPA_FAC, lb_spa_ori[-1] ], interpolator = sitk.sitkLinear,
                                             logging = True)

            label_array = sitk.GetArrayFromImage(res_lb_o)
            label_array = resize_image(label_array, (64,256,256), mode='symmetric')
            sitk.WriteImage(sitk.GetImageFromArray(label_array), os.path.join(Path + file, subfile))

    else:
        print('file not found')