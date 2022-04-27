from config import *

import os
import random
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt


classes_name = os.listdir("FewShot/train/")
classes = list(range(0, len(classes_name)))

chosen_classes = random.sample(classes, CLASS_NUM)


def get_oneshot_batch():

    classes_name = os.listdir("FewShot/train/")
    classes = list(range(0, len(classes_name)))
    chosen_classes = random.sample(classes, CLASS_NUM)

    support_images = np.zeros(
        (CLASS_NUM * SAMPLE_NUM_PER_CLASS, 1, 64, 256, 256), dtype=np.float32
    )
    support_labels = np.zeros(
        (CLASS_NUM * SAMPLE_NUM_PER_CLASS, CLASS_NUM, 64, 256, 256), dtype=np.float32
    )
    query_images = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 64, 256, 256), dtype=np.float32
    )
    query_labels = np.zeros(
        (CLASS_NUM * BATCH_NUM_PER_CLASS, CLASS_NUM, 64, 256, 256), dtype=np.float32
    )
    zeros = np.zeros((CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 64, 256, 256), dtype=np.float32)
    class_cnt = 0

    for i in chosen_classes:
        #print ('class %s is chosen' % i)
        imgnames = os.listdir('FewShot/train/%s/images' % classes_name[i])
        labelnames = os.listdir('FewShot/train/%s/masks' % classes_name[i])
        indexs = list(range(0,len(imgnames)))
        chosen_index = random.sample(indexs, SAMPLE_NUM_PER_CLASS + BATCH_NUM_PER_CLASS)
        j = 0
        for k in chosen_index:
            # process image

            image = sitk.ReadImage('FewShot/train/%s/images/%s' % (classes_name[i], imgnames[k]))
            image = sitk.GetArrayFromImage(image)
    
            # labels
            label = sitk.ReadImage('FewShot/train/%s/masks/%s' % (classes_name[i], labelnames[k]))
            label = sitk.GetArrayFromImage(label)
            if j < SAMPLE_NUM_PER_CLASS:
                support_images[j] = image
                support_labels[j][0] = label
            else:
                query_images[j-SAMPLE_NUM_PER_CLASS] = image
                query_labels[j-SAMPLE_NUM_PER_CLASS][class_cnt] = label
            j += 1

        class_cnt += 1

    zeros_tensor = torch.from_numpy(zeros)
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat((support_images_tensor, support_labels_tensor), dim=1)

    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return (
        support_images_tensor,
        support_labels_tensor,
        query_images_tensor,
        query_labels_tensor
    )


