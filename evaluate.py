from model import *

from torch.autograd import Variable
import torch
import os
import numpy as np
import SimpleITK as sitk
import random

import argparse
import sys

sys.argv = [""]
del sys
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ISIC dataset.")
    parser.add_argument(
        "--output-root", help="path to results directory ", default="results/"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument(
        "-modele",
        "--encoder_model",
        type=str,
        default="/content/drive/MyDrive/vnet_encoder_5000_1_way_1shot.pkl",
    )
    parser.add_argument(
        "-modeld",
        "--decoder_model",
        type=str,
        default="/content/drive/MyDrive/decoder_network_5000_1_way_1shot.pkl",
    )

    parser.add_argument("--use-cuda", type=bool, default=True)
    parser.add_argument(
        "--k", default=1, type=int, help="Number of training shots per task"
    )
    args = parser.parse_args()
    return args


def validate_args(args):
    assert os.path.exists(args.input_data)
    assert os.path.exists(args.encoder_model)
    assert os.path.exists(args.decoder_model)


def build_model(args):
    print("Building neural networks")

    encoder = VNetEncoder(in_channels=2).cuda(args.gpu)
    decoder = VNetDecoder(classes=1).cuda(args.gpu)

    if os.path.exists(args.encoder_model):
        if args.use_cuda:
            checkpoint1 = torch.load(args.encoder_model)
            encoder.load_state_dict(checkpoint1["encoder_state_dict"])
            print("loaded CUDA-enabled encoder")

    else:
        raise RuntimeError("Can not load feature encoder: %s" % args.encoder_model)
    if os.path.exists(args.decoder_model):
        if args.use_cuda:
            checkpoint2 = torch.load(args.decoder_model)
            decoder.load_state_dict(checkpoint2["decoder_state_dict"])
            print("loaded GPU decoder")

    else:
        raise RuntimeError("Can not load relation network: %s" % args.decoder_model)

    print("Model successfully built.")
    return encoder, decoder


def dice_score(predictions, labels):
    pred = predictions.data.cpu().numpy()
    dices = []
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    target = labels.numpy().astype(bool)
    numerator = 2 * np.sum(pred * target)
    denominator = np.sum(pred + target)
    dice = (numerator + 1) / (denominator + 1)
    if dice < 0.1:
        pass
    else:

        print("dice=%0.4f" % dice)
        dices.append(dice)
    return dices


def get_one_test_shot_batch(img_dir: str, lb_dir: str, k: int = 5, class_num: int = 1):
    test_shots: int = 1

    support_images = np.zeros((class_num * k, 1, 64, 256, 256), dtype=np.float32)
    support_labels = np.zeros(
        (class_num * k, class_num, 64, 256, 256), dtype=np.float32
    )
    query_images = np.zeros((class_num * test_shots, 1, 64, 256, 256), dtype=np.float32)
    query_labels = np.zeros(
        (class_num * test_shots, class_num, 64, 256, 256), dtype=np.float32
    )
    zeros = np.zeros((class_num * test_shots, 1, 64, 256, 256), dtype=np.float32)

    # Load tuples of images and masks in data_dir
    labels = [x for x in os.listdir(lb_dir)]
    images = [x for x in os.listdir(img_dir)]
    labels = [os.path.join(lb_dir, x) for x in labels]
    images = [os.path.join(img_dir, x) for x in images]
    images_labels = list(zip(images, labels))

    # Randomly sample k support tuples
    random.shuffle(images_labels)
    images_labels = [
        (
            sitk.GetArrayFromImage(sitk.ReadImage(i)),
            sitk.GetArrayFromImage(sitk.ReadImage(l)),
        )
        for i, l in images_labels
    ]
    images_labels = [(i, l) for i, l in images_labels]
    # print(len(images_labels))
    assert k <= len(images_labels) - test_shots

    support_image_labels = images_labels[:k]
    test_image_labels = images_labels[k : k + test_shots]

    for i, (image, label) in enumerate(support_image_labels):
        support_images[i] = image
        support_labels[i][0] = label
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat(
        (support_images_tensor, support_labels_tensor), dim=1
    )

    for i, (image, label) in enumerate(test_image_labels):
        query_images[i] = image
        query_labels[i][0] = label
    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return (
        support_images_tensor,
        support_labels_tensor,
        query_images_tensor,
        query_labels_tensor,
    )


def forward(encoder, decoder, support_images_tensor, query_images_tensor, args):
    class_num = 1
    if args.use_cuda:
        var = Variable(support_images_tensor).cuda(args.gpu)
    else:
        var = Variable(support_images_tensor)
    sample_features, _ = encoder(var)
    sample_features = sample_features.view(class_num, args.k, 256, 4, 16, 16)
    sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
    # print(sample_features.shape)
    if args.use_cuda:
        batch_features, ft_list = encoder(Variable(query_images_tensor).cuda(args.gpu))
    else:
        batch_features, ft_list = encoder(Variable(query_images_tensor))

    sample_features_ext = sample_features.unsqueeze(0).repeat(class_num, 1, 1, 1, 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(
        -1, 512, 4, 16, 16
    )
    # print(sample_features_ext.shape)
    # print(batch_features_ext.shape)
    out = decoder(relation_pairs, ft_list)
    output = torch.sigmoid(out)
    return output


def main():
    args = parse_args()
    validate_args(args)
    n_eval_samples_per_task = 10  # Following Hendryx et al. 2020

    test_tasks = ["FewShot/test/Stomach/"]

    encoder, decoder = build_model(args)

    # Loop through each task, evaluating each task n_eval_samples_per_task times

    dices = []

    for _ in range(n_eval_samples_per_task):
        for task in test_tasks:
            # Load examples for the task:
            (
                support_images_tensor,
                support_labels_tensor,
                query_images_tensor,
                query_labels_tensor,
            ) = get_one_test_shot_batch(task, k=args.k)

            predictions = forward(
                encoder,
                decoder,
                support_images_tensor,
                query_images_tensor,
                args,
            )

            dices.extend(dice_score(predictions, query_labels_tensor))

    print(f"Mean Dice Score: {np.nanmean(dices)*100:.2f}")
