from config import *
from vnet import *
from loader import *

import torch
from torch.autograd import Variable
import os


def main():
    encoder = VNetEncoder(in_channels=2).cuda(GPU)
    decoder = VNetDecoder(classes=1).cuda(GPU)

    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    if args.finetune:

        if os.path.exists(ENCODER_MODEL):

            checkpoint1 = torch.load(ENCODER_MODEL)
            encoder.load_state_dict(checkpoint1["encoder_state_dict"])

            print("load encoder success")

        else:
            print("Can not load encoder: %s" % ENCODER_MODEL)
            print("starting from scratch")

        if os.path.exists(DECODER_MODEL):

            checkpoint2 = torch.load(DECODER_MODEL)
            decoder.load_state_dict(checkpoint2["decoder_state_dict"])

            print("load decoder success")

        else:
            print("Can not load decoder: %s" % DECODER_MODEL)
            print("starting from scratch")

    print("Training...")

    for episode in range(args.start_episode, EPISODE):
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        (supp_img, supp_labels, qry_img, qry_labels) = get_oneshot_batch()

        sample_features, _ = encoder(Variable(supp_img).cuda(GPU))  ## 5*256*4*16*16
        sample_features = sample_features.view(
            CLASS_NUM, SAMPLE_NUM_PER_CLASS, 256, 4, 16, 16
        )  ## 1*5*256*4*16*16
        #print(sample_features.shape)
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*256*4*16*16
        #print(sample_features.shape)
        batch_features, ft_list = encoder(Variable(qry_img).cuda())  ## 5*256*4*16*16
        #print(batch_features.shape)

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(
            BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1, 1
        )  ## 5*256*4*16*16
        #print(sample_features_ext.shape)
        batch_features_ext = batch_features.unsqueeze(0).repeat(
            CLASS_NUM, 1, 1, 1, 1, 1
        )
        #print(batch_features_ext.shape)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  ## 5*1*256*4*16*16
        #print(batch_features_ext.shape)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(
            -1, 512, 4, 16, 16
        )  ## 5*512*4*16*16
        out = decoder(relation_pairs, ft_list)

        bce = nn.BCEWithLogitsLoss().cuda(GPU)
        loss = bce(out, Variable(qry_labels).cuda(GPU))
        loss.backward()

        encoder_optim.step()
        decoder_optim.step()

        if (episode + 1) % 10 == 0:
            print("\nepisode:", episode + 1, "loss", loss.item())

        if not os.path.exists(args.TrainResultPath):
            os.makedirs(args.TrainResultPath)
        if not os.path.exists(args.ModelSavePath):
            os.makedirs(args.ModelSavePath)

        if (episode + 1) % args.ModelSaveFreq == 0:

            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                },
                str(
                    "./%s/vnet_encoder_" % args.ModelSavePath
                    + str(episode+1)
                    + "_"
                    + str(CLASS_NUM)
                    + "_way_"
                    + str(SAMPLE_NUM_PER_CLASS)
                    + "shot.pkl"
                ),
            )

            torch.save(
                {
                    "decoder_state_dict": decoder.state_dict(),
                },
                str(
                    "./%s/decoder_network_" % args.ModelSavePath
                    + str(episode+1)
                    + "_"
                    + str(CLASS_NUM)
                    + "_way_"
                    + str(SAMPLE_NUM_PER_CLASS)
                    + "shot.pkl"
                ),
            )
            print("Save model for episode:", episode+1)


if __name__ == "__main__":
    main()
