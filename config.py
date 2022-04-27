import sys
import argparse

sys.argv = [""]
del sys

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-w", "--class_num", type=int, default=1)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=1)
parser.add_argument("-e", "--episode", type=int, default=50000)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-rf", "--TrainResultPath", type=str, default="result_1shot")
parser.add_argument("-rff", "--ResultSaveFreq", type=int, default=5000)
parser.add_argument("-msp", "--ModelSavePath", type=str, default="models_1shot")
parser.add_argument("-msf", "--ModelSaveFreq", type=int, default=5000)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-d", "--display_query_num", type=int, default=1)
parser.add_argument(
    "-modelf",
    "--encoder_model",
    type=str,
    default="",
)
parser.add_argument(
    "-modeld",
    "--decoder_model",
    type=str,
    default="",
)
parser.add_argument(
    "-modelh",
    "--head_model",
    type=str,
    default="",
)
parser.add_argument("-start", "--start_episode", type=int, default=0)
parser.add_argument("-fi", "--finetune", type=bool, default=True)
parser.add_argument("-dp", "--dataset_path", type=str, default="./spleen")


args = parser.parse_args()


## Hyper Parameters

CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
DISPLAY_QUERY = args.display_query_num
ENCODER_MODEL = args.encoder_model
DECODER_MODEL = args.decoder_model
HEAD_MODEL = args.head_model
DATASET_PATH = args.dataset_path