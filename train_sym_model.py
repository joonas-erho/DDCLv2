from models import *
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description="Training Configuration")

    parser.add_argument('--shuffle', action='store_true', default=True, help='Whether to shuffle the dataset')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', help='Disable shuffling')

    parser.add_argument('--seed', type=int, default=420, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--steps_per_epoch', type=int, default=400, help='Steps per epoch')
    parser.add_argument('--nepochs', type=int, default=400, help='Number of epochs')

    parser.add_argument('--bidirectional_audio', action='store_true', default=True, help='Use bidirectional audio')
    parser.add_argument('--no-bidirectional_audio', dest='bidirectional_audio', action='store_false', help='Disable bidirectional audio')

    parser.add_argument('--audio_to_history', action='store_true', default=False, help='Use audio-to-history mechanism')
    parser.add_argument('--no-audio_to_history', dest='audio_to_history', action='store_false', help='Disable audio-to-history')

    parser.add_argument('--use_diff', action='store_true', default=False, help='Use difference features')
    parser.add_argument('--no-use_diff', dest='use_diff', action='store_false', help='Disable difference features')

    parser.add_argument('--conv3d', action='store_true', default=True, help='Use 3D convolution')
    parser.add_argument('--no-conv3d', dest='conv3d', action='store_false', help='Disable 3D convolution')

    parser.add_argument('--aud_memlen', type=int, default=7, help='Audio memory length')
    parser.add_argument('--memlen', type=int, default=64, help='General memory length')
    parser.add_argument('--mem_size', type=int, default=5000, help='Memory size')
    parser.add_argument('--audio_radius', type=int, default=4, help='Audio radius')
    parser.add_argument('--narrow_types', type=int, default=4, help='Number of narrow types')

    parser.add_argument('--train_txt_fp', type=str, default='sym/songs/songs_train.txt', help='Path to training text file')
    parser.add_argument('--test_txt_fp', type=str, default='sym/songs/songs_test.txt', help='Path to testing text file')

    parser.add_argument('--model_dir', type=str, default='trained_models', help='Directory to save/load models')
    parser.add_argument('--model_name', type=str, default='sym_conv3d', help='Model name')
    
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='Load from checkpoint')
    parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false', help='Do not load checkpoint')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train_sym_model(shuffle = args.shuffle,
                    batch_size = args.batch_size,
                    steps_per_epoch = args.steps_per_epoch,
                    nepochs = args.nepochs,
                    bidirectional_audio = args.bidirectional_audio,
                    audio_to_history = args.audio_to_history,
                    aud_memlen = args.aud_memlen,
                    memlen = args.memlen,
                    mem_size = args.mem_size,
                    audio_radius = args.audio_radius,
                    narrow_types = args.narrow_types,
                    train_txt_fp = args.train_txt_fp,
                    test_txt_fp = args.test_txt_fp,
                    model_dir = args.model_dir,
                    load_checkpoint = args.load_checkpoint,
                   use_diff = args.use_diff)