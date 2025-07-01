import argparse

from models import generate_charts

def get_parser():
    parser = argparse.ArgumentParser(description="Generate charts from input songs.")
    
    parser.add_argument('--onset_model_fp', type=str, default='trained_models/onset_model.keras',
                        help='Path to the trained onset model file.')
    parser.add_argument('--sym_model_fp', type=str, default='trained_models/sym_model.keras',
                        help='Path to the trained symmetry model file.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing.')
    parser.add_argument('--model_frame_density', type=int, default=32,
                        help='Frame density used by the model.')
    parser.add_argument('--onset_history_len', type=int, default=15,
                        help='Length of onset history for context.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for prediction acceptance.')
    parser.add_argument('--in_directory', type=str, default='input_songs_for_generation',
                        help='Directory of input songs to process.')
    parser.add_argument('--out_directory', type=str, default='generated_charts',
                        help='Directory where generated charts will be saved.')
    parser.add_argument('--diffs', nargs='+', default=['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge'],
                        help='List of difficulty levels to generate.')
    parser.add_argument('--maxstep', type=int, default=12,
                        help='Maximum step count per measure or chart segment.')
    parser.add_argument('--use_song_length', action='store_true', default=False, help='Use song length parameter. For some stamina models.')
    parser.add_argument('--no-use_song_length', dest='use_song_length', action='store_false', help='Do not use song length parameter')
    parser.add_argument('--bpm_method', type=str, default='DDCL',
                        help='The BPM method to use. Options are DDCL and AV.')

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    generate_charts(
        onset_model_fp=args.onset_model_fp,
        sym_model_fp=args.sym_model_fp,
        batch_size=args.batch_size,
        model_frame_density=args.model_frame_density,
        onset_history_len=args.onset_history_len,
        threshold=args.threshold,
        in_directory=args.in_directory,
        out_directory=args.out_directory,
        diffs=args.diffs,
        maxstep=args.maxstep,
        use_song_length=args.use_song_length,
        bpm_method=args.bpm_method,
    )