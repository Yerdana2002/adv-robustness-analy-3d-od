import os.path as osp
import argparse

# Own imports
from summary_table import SummaryTable

def evaluate(base_path, save_path=None, suffix="", full=False, innout_thresh=0.8, input_suffix=""):
    path = osp.join(base_path, f"adversarial_attack.db")
    print(f"Creating results table based on results from {path}...", flush=True)
    if args.save:
        save_path = osp.join(args.save, f"evaluation{args.suffix}.db")
    else:
        save_path = osp.join(base_path, f"evaluation{args.suffix}.db")

    table = SummaryTable(path, save_path=save_path, innout_thresh=args.innout_thresh, full=args.full)

    print("Done! Saved table!", flush=True)


if __name__ == "__main__":
    # Parser arguments
    parser = argparse.ArgumentParser(description='Evaluation Pipeline for Adversarial attacks')
    parser.add_argument('--data_path', default="", help='Experiment Path', type=str)
    parser.add_argument('--suffix', default="", help="Add suffix to file name?", type=str.lower)
    parser.add_argument('--innout_thresh', default=0.8, help="Innout Threshold", type=float)
    parser.add_argument('--save', help='Save Path', type=str)
    parser.add_argument('--input_suffix', default="", help='if the input file has a suffix that deviates fron standart', type=str)
    parser.add_argument('--full', action='store_true',help="Also compute prediction box tables for mAP computation?") # not recommended, this takes a LOT of extra space and computation!!!


    args = parser.parse_args()
    # Run eval
    evaluate(args.data_path, save_path=args.save, suffix=args.suffix, full=args.full, innout_thresh=args.innout_thresh, input_suffix=args.input_suffix)



    