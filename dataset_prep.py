import os
import argparse
from utils import align_and_crop

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('jpg','jpeg','png'))]
    for p in files:
        fname = os.path.basename(p)
        outp = os.path.join(args.out_dir, fname)
        
        ok = align_and_crop(p, outp, output_size=args.size)
        if not ok:
            print('skip', p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--size', type=int, default=1024)
    args = parser.parse_args()
    main(args)
