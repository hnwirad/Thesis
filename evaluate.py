import os
import argparse
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_pair_metrics(real_path, fake_path):
    a = np.array(Image.open(real_path).resize((256,256))).astype(np.float32)
    b = np.array(Image.open(fake_path).resize((256,256))).astype(np.float32)
    s = ssim(a, b, channel_axis=-1)
    p = psnr(a, b)
    return s, p

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir')
    parser.add_argument('--fake_dir')
    args = parser.parse_args()
    s_list, p_list = [], []
    for fn in os.listdir(args.real_dir):
        r = os.path.join(args.real_dir, fn)
        f = os.path.join(args.fake_dir, fn)
        if os.path.exists(f):
            s,p = compute_pair_metrics(r,f)
            s_list.append(s); p_list.append(p)
    print('SSIM', np.mean(s_list), 'PSNR', np.mean(p_list))
