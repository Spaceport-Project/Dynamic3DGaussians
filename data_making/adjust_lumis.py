# gray_l_equalize.py
# Requires: Python ≥3.8, OpenCV-Python (cv2), NumPy, pathlib
import argparse
import os
import cv2, numpy as np
from pathlib import Path

# ------------------------------ parameters -----------------------------------
# in_dir  = Path("input")          # folder that contains 0.png … 42.png
# out_dir = Path("output")         # output folder (will be created)

# # hamit_burak_1 settings
gray_L_max   = 70  
gray_L_min   = 30              # L* < 40 is treated as "dark gray" (0–100 scale → 0–255 in OpenCV)
gray_ab_tol  = 0.07              # ±5 % of full a*, b* span  → 0.05*255 ≈ ±13
statistic    = "median"          # "mean" or "median" for the reference & per-image stats
min_gray_pix = 100               # safety: need at least this many gray pixels to compute scale
scale_bounds = (0.3, 2.5)        # clamp for numerical stability
# -----------------------------------------------------------------------------
# tahir_2 settings
# gray_L_max   = 50  
# gray_L_min   = 15              # L* < 40 is treated as "dark gray" (0–100 scale → 0–255 in OpenCV)
# gray_ab_tol  = 0.09            # ±5 % of full a*, b* span  → 0.05*255 ≈ ±13
# statistic    = "median"          # "mean" or "median" for the reference & per-image stats
# min_gray_pix = 100               # safety: need at least this many gray pixels to compute scale
# scale_bounds = (0.3, 2.5)       



def load_lab(path: Path):
    """Read PNG and return its Lab channels (OpenCV uses L 0-255, a/b 0-255)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a, b = cv2.split(lab)
    a -= 128.0   # re-centre a,b around 0
    b -= 128.0
    return L, a, b, bgr.shape


def gray_mask(L, a, b):
    """Boolean mask of pixels considered gray (dark enough + low chroma)."""
    mask_L =  (L < gray_L_max * 255 / 100)  & (L > gray_L_min * 255 / 100)    # L range given in 0–100
    tol    = gray_ab_tol * 100
    mask_ab = (np.abs(a) < tol) & (np.abs(b) < tol)
    return mask_L & mask_ab


def robust_stat(values):
    return np.median(values) if statistic == "median" else np.mean(values)


# # --------------------------- 1. reference image ------------------------------
# ref_L, ref_a, ref_b, shape = load_lab(in_dir / "0.png")
# ref_mask = gray_mask(ref_L, ref_a, ref_b)
# if ref_mask.sum() < min_gray_pix:
#     raise RuntimeError("Not enough gray pixels in reference image")
# ref_L_gray = ref_L[ref_mask]
# L_ref = robust_stat(ref_L_gray)

def save_gray_with_transparency(bgr_image, mask, output_path):
    """Save image with gray areas visible and non-gray areas transparent."""
    # Convert BGR to BGRA (add alpha channel)
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
    
    # Set alpha channel: 255 for gray pixels, 0 for non-gray pixels
    bgra[:, :, 3] = mask.astype(np.uint8) * 255
    
    cv2.imwrite(str(output_path), bgra)
# --------------------------- 2. batch normalisation --------------------------
# out_dir.mkdir(parents=True, exist_ok=True)

def adjust_lumis(path, ref_ind=0):
    idx =0
    listdir = sorted(os.listdir(path))
    listdir = [ el for el in listdir if "_black.png" in el]
    listdir.remove(f'{ref_ind}_black.png')
    listdir.insert(0, f'{ref_ind}_black.png')  # Insert at front  

    for  file in (listdir):
        # print(file)
        # if file.endswith("black.png"):
            
            
        input_path = Path(path) / file
        out_path = input_path # out_dir / file
        if idx == 0 :
            # print("------------------------------------------------------",input_path)
            ref_L, ref_a, ref_b, shape = load_lab(input_path)
            ref_mask = gray_mask(ref_L, ref_a, ref_b)
            if ref_mask.sum() < min_gray_pix:
                raise RuntimeError("Not enough gray pixels in reference image")
            ref_L_gray = ref_L[ref_mask]
            L_ref = robust_stat(ref_L_gray)
            print(f"Reference gray L* (Lab 0-255 scale): {L_ref:.1f}")
            # im = cv2.imread(input_path)
            # save_gray_with_transparency(im, ref_mask, out_path)

        else:
        
            L, a, b, shp = load_lab(input_path)
            m = gray_mask(L, a, b)

            if m.sum() < min_gray_pix:
                # not enough gray pixels → copy unchanged
                cv2.imwrite(str(out_path), cv2.imread(str(input_path)))
                print(f"{idx}.png  (skipped – insufficient gray)")
                continue

            L_curr = robust_stat(L[m])
            scale  = np.clip(L_ref / (L_curr + 1e-6), *scale_bounds)

            # scale full L channel
            L_eq   = np.clip(L * scale, 0, 255)

            # merge and convert back to BGR for saving
            lab_eq = cv2.merge([L_eq, a + 128.0, b + 128.0]).astype(np.uint8)
            bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
            # save_gray_with_transparency(bgr_eq, m, out_path)  
            cv2.imwrite(str(out_path), bgr_eq)
        # if idx > 0:
        #     print(f"{idx}.png  gray-L {L_curr:.1f} → scaled by {scale:.3f}")
        idx +=1


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_folder', type=str, default='', help='Path to the input data.')
    # args.add_argument('--output_path', type=str, default='data/', help='Path to the output data.')
    # args.add_argument('--dataset_name', type=str, default='', help='Dataset name.')

    args = args.parse_args()
    adjust_lumis(args.input_folder, 3)
    # print("Done! Equalised images are in:")
