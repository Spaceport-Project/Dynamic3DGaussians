import argparse
import os
import time
from loguru import logger
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
import sys
# sys.path.append('/home/')

sys.path.append('/home/hamit/Softwares/segment-anything')
sys.path.append('/home/hamit/Softwares/YOLOX/yolox')
sys.path.append('/home/hamit/Softwares/YOLOX/')

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from segment_anything import sam_model_registry, SamPredictor

from typing import Sequence
from copy import deepcopy


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def image_color_stats(image, crop_width_ratio=1.0, crop_height_ratio=1.0):
    """
    Calculate the color statistics of an image, cropping it to a central patch.

    Args:
    - image_path (str): Path to the image file.
    - crop_width_ratio (float): The ratio of the width to keep (0.5 will keep the central half of the width).
    - crop_height_ratio (float): The ratio of the height to keep (1.0 will keep the full height).

    Returns:
    - tuple: Mean and standard deviation of the LAB channels of the cropped image.
    """
    # Calculate the cropping coordinates based on the specified ratios
    height, width = image.shape[:2]
    x_crop_start = int((1 - crop_width_ratio) * width // 2)
    y_crop_start = int((1 - crop_height_ratio) * height // 2)
    x_crop_end = x_crop_start + int(crop_width_ratio * width)
    y_crop_end = y_crop_start + int(crop_height_ratio * height)

    # Crop the central part of the image
    central_crop = image[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
    
    # Extract the alpha channel and identify opaque pixels
    alpha_channel = central_crop[:, :, 3]
    opaque_mask = alpha_channel == 255

    # Convert the cropped image from BGR to LAB color space, processing only opaque pixels
    central_crop_rgb = central_crop[:, :, :3]  # Extract RGB channels
    central_crop_rgb[~opaque_mask] = 0  # Set non-opaque pixels to black
    image_lab = cv2.cvtColor(central_crop_rgb, cv2.COLOR_BGR2LAB)
    
    # Calculate mean and std dev for each channel in the LAB color space
    mean, std_dev = cv2.meanStdDev(image_lab)
    
    # Flatten the mean and std_dev arrays and round the values
    mean = mean.flatten().round(2)
    std_dev = std_dev.flatten().round(2)
    
    return mean, std_dev

def apply_color_transformation(target_image, mean_diff, std_diff):
    """
    Apply color transformation to an image to adjust its color statistics.

    Args:
    - target_image (numpy.ndarray): The image to transform, in BGR color space.
    - mean_diff (numpy.ndarray): The differences in the means of the LAB channels.
    - std_diff (numpy.ndarray): The ratios of the standard deviations of the LAB channels.

    Returns:
    - numpy.ndarray: The color-adjusted image in BGR color space.
    """
    # Extract the alpha channel and identify opaque pixels
    alpha_channel = target_image[:, :, 3]
    opaque_mask = alpha_channel == 255
    
    # Process only the RGB channels
    target_image_rgb = target_image[:, :, :3]
    target_image_rgb[~opaque_mask] = 0  # Set non-opaque pixels to black

    # Convert image from BGR to LAB color space
    lab_image = cv2.cvtColor(target_image_rgb, cv2.COLOR_BGR2LAB)

    # Split into channels
    l, a, b = cv2.split(lab_image)

    # Apply the mean difference to each channel
    l = l.astype(np.float32) + mean_diff[0]
    a = a.astype(np.float32) + mean_diff[1]
    b = b.astype(np.float32) + mean_diff[2]

    # Scale the L channel from 0-255 to 0-100 range for LAB
    l = (l / 255) * 100

    # Apply the standard deviation ratio to L channel, after scaling to 0-100 range
    l_std_ratio = std_diff[0] if std_diff[0] != 0 else 1
    l = ((l - np.mean(l)) * l_std_ratio) + np.mean(l)

    # Scale L back to 0-255 range from 0-100, clip and convert to uint8
    l = np.clip((l / 100) * 255, 0, 255).astype(np.uint8)

    # Ensure the a and b channels are properly clipped to 0-255 range
    a_std_ratio = std_diff[1] if std_diff[1] != 0 else 1
    b_std_ratio = std_diff[2] if std_diff[2] != 0 else 1

    # Adjust a and b channels according to standard deviation ratio
    a = np.clip((a - np.mean(a)) * a_std_ratio + np.mean(a), 0, 255).astype(np.uint8)
    b = np.clip((b - np.mean(b)) * b_std_ratio + np.mean(b), 0, 255).astype(np.uint8)

    # Merge the channels back together
    adjusted_lab = cv2.merge([l, a, b])

    # Convert back from LAB to BGR color space
    adjusted_image = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    # After color transformation, merge the alpha channel back
    adjusted_image_rgba = cv2.merge([adjusted_image, alpha_channel])

    return adjusted_image_rgba

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        
        # Create a list to hold bounding box data
        bboxes_data = []
        for i, box in enumerate(bboxes):
            bbox = box.tolist()
            cls_id = int(cls[i])
            score = float(scores[i])
            bbox_data = {
                "bbox": bbox,
                "class_id": cls_id,
                "class_name": self.cls_names[cls_id],
                "score": score
            }
            bboxes_data.append(bbox_data)

        return vis_res, bboxes_data
    

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--adjust_color",
        dest="adjust_color",
        default=False,
        action="store_true",
        help="Adjust the color of the masked image to match consecutive frames.",
    )
    parser.add_argument(
        "--sam_checkpoint", 
        type=str, 
        default="/home/hamit/Softwares/repos/cloned_repos/segment-anything/model_weights/sam_vit_h_4b8939.pth", 
        help="SAM checkpoint path")
    parser.add_argument(
        "--sam_model_type", 
        type=str, 
        default="vit_h", 
        help="SAM model type")
    parser.add_argument('--add_background', action='store_true', help='Flag to add background to the masked images otherwise the background will be blackish')
    parser.add_argument('--background_imgs_path', type=str, help='Path to the background image files')
    parser.add_argument('--mask_video_output', type=str, help='Output directory for mask videos')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for the foreground image')
    return parser

def get_model_info(model: torch.nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for id, filename in enumerate(sorted(file_name_list)):
            # if id > 99:
            #     break
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def image_inference(yolo_predictor, sam_predictor, vis_folder, path, output_base_path, current_time):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    cnt=0
    for image_name in files:
        # if cnt > 1:
        #     break
        cnt += 1
        try:    
            logger.info("Processing {}.".format(image_name))
            outputs, img_info = yolo_predictor.inference(image_name)
           
            # vis_res, bboxes_data = yolo_predictor.visual(outputs[0], img_info)
            
            # # Here add segment anything segmantation code that uses bbox_data

            # # First filter out the bboxes that are not person
            # person_bboxes = []
            # for bbox_data in bboxes_data:
            #     if bbox_data["class_name"] == "person" or bbox_data["class_name"] == "dog":
            #         person_bboxes.append(bbox_data["bbox"])
            #         break
            # person_bboxes = torch.tensor(person_bboxes, device="cuda")
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # sam_predictor.set_image(image)
            
            # masks, _, _ = sam_predictor.predict(
            #     box = person_bboxes.cpu().numpy(),
            #     multimask_output=True

            # )


            # Create a blank alpha channel
            alpha_channel = np.zeros(image.shape[:2], dtype=np.uint8)
            # rgba_image[:, :, 3] = alpha_channel  

            # for mask in masks:
            #     # Ensure mask is a 2D boolean array
            #     # mask_np = mask.cpu().numpy().squeeze()  # Squeeze in case it's 1 x H x W
            #     mask_np = mask.squeeze() 
            #     if mask_np.ndim == 2:
            #         alpha_channel[mask_np] = 255  # Set alpha channel to opaque for masked areas
           
            # box = person_bboxes.cpu().numpy()
            # x, y, x_w, y_h = box.astype(int).flatten().tolist()
            # x = x - 600 if x - 600 >= 0 else 0
            # y = y -300 if  y -300 >=0 else 0
            # x_w = x_w + 600 if x_w + 600 <= image.shape[1] else image.shape[1]
            # y_h = y_h + 300 if y_h + 300 <= image.shape[0] else image.shape[0]

            x = 980
            y = 600
            x_w = image.shape[1]
            y_h = image.shape[0]
            alpha_channel[y:y_h, x:x_w] = 255  
 
            # Apply morphological operations to aplha channel to remove the holes inside the mask
            kernel = np.ones((7,7),np.uint8)
            alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_CLOSE, kernel)
            
            # Add the alpha channel to the original image
            rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            rgba_image[:, :, 3] = alpha_channel
            norm_data = rgba_image / 255.0

            # bg = np.array([1,1,1])
            # bg = np.array([8.0/255, 39.0/255, 245.0/255]) # blue screen
            # bg = np.array([100.0/255, 100.0/255, 100.0/255]) # grey screen
            # bg = np.array([4.0/255, 244.0/255, 4.0/255]) # green screen 1
            
            # bg = np.array([0, 177./255, 64.0/255]) # green screen 2
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # rgb_save = arr * 255.0

            #Black background
            bg = np.array([0,0,0])
            arr1 = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            rgb_save1 = arr1 * 255.0


            # Background black, foreground white
            bg = np.array([0, 0, 0])  # black background
            fg = np.array([1, 1, 1])  # white foreground
            # norm_data = rgba_image / 255.0

            arr2 = fg * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            rgb_save2 = arr2 * 255.0


            vis_name = os.path.basename(image_name)
            vis_path = os.path.join(vis_folder, vis_name)
            #cv2.imwrite(vis_path, vis_res)
            json_path = os.path.join(vis_folder, vis_name.replace(".jpg", ".json"))
            # with open(json_path, "w") as f:
            #     json.dump(bboxes_data, f)      

            # Determine the subfolder structure
            subfolder_path = os.path.relpath(os.path.dirname(image_name), path)
            output_subfolder_path = os.path.join(output_base_path, subfolder_path)

            # Create the subfolder in the output path if it does not exist
            os.makedirs(output_subfolder_path, exist_ok=True)

            # Save the processed image to the output subfolder
            output_image_name = os.path.basename(image_name).replace(".jpg", "_masked.png")
            output_image_path = os.path.join(output_subfolder_path, output_image_name)
            print('Saving image to {}'.format(output_image_path))
            # rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
            # cv2.imwrite(output_image_path, rgba_image)

            # rgb_save = cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(output_image_path, rgb_save)

            #Green background
            # save_rgb = Image.fromarray(np.array(rgb_save, dtype=np.byte), "RGB")
            # save_rgb.save(output_image_path)

            # Black background
            output_image_name1 = output_image_name.replace(".png", "_black.png")
            output_image_path = os.path.join(output_subfolder_path, output_image_name1)
            save_rgb1 = Image.fromarray(np.array(rgb_save1, dtype=np.byte), "RGB")
            save_rgb1.save(output_image_path)


            # Black and white
            output_image_name2 = output_image_name.replace(".png", "_black_white.png")
            output_image_path = os.path.join(output_subfolder_path, output_image_name2)
            save_rgb2 = Image.fromarray(np.array(rgb_save2, dtype=np.byte), "RGB")
            save_rgb2.save(output_image_path)



            # Optionally save the original YOLO visualization and JSON data
            vis_name = os.path.basename(image_name)
            vis_path = os.path.join(output_subfolder_path, vis_name)
            json_path = os.path.join(output_subfolder_path, vis_name.replace(".jpg", ".json"))
            #cv2.imwrite(vis_path, vis_res)
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            continue

def video_inference(yolo_predictor, sam_predictor, vis_folder, path, output_base_path):
    # Check if the input path is a video file
    print("Starting video inference for path: {}".format(path))
    if path.endswith(".mp4") or path.endswith(".avi") or path.endswith(".mov"):
        print("passing....")
        pass
    elif os.path.isdir(path):
        # Make sure video_list only consists of video files
        file_list = os.listdir(path)
        video_list = [f_name for f_name in file_list if f_name.endswith(".mp4") 
                      or f_name.endswith(".avi") or f_name.endswith(".mov")]
        video_list.sort()
        print("List of files to process: {}".format(video_list))
        # Loop through the video files
        for video_name in video_list:
            print("Processing video: {}".format(video_name))
            cap = cv2.VideoCapture(os.path.join(path, video_name))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)

            mask_vid_writer = None
            if args.save_result:
                if not os.path.exists(output_base_path):
                    os.mkdir(output_base_path)
                out_video_path = os.path.join(output_base_path, video_name)
                vid_writer = cv2.VideoWriter(
                    out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
                )
                mask_dir = os.path.join(output_base_path, "mask")
                if not os.path.exists(mask_dir):
                    os.mkdir(mask_dir)
                mask_output_path = os.path.join(mask_dir, video_name)  # Prefix with 'mask_' to differentiate
                mask_output_path = mask_output_path.replace(".mp4", ".avi")
                print("mask_output_path: {}".format(mask_output_path))
                mask_vid_writer = cv2.VideoWriter(
                    mask_output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (int(width), int(height)), isColor=False
                )
                
                # For saving video with alpha channel
                # out_video_path_2 = os.path.join(output_base_path, video_name.split(".")[0] + ".avi")
                # codec = cv2.VideoWriter_fourcc(*'FMP4')  # Codec that supports alpha channel   
                # vid_writer_2 = cv2.VideoWriter(
                #     out_video_path_2, codec, fps, (int(width), int(height)), isColor=True
                # )
            
            bg_image = None
            if args.add_background and args.background_imgs_path:
                video_name = os.path.splitext(video_name)[0]  # Remove the extension
                numeric_part = video_name.replace("output_grid_", "").replace(".mp4", "")
                numeric_part = str(int(numeric_part) + 1)
                # Construct the corresponding image name with leading zeros
                image_name = f"{numeric_part.zfill(3)}.png"
                bg_image = cv2.imread(os.path.join(args.background_imgs_path, image_name))
                if bg_image is None:
                    raise ValueError(f"Background image at {args.background_imgs_path} could not be loaded.")
                # Resize background image to match the frames if needed
                bg_image = cv2.resize(bg_image, (int(width), int(height)))
                bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            
            frame_count = 0
            # Or while cap.isOpened():
            while True:
                ret, frame = cap.read()
                if ret:
                    #cv2.imwrite(output_base_path+"/oframe_rgb"+str(frame_count)+".png", frame)
                    frame_copy = frame.copy()
                    outputs, img_info = yolo_predictor.inference(frame)
                    vis_res, bboxes_data = yolo_predictor.visual(outputs[0], img_info)

                    person_bboxes = []
                    for bbox_data in bboxes_data:
                        if bbox_data["class_name"] == "person":
                            person_bboxes.append(bbox_data["bbox"])

                    # # Check if current frame has detected persons
                    # if person_bboxes:
                    #     # Update stored bounding boxes
                    #     stored_bboxes = torch.tensor(person_bboxes, device="cuda")
                    #     transformed_boxes = stored_bboxes
                    # elif stored_bboxes is not None:
                    #     # Use stored bounding boxes if current frame has no detected persons
                    #     transformed_boxes = stored_bboxes
                    # else:
                    #     # If there's sinning of the video), you might want to skip or handle it differently
                    #     continue  # or handle this case as you see fit
                    

                    # frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    # sam_predictor.set_image(frame_copy)
                    # transformed_boxes = sam_predictor.transform.apply_boxes_torch(transformed_boxes, frame_copy.shape[:2])
                    # #print(transformed_boxes.shape)

                    # masks, _, _ = sam_predictor.predict_torch(point_coords=None,
                    #     point_labels=None,
                    #     boxes=transformed_boxes,
                    #     multimask_output=False,
                    #     )

                    if person_bboxes:
                        # Update stored bounding boxes
                        stored_bboxes = torch.tensor(person_bboxes, device="cuda")
                        transformed_boxes = stored_bboxes

                        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                        sam_predictor.set_image(frame_copy)
                        transformed_boxes = sam_predictor.transform.apply_boxes_torch(transformed_boxes, frame_copy.shape[:2])

                        masks, _, _ = sam_predictor.predict_torch(point_coords=None,
                                                                point_labels=None,
                                                                boxes=transformed_boxes,
                                                                multimask_output=False,
                                                                )
                        
                        # We need to apply morphological operations to the masks to make it smaller
                        # and remove the noise
                        kernel = np.ones((5,5),np.uint8)
                        for i, mask in enumerate(masks):
                            mask = mask.cpu().numpy().squeeze()
                            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                            masks[i] = torch.from_numpy(mask).to(device="cuda")
                    else:
                        # If no persons are detected, skip SAM model prediction and set masks to all black
                        masks = [np.zeros(frame_copy.shape[:2], dtype=np.uint8)]  # Create a single black mask

                    alpha_channel = np.zeros(frame_copy.shape[:2], dtype=np.uint8)
                    
                    for mask in masks:
                        # Ensure mask is a 2D boolean array
                        mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy().squeeze()  # Squeeze in case it's 1 x H x W
                        if mask_np.ndim == 2:
                            alpha_channel[mask_np] = 255  # Set alpha channel to opaque for masked areas

                    # Check for intermediate values
                    unique_values = np.unique(alpha_channel)
                    intermediate_values = [val for val in unique_values if val not in (0, 255)]

                    # If there are intermediate values, threshold the alpha channel
                    if intermediate_values:
                        # Threshold value can be adjusted as needed. A common choice might be 127 for a mid-point threshold.
                        threshold_value = 227
                        alpha_channel = np.where(alpha_channel > threshold_value, 255, 0)

                    if mask_vid_writer is not None:
                        # Convert alpha_channel to 3 channels to match the writer's expectation, but it will still be B/W
                        #mask_frame = np.repeat(alpha_channel[:, :, np.newaxis], 3, axis=2)
                        #cv2.imwrite(output_base_path+"/frame_alpha"+str(frame_count)+".png", mask_frame)
                        mask_vid_writer.write(alpha_channel)

                    # Add the alpha channel to the original image
                    rgba_image = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2RGBA)
                    rgba_image[:, :, 3] = alpha_channel
                    
                    if args.adjust_color:
                        if frame_count == 0:
                            # Calculate the color statistics of the original image
                            target_stats = image_color_stats(rgba_image)
                            print("mean_orig: {}, std_dev_orig: {}".format(target_stats[0], target_stats[1]))
                        else:
                            # Calculate the color statistics of the current frame
                            current_stats = image_color_stats(rgba_image)
                            #print("mean_curr: {}, std_dev_curr: {}".format(current_stats[0], current_stats[1]))

                            # Calculate the differences in the means and standard deviations
                            mean_diff = target_stats[0] - current_stats[0]
                            std_diff = target_stats[1] / current_stats[1]
                            #print("mean_diff: {}, std_dev_diff: {}".format(mean_diff, std_diff))

                            # Apply the color transformation to the current frame
                            rgba_image = apply_color_transformation(rgba_image, mean_diff, std_diff)

                    # For debugging purposes save rgba_image
                    #rgba_image1 = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
                    #frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
                    #cv2.imwrite(output_base_path+"/frame_rgb"+str(frame_count)+".png", frame_copy)
                    #cv2.imwrite(output_base_path+"/rgba_image"+str(frame_count)+".png", rgba_image1)
                    #vid_writer_2.write(rgba_image)
                    

                    # Split the channels from RGBA image
                    B, G, R, A = cv2.split(rgba_image)

                    # Create a white background image
                    height, width, _ = rgba_image.shape
                    
                    # background color
                    color = (38, 42, 55)  # RGB color same as in nerfstudio background

                    # Create a background image of the specified color
                    background = np.ones((height, width, 3), dtype=np.uint8)
                    background[:] = color  # Set the color for the entire background

                    # Normalize the alpha channel
                    alpha_channel = A / 255.0

                    # # Foreground and background images, prepped for blending
                    # foreground = cv2.merge([B, G, R]).astype(float)
                    # background = background.astype(float)

                    # # for debugging purposes save foreground and background images
                    # #cv2.imwrite(output_base_path+"/foreground"+str(frame_count)+".png", foreground)
                    # #cv2.imwrite(output_base_path+"/background"+str(frame_count)+".png", background)

                    # # Blend the foreground (original image) with the background
                    # # The operation is: (foreground * alpha) + (background * (1 - alpha))
                    # foreground *= alpha_channel[:, :, np.newaxis]
                    # #background *= (1 - alpha_channel)[:, :, np.newaxis]

                    # if args.add_background:
                    #     # Use the preloaded background image
                    #     if bg_image is not None:
                    #         # Normalize the alpha channel
                    #         alpha_channel = A / 255.0
                            
                    #         # Prepare the background, repeat the alpha channel 3 times to make it 3D
                    #         alpha_channel_3d = np.repeat(alpha_channel[:, :, np.newaxis], 3, axis=2)
                            
                    #         # Use the alpha channel to blend the foreground with the background
                    #         foreground = cv2.merge([B, G, R]).astype(float)
                    #         background = bg_image.astype(float) * (1 - alpha_channel_3d)
                    #         result_image = cv2.add(foreground, background).astype(np.uint8)

                    # else:
                    #     background *= (1 - alpha_channel)[:, :, np.newaxis]
                    #     result_image = cv2.add(foreground, background).astype(np.uint8)

                    # ...
                    # scale_factor is the factor by which the foreground should be scaled
                    scale_factor = args.scale_factor  # Assuming the scale factor is provided in args

                    # After obtaining alpha_channel, B, G, R, and creating the background:
                    # 1. Scale the foreground and the alpha channel
                    foreground = cv2.merge([B, G, R]).astype(float)
                    height_fg, width_fg, _ = foreground.shape
                    scaled_height = int(height_fg * scale_factor)
                    scaled_width = int(width_fg * scale_factor)

                    # Resize the foreground and the alpha channel
                    foreground = cv2.resize(foreground, (scaled_width, scaled_height))
                    alpha_channel_scaled = cv2.resize(alpha_channel, (scaled_width, scaled_height))

                    # 2. Calculate the position where the foreground should be placed
                    x_offset = (width - scaled_width) // 2  # Center in width
                    #y_offset = height - scaled_height       # Start from the bottom in height
                    # y_offset = (height - scaled_height) // 2  # Center in height
                    y_offset = ((height - scaled_height) // 16) * 10  # Center in height
                    # 3. Create an empty matrix for the new alpha channel with the size of the background
                    alpha_channel_new = np.zeros((height, width), dtype=np.uint8)

                    # Place the scaled alpha channel in the new alpha channel matrix
                    alpha_channel_new[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width] = alpha_channel_scaled

                    # Create an empty matrix for the new foreground with the size of the background
                    foreground_new = np.zeros((height, width, 3), dtype=float)

                    # Place the scaled foreground in the new foreground matrix
                    foreground_new[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width, :] = foreground

                    # 4. Blend the new foreground with the background using the new alpha channel
                    #alpha_channel_new_3d = alpha_channel_new[:, :, np.newaxis] / 255.0  # Normalize and expand dimensions
                    alpha_channel_new_3d = np.repeat(alpha_channel_new[:, :, np.newaxis], 3, axis=2)

                    #foreground_new = cv2.cvtColor(foreground_new.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    if args.add_background:
                        if bg_image is not None:
                            background = bg_image.astype(float) * (1 - alpha_channel_new_3d)
                            result_image = cv2.add(foreground_new, background).astype(np.uint8)

                    else:
                        background = background.astype(float) * (1 - alpha_channel_new_3d)
                        result_image = cv2.add(foreground_new, background).astype(np.uint8)

                    # Now result_image RGB with white background, we can save it using videowriter
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                    vid_writer.write(result_image)
                    frame_count += 1

                else:
                    if vid_writer is not None:
                        vid_writer.release()
                    if mask_vid_writer is not None:
                        mask_vid_writer.release()
                    break

def image_inference_fixed_bbox(yolo_predictor, sam_predictor, vis_folder, path, output_base_path):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    for image_name in files:
        logger.info("Processing {}.".format(image_name))
        
        input_boxes = torch.tensor([
                        [948, 174, 1706, 1102],
                    ], device=sam_predictor.device)

        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])    
        masks, _, _ = sam_predictor.predict_torch(point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            )

        # Create a blank alpha channel
        alpha_channel = np.zeros(image.shape[:2], dtype=np.uint8)

        for mask in masks:
            # Ensure mask is a 2D boolean array
            mask_np = mask.cpu().numpy().squeeze()  # Squeeze in case it's 1 x H x W
            if mask_np.ndim == 2:
                alpha_channel[mask_np] = 255  # Set alpha channel to opaque for masked areas

        # Add the alpha channel to the original image
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba_image[:, :, 3] = alpha_channel

        bg = np.array([1,1,1])
        norm_data = rgba_image / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        rgb_save = arr * 255.0
   

        # Determine the subfolder structure
        subfolder_path = os.path.relpath(os.path.dirname(image_name), path)
        output_subfolder_path = os.path.join(output_base_path, subfolder_path)

        # Create the subfolder in the output path if it does not exist
        os.makedirs(output_subfolder_path, exist_ok=True)

        # Save the processed image to the output subfolder
        output_image_name = os.path.basename(image_name).replace(".jpg", "_masked.png")
        output_image_path = os.path.join(output_subfolder_path, output_image_name)
        print('Saving image to {}'.format(output_image_path))


        save_rgb = Image.fromarray(np.array(rgb_save, dtype=np.byte), "RGB")
        save_rgb.save(output_image_path)


def video_inference_fixed_bbox(sam_predictor, path, output_base_path, fixed_bounding_boxes):
    print("Starting video inference for path: {}".format(path))
    if path.endswith(".mp4") or path.endswith(".avi") or path.endswith(".mov"):
        pass
    elif os.path.isdir(path):
        file_list = os.listdir(path)
        video_list = [f_name for f_name in file_list if f_name.endswith(".mp4") 
                      or f_name.endswith(".avi") or f_name.endswith(".mov")]
        video_list.sort()
        print("List of files to process: {}".format(video_list))
        for video_name in video_list:
            print("Processing video: {}".format(video_name))
            cap = cv2.VideoCapture(os.path.join(path, video_name))
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_video_path = os.path.join(output_base_path, video_name)
            # vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_copy = frame.copy()
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    # Use fixed bounding boxes for this frame
                    input_boxes = torch.tensor([
                        [400, 120, 1204, 1100],
                    ], device=sam_predictor.device)
                    # input_points = np.array([[754, 692], [910, 698]])
                    # input_labels = np.array([1, 1])

                    input_points = np.array([[754, 692],[910, 698]])
                    input_labels = np.array([1, 0])
                    input_points = torch.from_numpy(input_points).to('cuda')
                    input_labels = torch.from_numpy(input_labels).to('cuda')


                    sam_predictor.set_image(frame_copy)
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, frame_copy.shape[:2])
                    masks, _, _ = sam_predictor.predict_torch(
                        point_coords=input_points,
                        point_labels= input_labels,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                    alpha_channel = np.zeros(frame_copy.shape[:2], dtype=np.uint8)

                    for mask in masks:
                        print(len(masks))
                        # Ensure mask is a 2D boolean array
                        mask_np = mask.cpu().numpy().squeeze()  # Squeeze in case it's 1 x H x W
                        if mask_np.ndim == 2:
                            alpha_channel[mask_np] = 255  # Set alpha channel to opaque for masked areas

                    # Add the alpha channel to the original image
                    rgba_image = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2RGBA)
                    rgba_image[:, :, 3] = alpha_channel

                    bg = np.array([1,1,1])
                    norm_data = rgba_image / 255.0
                    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    rgb_save = arr * 255.0
                  
                    output_image_name = str(frame_count) + ".png"
                    output_base_path = "/media/hamit/Elements/21-02-2024_DATA/test_masking/output"
                    output_image_path = os.path.join(output_base_path, output_image_name)
                    print('Saving image to {}'.format(output_image_path))
                    # rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
                    # cv2.imwrite(output_image_path, rgba_image)

                    # rgb_save = cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(output_image_path, rgb_save)

                    save_rgb = Image.fromarray(np.array(rgb_save, dtype=np.byte), "RGB")
                    save_rgb.save(output_image_path)
                    frame_count += 1

# Usage example
fixed_bounding_boxes = [
    # Format: [x1, y1, x2, y2]
    [469, 142, 1104, 949]
]


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )

    sam_checkpoint = args.sam_checkpoint
    sam_model_type = args.sam_model_type
    sam_device = "cuda"
    sam = sam_model_registry[sam_model_type](checkpoint = sam_checkpoint)
    sam.to(sam_device)
    sam_predictor = SamPredictor(sam)
    current_time = time.localtime()
    paren_dir = os.path.dirname(args.path)

    if args.demo == "images":
        # paren_dir=os.path.dirname(paren_dir)
        output_path = os.path.join(paren_dir, 'masked_undistorted_images')
        print("image inference starting...")
        image_inference(predictor, sam_predictor, vis_folder, args.path, output_path, current_time)
        print("image inference done!")
    elif args.demo == "video":
        output_path = os.path.join(paren_dir, 'masked_videos')
        print('Output path: {}'.format(output_path))
        video_inference(predictor, sam_predictor, vis_folder, args.path, output_path)
    # output_path = os.path.join(paren_dir, 'masked_videos')
    # # video_inference_fixed_bbox(sam_predictor, args.path, output_path, fixed_bounding_boxes)
    # image_inference_fixed_bbox(predictor, sam_predictor, vis_folder, args.path, output_path)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    print(args.exp_file, args.name)
    print(args,"and", exp)
    main(exp, args)

