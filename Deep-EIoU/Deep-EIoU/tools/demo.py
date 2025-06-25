import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from reid.torchreid.utils import FeatureExtractor
import torchvision.transforms as T

from tracker.yolov11_det import YOLOv11Detector

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="../demo.mp4", help="path to images or video"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="yolox/yolox_x_ch_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
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
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    # Modifition Point
    # YOLOv11 args, model weight path, defailt det_conf
    parser.add_argument('--detector', default='npy', choices=['npy', 'yolov11'], help='detector type')
    parser.add_argument('--det_ckpt', default='/home/y_li/workspace3/ultralytics/yolov11-finetune/train/weights/best.pt', help='yolov11 model path')
    parser.add_argument('--det_conf', type=float, default=0.2, help='confidence threshold for detection')

    # SigLIP2 ReID args
    parser.add_argument("--use_siglip2", action="store_true", default=False, help="use SigLIP2 for ReID feature extraction")
    parser.add_argument("--siglip2_model", type=str, default="google/siglip2-base-patch16-224", help="SigLIP2 model name from Hugging Face Hub")
    parser.add_argument("--siglip2_quantization", action="store_true", default=False, help="use 4-bit quantization for SigLIP2")
    parser.add_argument("--siglip2_text_prompts", type=str, nargs='+', default=None, help="text prompts for SigLIP2 vision-language matching (optional)")
    parser.add_argument("--reid_model_name", type=str, default="osnet_x1_0", help="ReID model name (when not using SigLIP2)")
    parser.add_argument("--reid_model_path", type=str, default="checkpoints/sports_model.pth.tar-60", help="ReID model path (when not using SigLIP2)")

    # サッカー特化パラメータ (from sport_track.py)
    parser.add_argument('--enhanced_reid', action='store_true', default=False, help='use enhanced ReID for soccer')
    parser.add_argument('--enable_id_correction', action='store_true', default=False, help='enable ID correction mechanism')
    parser.add_argument('--correction_buffer_size', type=int, default=10, help='buffer size for ID correction')
    parser.add_argument('--correction_thresh', type=float, default=0.3, help='threshold for ID correction')

    # Soccer-specific preprocessing arguments
    parser.add_argument("--enable_soccer_preprocessing", action="store_true", help="Enable soccer-specific frame preprocessing")
    parser.add_argument("--enhance_contrast", action="store_true", help="Apply CLAHE contrast enhancement")
    parser.add_argument("--gamma_correction", action="store_true", help="Apply gamma correction for better visibility")
    parser.add_argument("--denoise", action="store_true", help="Apply bilateral filtering for noise reduction")
    parser.add_argument("--use_field_roi", action="store_true", help="Focus on field area only")
    parser.add_argument("--adaptive_detection", action="store_true", help="Enable adaptive detection parameters")

    # CMC (Camera Motion Compensation)
    parser.add_argument("--cmc-method", default="none", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    return parser



def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

# Modifition Point
def safe_logit(y, eps=1e-6):
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))
# Modifition Point
def det_conf_inverse_sigmoid(det):
    det_new = det.copy()
    det_new[:, 4] = safe_logit(det[:, 4])
    return det_new


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def preprocess_soccer_frame(frame, args):
    """サッカー映像専用の前処理"""
    processed = frame.copy()

    # 1. 画像品質の向上
    if hasattr(args, 'enhance_contrast') and args.enhance_contrast:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

    # 2. ガンマ補正（暗い部分を明るく）
    if hasattr(args, 'gamma_correction') and args.gamma_correction:
        gamma = 1.2  # 暗い部分を明るく
        processed = np.power(processed/255.0, gamma) * 255
        processed = processed.astype(np.uint8)

    # 3. ノイズ除去
    if hasattr(args, 'denoise') and args.denoise:
        processed = cv2.bilateralFilter(processed, 9, 75, 75)

    # 4. ROI設定（フィールド部分のみ）
    if hasattr(args, 'use_field_roi') and args.use_field_roi:
        # フィールドのマスクを作成（緑色の領域を検出）
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        # 緑色の範囲（フィールド）
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        field_mask = cv2.inRange(hsv, lower_green, upper_green)

        # モルフォロジー演算でマスクを改善
        kernel = np.ones((5,5), np.uint8)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

        # フィールド外を暗くする（完全に黒にせず、少し見えるようにする）
        field_mask_3ch = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
        field_mask_3ch = field_mask_3ch.astype(np.float32) / 255.0
        non_field_mask = 1.0 - field_mask_3ch
        processed = processed.astype(np.float32)
        processed = processed * field_mask_3ch + processed * non_field_mask * 0.3  # 非フィールド部分を30%の明度に
        processed = processed.astype(np.uint8)

    return processed


def adaptive_detection_params(frame, base_conf=0.2):
    """フレームの特徴に基づいて検出パラメータを動的調整"""
    # 画像の明度を計算
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # 明度に基づいて信頼度閾値を調整
    if brightness < 80:  # 暗い場面
        conf_thresh = base_conf * 0.8  # 閾値を下げる
    elif brightness > 180:  # 明るい場面
        conf_thresh = base_conf * 1.2  # 閾値を上げる
    else:
        conf_thresh = base_conf

    return conf_thresh


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

def imageflow_demo(det_or_pre, extractor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = Deep_EIoU(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 200 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
             # Modifition Point
             # YOLOv11 Detector
            if args.detector == 'yolov11':

                timer.tic()
                det = det_or_pre(frame)  #  return np.ndarray, shape=(N,5)
                # Since the score output from YOLOX can exceed 1, when using YOLOv11,
                # the score value should be increased accordingly to avoid being filtered out by tracker.update()
                det = det_conf_inverse_sigmoid(det)
                img_info = {'raw_img': frame}

            # YOLOx Detecot
            else:
                outputs, img_info = det_or_pre.inference(frame, timer)
                det = None if outputs[0] is None else outputs[0].cpu().numpy() # return np.ndarray, shape=(N,5)


            if det is not None and len(det):

                scale = min(1440/width, 800/height)
                # Modifition Point
                # YOLOv11 does not require
                if args.detector != 'yolov11':
                    det /= scale

                rows_to_remove = np.any(det[:, 0:4] < 1, axis=1) # remove edge detection
                det = det[~rows_to_remove]
                # YOLOv11 detector outputs have 5 columns (x1y1x2y2,score)
                cropped_imgs = [frame[max(0,int(y1)):min(height,int(y2)),max(0,int(x1)):min(width,int(x2))] for x1,y1,x2,y2,_ in det]

                # YOLOx detector outputs have 7 columns
                # cropped_imgs = [frame[max(0,int(y1)):min(height,int(y2)),max(0,int(x1)):min(width,int(x2))] for x1,y1,x2,y2,_,_,_ in det]

                embs = extractor(cropped_imgs)
                embs = embs.cpu().detach().numpy()
                online_targets = tracker.update(det, embs)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    if args.detector == 'yolov11':
        logger.info("Using YOLOv11Detector for detection...")
        det_or_pre = YOLOv11Detector(
            ckpt=args.det_ckpt,
            conf=args.det_conf,
            device=args.device
        )

    else:
        model = exp.get_model().to(args.device)
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if not args.trt:
            if args.ckpt is None:
                ckpt_file = "checkpoints/best_ckpt.pth.tar"
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

        if args.fp16:
            model = model.half()  # to FP16

        if args.trt:
            assert not args.fuse, "TensorRT model is not support model fusing!"
            trt_file = osp.join(output_dir, "model_trt.pth")
            assert osp.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
            logger.info("Using TensorRT to inference")
        else:
            trt_file = None
            decoder = None

        det_or_pre = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()

    # Feature extractor initialization with SigLIP2 support
    if args.use_siglip2:
        logger.info("Using SigLIP2 for ReID feature extraction...")

        # Set up quantization if requested
        quantization_config = None
        if args.siglip2_quantization:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                logger.info("Using 4-bit quantization for SigLIP2")
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, falling back to standard precision")

        try:
            extractor = FeatureExtractor(
                use_siglip2=True,
                siglip2_model=args.siglip2_model,
                text_prompts=args.siglip2_text_prompts,
                return_image_features=True,  # For ReID, we want image features
                quantization_config=quantization_config,
                device=str(args.device),
                verbose=True
            )
            logger.info(f"SigLIP2 model loaded successfully: {args.siglip2_model}")

        except Exception as e:
            logger.error(f"Failed to initialize SigLIP2 model: {e}")
            logger.info("Falling back to traditional ReID model...")
            # SigLIP2の初期化に失敗した場合、従来のReIDモデルにフォールバック
            extractor = FeatureExtractor(
                model_name=args.reid_model_name,
                model_path=args.reid_model_path,
                device=str(args.device),
                verbose=True
            )
            logger.info(f"ReID model loaded as fallback: {args.reid_model_name}")

    else:
        logger.info("Using traditional ReID model for feature extraction...")
        extractor = FeatureExtractor(
            model_name=args.reid_model_name,
            model_path=args.reid_model_path,
            device=str(args.device),
            verbose=True
        )
        logger.info(f"ReID model loaded: {args.reid_model_name}")

    # サッカー特化機能のログ出力
    if args.enhanced_reid:
        logger.info("Enhanced ReID for soccer enabled")
    if args.enable_id_correction:
        logger.info(f"ID correction enabled - buffer size: {args.correction_buffer_size}, threshold: {args.correction_thresh}")
    if args.cmc_method != "none":
        logger.info(f"Camera Motion Compensation enabled: {args.cmc_method}")

    # サッカー映像前処理のログ出力
    if hasattr(args, 'enable_soccer_preprocessing') and args.enable_soccer_preprocessing:
        logger.info("Soccer-specific preprocessing enabled:")
        if hasattr(args, 'enhance_contrast') and args.enhance_contrast:
            logger.info("  - CLAHE contrast enhancement")
        if hasattr(args, 'gamma_correction') and args.gamma_correction:
            logger.info("  - Gamma correction for better visibility")
        if hasattr(args, 'denoise') and args.denoise:
            logger.info("  - Bilateral filtering for noise reduction")
        if hasattr(args, 'use_field_roi') and args.use_field_roi:
            logger.info("  - Field ROI focusing")
        if hasattr(args, 'adaptive_detection') and args.adaptive_detection:
            logger.info("  - Adaptive detection parameters based on lighting")

    imageflow_demo(det_or_pre, extractor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
