import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument(
        '--retarget_cam',
        action='store_true',
        help=(
            'Post-process exported SMPL-X parameters so all frames share a fixed camera: '
            'constant focal length (from first frame) and principal point at image center. '
            'This only affects exported params, not the rendered demo frames.'
        ),
    )
    args = parser.parse_args()
    return args


def retarget_transl_to_fixed_camera(transl, focal_xy, princpt_xy, img_width, img_height):
    """
    Retarget camera translation to a fixed camera intrinsics.

    We assume the original (per-frame) intrinsics are bbox-derived:
      (fx_i, fy_i, cx_i, cy_i)

    We retarget to a fixed camera:
      focal: (fx_ref, fy_ref) = first frame focal
      princpt: (cx_ref, cy_ref) = image center

    This conversion is commonly used for crop-camera -> full-camera mapping:
      z' = z * fx_ref / fx_i
      x' = x + (cx_i - cx_ref) * z / fx_i
      y' = y + (cy_i - cy_ref) * z / fy_i
    """
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"Expected transl shape (T, 3), got {tuple(transl.shape)}")
    if focal_xy.ndim != 2 or focal_xy.shape[1] != 2:
        raise ValueError(f"Expected focal_xy shape (T, 2), got {tuple(focal_xy.shape)}")
    if princpt_xy.ndim != 2 or princpt_xy.shape[1] != 2:
        raise ValueError(f"Expected princpt_xy shape (T, 2), got {tuple(princpt_xy.shape)}")
    if focal_xy.shape[0] != transl.shape[0] or princpt_xy.shape[0] != transl.shape[0]:
        raise ValueError("transl/focal_xy/princpt_xy must have the same T dimension")

    eps = 1e-6
    fx = focal_xy[:, 0].clamp(min=eps)
    fy = focal_xy[:, 1].clamp(min=eps)
    fx_ref = float(fx[0].item())
    fy_ref = float(fy[0].item())

    cx_ref = float(img_width) / 2.0
    cy_ref = float(img_height) / 2.0

    tx, ty, tz = transl[:, 0], transl[:, 1], transl[:, 2]
    cx, cy = princpt_xy[:, 0], princpt_xy[:, 1]

    tz_new = tz * (fx_ref / fx)
    tx_new = tx + (cx - cx_ref) * tz / fx
    ty_new = ty + (cy - cy_ref) * tz / fy

    transl_new = torch.stack([tx_new, ty_new, tz_new], dim=-1)
    focal_ref_xy = torch.tensor([fx_ref, fy_ref], dtype=torch.float32)
    return transl_new, focal_ref_xy

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    # [NEW] Initialize result container
    all_results = {
        'global_orient': [],
        'body_pose': [],
        'transl': [],
        'focal_length': [],
        'focal_length_xy': [],
        'width': [],
        'height': [],
        'princpt': [],
    }

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            # save original image if no bbox
            num_bbox = 0
        elif not args.multi_person:
            # only select the largest bbox
            num_bbox = 1
            areas = (yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (yolo_bbox[:, 3] - yolo_bbox[:, 1])
            yolo_bbox = yolo_bbox[[np.argmax(areas)]]
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img, 
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            # render mesh
            focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                     cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
            princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                       cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]

            # [NEW] Collect current frame data
            # Only collect for bbox_id == 0 (first person) to avoid dimension mismatch
            if bbox_id == 0:
                # global_orient: (1, 3)
                all_results['global_orient'].append(out['smplx_root_pose'].detach().cpu())
                # body_pose: (1, 63)
                all_results['body_pose'].append(out['smplx_body_pose'].detach().cpu())
                # transl: (1, 3)
                all_results['transl'].append(out['cam_trans'].detach().cpu())
                # focal, princpt: (1, 2)
                all_results['focal_length_xy'].append(torch.tensor([[focal[0], focal[1]]], dtype=torch.float32))
                all_results['princpt'].append(torch.tensor([[princpt[0], princpt[1]]], dtype=torch.float32))
                # focal_length, width, height: (1, 1)
                all_results['focal_length'].append(torch.tensor([[focal[0]]], dtype=torch.float32))
                all_results['width'].append(torch.tensor([[original_img.shape[1]]], dtype=torch.float32))
                all_results['height'].append(torch.tensor([[original_img.shape[0]]], dtype=torch.float32))
            
            # draw the bbox on img
            vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                    (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
            # draw mesh
            vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

        # save rendered image
        frame_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])

    # [NEW] Save all parameters to .pt file after loop
    print(f"Saving parameters to {output_folder}...")

    if len(all_results['global_orient']) > 0:
        final_dict = {k: torch.cat(v, dim=0) for k, v in all_results.items()}

        if args.retarget_cam:
            img_w = float(final_dict['width'].flatten()[0].item())
            img_h = float(final_dict['height'].flatten()[0].item())

            final_dict['transl_raw'] = final_dict['transl'].clone()
            final_dict['focal_length_raw'] = final_dict['focal_length'].clone()
            final_dict['focal_length_xy_raw'] = final_dict['focal_length_xy'].clone()
            final_dict['princpt_raw'] = final_dict['princpt'].clone()

            transl_new, focal_ref_xy = retarget_transl_to_fixed_camera(
                transl=final_dict['transl_raw'],
                focal_xy=final_dict['focal_length_xy_raw'],
                princpt_xy=final_dict['princpt_raw'],
                img_width=img_w,
                img_height=img_h,
            )

            final_dict['transl'] = transl_new
            final_dict['focal_length'] = torch.full(
                (transl_new.shape[0], 1), float(focal_ref_xy[0].item()), dtype=torch.float32
            )
            final_dict['focal_length_xy'] = focal_ref_xy[None].repeat(transl_new.shape[0], 1)

            cx_ref = img_w / 2.0
            cy_ref = img_h / 2.0
            final_dict['princpt'] = torch.tensor([[cx_ref, cy_ref]], dtype=torch.float32).repeat(transl_new.shape[0], 1)

            final_dict['retargeted_cam'] = torch.tensor([1], dtype=torch.uint8)

        pt_save_path = os.path.join(output_folder, f'{args.file_name}_params.pt')
        torch.save(final_dict, pt_save_path)
        print(f"Successfully saved parameters to: {pt_save_path}")
    else:
        print("No parameters collected (maybe no person detected).")


if __name__ == "__main__":
    main()
