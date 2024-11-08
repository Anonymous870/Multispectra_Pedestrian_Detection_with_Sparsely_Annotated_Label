from datetime import datetime
from typing import Dict
import config
import logging
import numpy as np
import os
import time
import copy
from collections import defaultdict
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.ops as ops

from datasets import KAISTPed
from inference import val_epoch_baseline, val_epoch, save_results
from model import SSD300, SSD300_3Way, MultiBoxLoss
from utils import utils
from utils.evaluation_script import evaluate

from train_utils import *
import cv2


torch.backends.cudnn.benchmark = False

import os
import shutil

utils.set_seed(seed=9)

def main():
    """Train and validate a model"""

    args = config.args
    train_conf = config.train
    epochs = train_conf.epochs
    phase = "Multispectral"
        

    # Initialize model or load checkpoint
    s_model, optimizer, optim_scheduler, s_epochs, \
    t_model,       _,                 _,        _ = create_teacher_student()

    # Move to default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_func = t_model.detect_objects_cuda
    s_model = s_model.to(device)
    t_model = t_model.to(device)
    s_model = nn.DataParallel(s_model)
    t_model = nn.DataParallel(t_model)
    t_model.eval()

    criterion = MultiBoxLoss(priors_cxcy=s_model.module.priors_cxcy).to(device)

    train_dataset = KAISTPed(args, condition='train')
    train_loader = DataLoader(train_dataset, batch_size=train_conf.batch_size, shuffle=True,
                              num_workers=config.dataset.workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)  # note that we're passing the collate function here

    test_dataset = KAISTPed(args, condition='test')
    test_batch_size = args["test"].eval_batch_size * torch.cuda.device_count()
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             num_workers=config.dataset.workers,
                             collate_fn=test_dataset.collate_fn,
                             pin_memory=True)
    # Set job directory
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    
    exp_name = ('_' + args.exp_name) if args.exp_name else '_'
    jobs_dir = os.path.join('jobs', args.exp_time + exp_name)
    os.makedirs(jobs_dir, exist_ok=True)
    args.jobs_dir = jobs_dir

    # Make logger
    logger = utils.make_logger(args)

    # Epochs
    kwargs = {'grad_clip': args['train'].grad_clip, 'print_freq': args['train'].print_freq}
    for epoch in range(s_epochs, epochs):
        # One epoch's training
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        train_loss = train_epoch(s_model=s_model,
                                 t_model=t_model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 logger=logger,
                                 epoch=epoch,
                                 inference_func=inference_func,
                                 **kwargs)

        optim_scheduler.step()

        # Save checkpoint
        utils.save_checkpoint(epoch, s_model.module, optimizer, train_loss, jobs_dir, "student")
        utils.save_checkpoint(epoch, t_model.module,      None,       None, jobs_dir, "teacher")
        
        if epoch >= 0:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')
            results = val_epoch(s_model, test_loader, config.test.input_size, inference_func, min_score=0.1)

            save_results(results, result_filename)
            
            evaluate(config.PATH.JSON_GT_FILE, result_filename, phase) 

def rvs_norm(boxes):
    x1 = boxes[:, 0] * 640
    y1 = boxes[:, 1] * 512
    x2 = boxes[:, 2] * 640
    y2 = boxes[:, 3] * 512
    return torch.stack((x1,y1,x2,y2), dim=1)

def box_area(boxes):
    """Calculate the area of the boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """Calculate intersection-over-union (IoU) of two sets of boxes."""
    area1 = box_area(rvs_norm(boxes1))
    area2 = box_area(rvs_norm(boxes2))

    # Find intersections
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) 
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) 

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1] 

    # Calculate IoU
    union = area1[:, None] + area2 - inter
    iou = inter / union
    
    # Replace NaN values with 0
    iou[union == 0] = 1
    return iou

def GT_box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def CVIU(image_vis, image_lwir, boxes, labels):
    batch_size = len(boxes)
    for ref_idx in range(batch_size):
        ref_boxes = boxes[ref_idx].clone()
        ref_labels = labels[ref_idx].clone()
 
        for batch_idx in range(batch_size):
            if batch_idx == ref_idx:
                continue

            current_boxes = boxes[batch_idx]
            current_labels = labels[batch_idx]
            iou = box_iou(ref_boxes, current_boxes.clone())
            non_overlap_mask = (iou <= 0.1).all(dim=0)
            non_overlap_indices = torch.nonzero(non_overlap_mask, as_tuple=False).squeeze(1)

            if non_overlap_indices.numel() > 0:
                non_overlap_index = non_overlap_indices[0]
                non_overlap_box = current_boxes[non_overlap_index]
                non_overlap_label = current_labels[non_overlap_index]

                x1, y1, x2, y2 = non_overlap_box
                x1, x2 = int(x1 * 640), int(x2 * 640)
                y1, y2 = int(y1 * 512), int(y2 * 512)
                
                cropped_image_vis = image_vis[batch_idx, :, y1:y2, x1:x2]
                cropped_image_lwir = image_lwir[batch_idx, :, y1:y2, x1:x2]
                
                image_vis[ref_idx, :, y1:y2, x1:x2] = cropped_image_vis
                image_lwir[ref_idx, :, y1:y2, x1:x2] = cropped_image_lwir

                ref_boxes = torch.cat([ref_boxes, non_overlap_box.unsqueeze(0)], dim=0)
                ref_labels = torch.cat([ref_labels, torch.tensor([non_overlap_label], device=ref_labels.device)], dim=0)

                break
    return image_vis, image_lwir, boxes, labels

def train_epoch(s_model: SSD300_3Way,
                t_model: SSD300_3Way,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                epoch: int,
                inference_func,
                **kwargs: Dict) -> float:
    device = next(s_model.parameters()).device
    s_model.train()  # training mode enables dropout
    number = 90
    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum

    start = time.time()
    if epoch == 0:
        with open (f"./Fully_added_case{number}.txt", "w") as f:
            f.write(f"Start the Record!\n\n")
            f.write(f"epoch: {epoch}\n")
    if epoch > 0:
        with open (f"./Fully_added_case{number}.txt", "a") as f:
            f.write(f"epoch: {epoch}\n")
    # Batches
    #print(f"data loader length: {len(dataloader)}")
    added_cnt = 0
    for batch_idx, (image_vis, image_lwir, image_vis2, image_lwir2, boxes, labels, boxes2, labels2, _, return_id, dst_folder) in enumerate(dataloader):
        data_time.update(time.time() - start)

        # Move to default device
        image_vis = image_vis.to(device)
        image_lwir = image_lwir.to(device)
        image_vis2 = image_vis2.to(device)
        image_lwir2 = image_lwir2.to(device)

        if len(boxes) == 0:
            print(f"len of boxes: {len(boxes)}")
            continue
        if len(boxes) < 6:
            print(f"len of boxes: {len(boxes)} is less than 6")
            continue

        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        boxes2 = [box.to(device) for box in boxes2]
        labels2 = [label.to(device) for label in labels2]

        #image_vis, image_lwir, boxes, labels = CVIU(image_vis, image_lwir, boxes, labels) #Robust Teacher implementation

        # Forward prop.
        predicted_locs_fusion, predicted_scores_fusion, \
        predicted_locs_vis, predicted_scores_vis, \
        predicted_locs_lwir, predicted_scores_lwir, \
        features_fusion, features_vis, features_lwir = s_model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        predicted_locs_fusion2, predicted_scores_fusion2, \
        predicted_locs_vis2, predicted_scores_vis2, \
        predicted_locs_lwir2, predicted_scores_lwir2, \
        features_fusion2, features_vis2, features_lwir2 = s_model(image_vis2, image_lwir2)  # (N, 8732, 4), (N, 8732, n_classes)
        
        if config.train.train_mode == "baseline": # original Loss
            loss, _, _, n_positives = criterion(predicted_locs_fusion, predicted_scores_fusion, boxes, labels)  # scalar
        elif config.train.train_mode == "3way": # 3way baseline
            loss_fusion, _, _, n_positives = criterion(predicted_locs_fusion, predicted_scores_fusion, boxes, labels)  # scalar
            loss_vis,    _, _,           _ = criterion(predicted_locs_vis,    predicted_scores_vis,    boxes, labels)  # scalar
            loss_lwir,   _, _,           _ = criterion(predicted_locs_lwir,   predicted_scores_lwir,   boxes, labels)  # scalar
            loss = loss_fusion + loss_vis + loss_lwir
        elif config.train.train_mode == "ours":
            with torch.no_grad():
                locs_fusion, classes_scores_fusion, locs_vis, classes_scores_vis, locs_lwir, classes_scores_lwir, _, _, _ = t_model(image_vis, image_lwir)  # (N, 8732, 4), (N, 8732, n_classes)
                locs_fusion2, classes_scores_fusion2, locs_vis2, classes_scores_vis2, locs_lwir2, classes_scores_lwir2, _, _, _ = t_model(image_vis2, image_lwir2)  # (N, 8732, 4), (N, 8732, n_classes)
                f_inference = inference_func(locs_fusion, classes_scores_fusion, min_score=0.1, max_overlap=0.425, top_k=200)
                v_inference = inference_func(locs_vis, classes_scores_vis, min_score=0.1, max_overlap=0.425, top_k=200)
                t_inference = inference_func(locs_lwir, classes_scores_lwir, min_score=0.1, max_overlap=0.425, top_k=200)

                f_inference2 = inference_func(locs_fusion2, classes_scores_fusion2, min_score=0.1, max_overlap=0.425, top_k=200)
                fusion_pseudo_boxes2, fusion_scores2, fusion_pseudo_labels2 = detect(f_inference2, len(boxes2), "PL")
                fusion_pseudo_boxes2 = [pseudo_box.to(device) for pseudo_box in fusion_pseudo_boxes2]
                #(f_inference,v_inference,t_inference, len(boxes), batch_idx)
            fusion_pseudo_boxes, fusion_scores, fusion_pseudo_labels = detect(f_inference, len(boxes), "PL")
            vis_pseudo_boxes, vis_scores, vis_pseudo_labels = detect(v_inference, len(boxes), "Test")
            lwir_pseudo_boxes, lwir_scores, lwir_pseudo_labels = detect(t_inference, len(boxes), "Test")

            
           
            fusion_pseudo_boxes = [pseudo_box.to(device) for pseudo_box in fusion_pseudo_boxes]
            vis_pseudo_boxes = [vis_pseudo_box.to(device) for vis_pseudo_box in vis_pseudo_boxes]
            lwir_pseudo_boxes = [lwir_pseudo_box.to(device) for lwir_pseudo_box in lwir_pseudo_boxes]

            
            make_GT = config.args.Make_GT
            keep = config.args.keep
            #exam = config.args.exam

            cos_w_fusion, cont_loss_fusion, comb_boxes_fusion, comb_labels_fusion = GAP_similarity_based_method(features_fusion, fusion_pseudo_boxes, fusion_pseudo_labels, boxes, labels)
            cos_w_vis,    cont_loss_vis,    comb_boxes_vis,    comb_labels_vis    = GAP_similarity_based_method(features_vis,    fusion_pseudo_boxes, fusion_pseudo_labels, boxes, labels)
            cos_w_lwir,   cont_loss_lwir,   comb_boxes_lwir,   comb_labels_lwir = GAP_similarity_based_method(features_lwir,   fusion_pseudo_boxes, fusion_pseudo_labels, boxes, labels)
            
            with torch.no_grad():
               cos_w_fusion2, cont_loss_fusion2, comb_boxes_fusion2, comb_labels_fusion2 = GAP_similarity_based_method(features_fusion2, fusion_pseudo_boxes2, fusion_pseudo_labels2, boxes2, labels2)
            save_format_path = f"../data/KAIST_gt_samples_{config.args.MP}/" + "%s/%s.jpg"
            if keep:
                for i in range(len(fusion_pseudo_boxes2)):
                    if len(fusion_scores2[i]) !=0:
                        for j in range(len(fusion_scores2[i])):
                            iou_with_original_box = list()
                            if fusion_scores2[i][j] > float(f"0.{number}") and cos_w_fusion2 > 0.9 and fusion_pseudo_labels2[i][j] == 3:
                        
                                x, y, w, h = fusion_pseudo_boxes2[i][j]
                                x, y, w, h = int(x*640) , int(y*512), int(w*640), int(h*512)
                                w, h = w-x, h-y
                                

                                x2 = x + w
                                y2 = y + h
                                save_path_vis = os.path.join(dst_folder[i], return_id[i][0], return_id[i][1], "visible", f"{return_id[i][2]}.txt")
                                save_path_lwir = os.path.join(dst_folder[i], return_id[i][0], return_id[i][1], "lwir", f"{return_id[i][2]}.txt")
                    
                                box = unnormalize_boxes(copy.deepcopy(boxes2[i]))
                                
                                label = labels2[i]
                                for lines, l in zip(box,label):
                                    if l == 3:
                                        lines = [int(b) for b in lines]
                                        if len(lines) > 0:
                                            original_box = lines
                                            iou_with_original_box.append(GT_box_iou(original_box, [x, y, w, h]))
                                
                                if len(iou_with_original_box) > 0:
                                        if all(iou < 0.425 for iou in iou_with_original_box):
                                            with open(save_path_vis, "a") as f:
                                                f.write(f"person {x} {y} {w} {h} 0 0 0 0 0 0 0\n")
                                                f.flush()
                                            rgb_sample = image_vis2[y:y2, x:x2, :]
                                            
                                            base_filename = f"{return_id[0]}_{return_id[1]}_{return_id[2]}_"
                                            existing_files = [f for f in os.listdir("path_to_save_folder") if f.startswith(base_filename)]
                                            

                                            existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                            

                                            ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                            

                                            save_path = save_format_path % ("visible", f"{return_id[0]}_{return_id[1]}_{return_id[2]}_{ridx}")
                                            cv2.imwrite(save_path, rgb_sample)
                                            with open(save_path_lwir, "a") as f:
                                                f.write(f"person {x} {y} {w} {h} 0 0 0 0 0 0 0\n")
                                                f.flush()
                                            lwir_sample = image_lwir2[y:y2, x:x2, :]
                                            existing_files = [f for f in os.listdir("path_to_save_folder") if f.startswith(base_filename)]
                                            

                                            existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                            

                                            ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                            save_path = save_format_path % ("lwir", f"{return_id[0]}_{return_id[1]}_{return_id[2]}_{ridx}")
                                            cv2.imwrite(save_path, lwir_sample)

                                            added_cnt += 1
                                            with open (f"./Fully_added_case{number}.txt", "a") as f:
                                                f.write(f"save_path_lwir: {save_path_lwir} / predicted score : {fusion_scores2[i][j]} / person {x} {y} {w} {h} 0 0 0 0 0 0 0 / added_cnt: {added_cnt}\n")
                                                f.flush()
                                elif len(iou_with_original_box) == 0 and len(box) == 0:
                                    with open(save_path_vis, "a") as f:
                                        f.write(f"\nperson {x} {y} {w} {h} 0 0 0 0 0 0 0")
                                        f.flush()
                                    rgb_sample = image_vis2[y:y2, x:x2, :]
                                            
                                    base_filename = f"{return_id[0]}_{return_id[1]}_{return_id[2]}_"
                                    existing_files = [f for f in os.listdir("path_to_save_folder") if f.startswith(base_filename)]
                                    

                                    existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                    

                                    ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                    

                                    save_path = save_format_path % ("visible", f"{return_id[0]}_{return_id[1]}_{return_id[2]}_{ridx}")
                                    cv2.imwrite(save_path, rgb_sample)
                                    
                                    with open(save_path_lwir, "a") as f:
                                        f.write(f"\nperson {x} {y} {w} {h} 0 0 0 0 0 0 0")
                                        f.flush()
                                    lwir_sample = image_vis2[y:y2, x:x2, :]
                                            
                                    base_filename = f"{return_id[0]}_{return_id[1]}_{return_id[2]}_"
                                    existing_files = [f for f in os.listdir("path_to_save_folder") if f.startswith(base_filename)]
                                    
                                    existing_ridxs = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.split('_')[-1].split('.')[0].isdigit()]
                                    

                                    ridx = max(existing_ridxs) + 1 if existing_ridxs else 0
                                    

                                    save_path = save_format_path % ("visible", f"{return_id[0]}_{return_id[1]}_{return_id[2]}_{ridx}")
                                    cv2.imwrite(save_path, rgb_sample)
                                    added_cnt += 1
                                    with open (f"./Fully_added_case{number}.txt", "a") as f:
                                        f.write(f"save_path_lwir: {save_path_lwir} / predicted score : {fusion_scores2[i][j]} / person {x} {y} {w} {h} 0 0 0 0 0 0 0 / added_cnt: {added_cnt}\n")
                                        f.flush()

           
            loss_fusion, _, _, n_positives = criterion(predicted_locs_fusion, predicted_scores_fusion, comb_boxes_fusion, comb_labels_fusion)
            loss_vis,    _, _,           _ = criterion(predicted_locs_vis,    predicted_scores_vis,    comb_boxes_vis,    comb_labels_vis)
            loss_lwir,   _, _,           _ = criterion(predicted_locs_lwir,   predicted_scores_lwir,   comb_boxes_lwir,   comb_labels_lwir)

            loss = cos_w_fusion * loss_fusion + cos_w_vis * loss_vis + cos_w_lwir * loss_lwir  + cont_loss_fusion + cont_loss_vis + cont_loss_lwir
            if torch.isnan(loss).any(): continue
        else: raise "Incorrect train_mode type..."
        

        optimizer.zero_grad()
        loss.backward()

        if kwargs.get('grad_clip', None):
            utils.clip_gradient(optimizer, kwargs['grad_clip'])


        optimizer.step()

        losses_sum.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        if batch_idx % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'num of Positive {Positive}\t'.format(batch_idx, len(dataloader),
                                                              batch_time=batch_time,
                                                              data_time=data_time,
                                                              loss=losses_sum,
                                                              Positive=n_positives))
    return losses_sum.avg

if __name__ == '__main__':
    main()
