from tkinter import image_names
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation, visual_segmentation_binary, visual_segmentation_sets, \
    visual_segmentation_sets_with_pt
from einops import rearrange
from utils.generate_prompts import get_click_prompt
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import SimpleITK
from tqdm import tqdm


class Evaluation:
    def __init__(self, a, b):
        self.results = []
        self.pre_image = SimpleITK.GetImageFromArray(a)
        self.truth_image = SimpleITK.GetImageFromArray(b)

    def evaluation(self, pred: SimpleITK.Image, label: SimpleITK.Image):
        result = dict()
        pred_aop = self.cal_aop(pred)
        label_aop = self.cal_aop(label)
        aop = float(abs(pred_aop - label_aop))
        result['aop'] = aop
        result['aop_gt'] = label_aop
        result['aop_truth'] = pred_aop
        # 计算耻骨指标
        pred_data_up = SimpleITK.GetArrayFromImage(pred)
        pred_data_up[pred_data_up == 2] = 0
        pred_up = SimpleITK.GetImageFromArray(pred_data_up)

        label_data_up = SimpleITK.GetArrayFromImage(label)
        label_data_up[label_data_up == 2] = 0
        label_up = SimpleITK.GetImageFromArray(label_data_up)

        if (pred_data_up == 0).all():
            result['asd_up'] = 100.0
            result['dice_up'] = 0.0
            result['hd_up'] = 100.0
        else:
            result['asd_up'] = float(self.cal_asd(pred_up, label_up))
            result['dice_up'] = float(self.cal_dsc(pred_up, label_up))
            result['hd_up'] = float(self.cal_hd(pred_up, label_up))

        # 计算胎头指标
        pred_data_head = SimpleITK.GetArrayFromImage(pred)
        pred_data_head[pred_data_head == 1] = 0
        pred_data_head[pred_data_head == 2] = 1
        pred_head = SimpleITK.GetImageFromArray(pred_data_head)

        label_data_head = SimpleITK.GetArrayFromImage(label)
        label_data_head[label_data_head == 1] = 0
        label_data_head[label_data_head == 2] = 1
        label_head = SimpleITK.GetImageFromArray(label_data_head)

        if (pred_data_head == 0).all():
            result['asd_low'] = 100.0
            result['dice_low'] = 0.0
            result['hd_low'] = 100.0
        else:
            result['asd_low'] = float(self.cal_asd(pred_head, label_head))
            result['dice_low'] = float(self.cal_dsc(pred_head, label_head))
            result['hd_low'] = float(self.cal_hd(pred_head, label_head))

        # 计算总体指标
        pred_data_all = SimpleITK.GetArrayFromImage(pred)
        pred_data_all[pred_data_all == 2] = 1
        pred_all = SimpleITK.GetImageFromArray(pred_data_all)

        label_data_all = SimpleITK.GetArrayFromImage(label)
        label_data_all[label_data_all == 2] = 1
        label_all = SimpleITK.GetImageFromArray(label_data_all)
        if (pred_data_all == 0).all():
            result['asd_all'] = 100.0
            result['dice_all'] = 0.0
            result['hd_all'] = 100.0
        else:
            result['asd_all'] = float(self.cal_asd(pred_all, label_all))
            result['dice_all'] = float(self.cal_dsc(pred_all, label_all))
            result['hd_all'] = float(self.cal_hd(pred_all, label_all))

        return result

    def process(self):
        pre_image = self.pre_image
        truth_image = self.truth_image

        result = self.evaluation(pre_image, truth_image)
        return result

    def cal_asd(self, a, b):
        filter1 = SimpleITK.SignedMaurerDistanceMapImageFilter()  # 于计算二值图像中像素到最近非零像素距离的算法
        filter1.SetUseImageSpacing(True)  # 计算像素距离时要考虑像素之间的间距
        filter1.SetSquaredDistance(False)  # 计算距离时不要对距离进行平方处理
        a_dist = filter1.Execute(a)
        a_dist = SimpleITK.GetArrayFromImage(a_dist)
        a_dist = np.abs(a_dist)
        a_edge = np.zeros(a_dist.shape, a_dist.dtype)
        a_edge[a_dist == 0] = 1
        a_num = np.sum(a_edge)

        filter2 = SimpleITK.SignedMaurerDistanceMapImageFilter()
        filter2.SetUseImageSpacing(True)
        filter2.SetSquaredDistance(False)
        b_dist = filter2.Execute(b)

        b_dist = SimpleITK.GetArrayFromImage(b_dist)
        b_dist = np.abs(b_dist)
        b_edge = np.zeros(b_dist.shape, b_dist.dtype)
        b_edge[b_dist == 0] = 1
        b_num = np.sum(b_edge)

        a_dist[b_edge == 0] = 0.0
        b_dist[a_edge == 0] = 0.0

        asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

        return asd

    def cal_dsc(self, pd, gt):
        pd = SimpleITK.GetArrayFromImage(pd).astype(np.uint8)
        gt = SimpleITK.GetArrayFromImage(gt).astype(np.uint8)
        y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
        return y

    def cal_hd(self, a, b):
        a = SimpleITK.Cast(SimpleITK.RescaleIntensity(a), SimpleITK.sitkUInt8)
        b = SimpleITK.Cast(SimpleITK.RescaleIntensity(b), SimpleITK.sitkUInt8)
        try:
            filter1 = SimpleITK.HausdorffDistanceImageFilter()
            filter1.Execute(a, b)
            hd = filter1.GetHausdorffDistance()
        except:
            hd = 1e3
        return hd

    def cal_score(self, result):
        m = len(result)
        dice_all_score = 0.
        dice_low_score = 0.
        dice_up_score = 0.
        aop_score = 0.
        hd_up_score = 0.
        hd_all_score = 0.
        hd_low_score = 0.
        asd_all_score = 0.
        asd_low_score = 0.
        asd_up_score = 0.
        for i in range(m):
            # dice
            dice_all_score += float(result[i].get("dice_all"))
            dice_up_score += float(result[i].get("dice_up"))
            dice_low_score += float(result[i].get("dice_low"))
            # asa
            asd_all_score += float(result[i].get("asd_all"))
            asd_low_score += float(result[i].get("asd_low"))
            asd_up_score += float(result[i].get("asd_up"))
            # hd
            hd_all_score += float(result[i].get("hd_all"))
            hd_up_score += float(result[i].get("hd_up"))
            hd_low_score += float(result[i].get("hd_low"))

        dice_score = (dice_all_score + dice_up_score + dice_low_score) / (3 * m)
        hd_score = (hd_all_score + hd_up_score + hd_low_score) / (3 * m)
        asd_score = (asd_all_score + asd_up_score + asd_low_score) / (3 * m)

        score = 0.75 + 0.25 * round(dice_score, 6) - 0.25 * round(hd_score / 100.0, 10) - 0.25 * round(
            asd_score / 100.0, 10)

        aggregates = dict()
        aggregates['dice_up'] = dice_up_score / m
        aggregates['dice_low'] = dice_low_score / m
        aggregates['dice_all'] = dice_all_score / m
        aggregates['hd_up'] = hd_up_score / m
        aggregates['hd_low'] = hd_low_score / m
        aggregates['hd_all'] = hd_all_score / m
        aggregates['asd_up'] = asd_up_score / m
        aggregates['asd_low'] = asd_low_score / m
        aggregates['asd_all'] = asd_all_score / m

        return score, aggregates

    def cal_aop(self, pred):
        aop = 0.0
        ellipse = None
        ellipse2 = None
        pred_data = SimpleITK.GetArrayFromImage(pred)
        aop_pred = np.array(self.onehot_to_mask(pred_data)).astype(np.uint8)
        contours, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 1], 1), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 2], 1), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        maxindex1 = 0
        maxindex2 = 0
        max1 = 0
        max2 = 0
        flag1 = 0
        flag2 = 0
        for j in range(len(contours)):
            if contours[j].shape[0] > max1:
                maxindex1 = j
                max1 = contours[j].shape[0]
            if j == len(contours) - 1:
                approxCurve = cv2.approxPolyDP(contours[maxindex1], 1, closed=True)
                if approxCurve.shape[0] > 5:
                    ellipse = cv2.fitEllipse(approxCurve)
                flag1 = 1
        for k in range(len(contours2)):
            if contours2[k].shape[0] > max2:
                maxindex2 = k
                max2 = contours2[k].shape[0]
            if k == len(contours2) - 1:
                approxCurve2 = cv2.approxPolyDP(contours2[maxindex2], 1, closed=True)
                if approxCurve2.shape[0] > 5:
                    ellipse2 = cv2.fitEllipse(approxCurve2)
                flag2 = 1
        if flag1 == 1 and flag2 == 1 and ellipse2 != None and ellipse != None:
            aop = drawline_AOD(ellipse2, ellipse)
        return aop

    def onehot_to_mask(self, mask):
        ret = np.zeros([3, 256, 256])
        tmp = mask.copy()
        tmp[tmp == 1] = 255
        tmp[tmp == 2] = 0
        ret[1] = tmp
        tmp = mask.copy()
        tmp[tmp == 2] = 255
        tmp[tmp == 1] = 0
        ret[2] = tmp
        b = ret[0]
        r = ret[1]
        g = ret[2]
        ret = cv2.merge([b, r, g])
        mask = ret.transpose([0, 1, 2])
        return mask


def drawline_AOD(element_, element_1):
    import math
    element = (element_[0], (element_[1][1], element_[1][0]), element_[2] - 90)
    element1 = (element_1[0], (element_1[1][1], element_1[1][0]), element_1[2] - 90)

    [d11, d12] = [element1[0][0] - element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] - element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    [d21, d22] = [element1[0][0] + element1[1][0] / 2 * math.cos(element1[2] * 0.01745),
                  element1[0][1] + element1[1][0] / 2 * math.sin(element1[2] * 0.01745)]
    # cv2.line(background, (round(d11), round(d12)), (round(d21), round(d22)), (255, 255, 255), 2)
    a = element[1][0] / 2
    b = element[1][1] / 2
    angel = 2 * math.pi * element[2] / 360
    dp21 = d21 - element[0][0]
    dp22 = d22 - element[0][1]

    dp2 = np.array([[dp21], [dp22]])
    Transmat1 = np.array([[math.cos(-angel), -math.sin(-angel)],
                          [math.sin(-angel), math.cos(-angel)]])
    Transmat2 = np.array([[math.cos(angel), -math.sin(angel)],
                          [math.sin(angel), math.cos(angel)]])
    dpz2 = Transmat1 @ dp2
    dpz21 = dpz2[0][0]
    dpz22 = dpz2[1][0]
    if dpz21 ** 2 - a ** 2 == 0:
        dpz21 += 1
    if (b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2) >= 0:
        xielv_aod = (dpz21 * dpz22 - math.sqrt(b ** 2 * dpz21 ** 2 + a ** 2 * dpz22 ** 2 - a ** 2 * b ** 2)) / (
                dpz21 ** 2 - a ** 2)
    else:
        xielv_aod = 0
    bias_aod = dpz22 - xielv_aod * dpz21
    qiepz1 = (-2 * xielv_aod * bias_aod / b ** 2) / (2 * (1 / a ** 2 + xielv_aod ** 2 / b ** 2))
    qiepz2 = qiepz1 * xielv_aod + bias_aod
    qiepz = np.array([[qiepz1], [qiepz2]])
    qiep = list(Transmat2 @ qiepz)
    qie1 = qiep[0][0] + element[0][0]
    qie2 = qiep[1][0] + element[0][1]

    ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)
    ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)
    ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)

    aod = math.acos((ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / (2 * ld1d3 * ld3x4)) / math.pi * 180  ##余弦定理
    # cv2.line(background, (round(d21), round(d22)), (int(qie1), int(qie2)), (255, 255, 255), 2)
    return aod


def myeval(valloader, model, criterion, opt, args, is_val=False, has_mask_prompt=True, subnet=None):
    model.eval()

    if has_mask_prompt:
        print("Using mask prompt")
    else:
        print("No prompt")

    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    asds = np.zeros((max_slice_number, opt.classes))
    aops = np.zeros((max_slice_number, opt.classes))
    dices_up = np.zeros((max_slice_number, opt.classes))
    dices_low = np.zeros((max_slice_number, opt.classes))
    hds_up = np.zeros((max_slice_number, opt.classes))
    hds_low = np.zeros((max_slice_number, opt.classes))
    asds_up = np.zeros((max_slice_number, opt.classes))
    asds_low = np.zeros((max_slice_number, opt.classes))
    aop_records = []
    eval_number = 0
    sum_time = 0
    if has_mask_prompt:
        subnet.eval()
    for batch_idx, (datapack) in tqdm(enumerate(valloader)):
        imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
        label = datapack['label'].to(dtype=torch.float32, device=opt.device)  # (B,1,256,256)
        prompt_mask = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)

        with torch.no_grad():
            start_time = time.time()
            if has_mask_prompt:
                subnet_prompt_masks = subnet(imgs)
                subnet_prompt_masks = F.interpolate(subnet_prompt_masks, size=(128, 128), mode='bilinear').argmax(dim=1,
                                                                                                                  keepdim=True)
                pred = model(imgs, None, None, subnet_prompt_masks)  # (B,3,256,256)
            else:
                pred = model(imgs, None, None, None)
            sum_time = sum_time + (time.time() - start_time)

        val_loss = criterion(pred, label)
        val_losses += val_loss.item()

        gt = F.one_hot(label.squeeze(1).long(), 3).detach().cpu().numpy()  # one-hot
        gt = gt.argmax(axis=-1)  # (B,256,256) 3-value pics
        predict_masks = pred['masks']

        if is_val:
            pred_origin = pred['masks']
            pred_origin = pred_origin.detach().cpu().numpy()
            pred_origin = pred_origin.argmax(axis=1)

        pred = predict_masks.detach().cpu().numpy()  # (B,3,256,256)
        pred = pred.argmax(axis=1)  # 3-value pics  (B,256,256)

        b, h, w = pred.shape

        for j in range(0, b):
            result = Evaluation(pred[j], gt[j]).process()

            if is_val:
                pred[j][np.where(pred[j] == 1)] = 127
                pred[j][np.where(pred[j] == 2)] = 255
                pred_origin[j][np.where(pred_origin[j] == 1)] = 127
                pred_origin[j][np.where(pred_origin[j] == 2)] = 255
                cv2.imwrite(f"./validation/PHASE_2/no_prompt/{eval_number + j}_low.png", pred[j])
                cv2.imwrite(f"./validation/PHASE_2/no_prompt/{eval_number + j}.png", pred_origin[j])

            dices[eval_number + j, 1] += result['dice_all']
            hds[eval_number + j, 1] += result['hd_all']
            asds[eval_number + j, 1] += result['asd_all']
            dices_up[eval_number + j, 1] += result['dice_up']
            hds_up[eval_number + j, 1] += result['hd_up']
            asds_up[eval_number + j, 1] += result['asd_up']
            dices_low[eval_number + j, 1] += result['dice_low']
            hds_low[eval_number + j, 1] += result['hd_low']
            asds_low[eval_number + j, 1] += result['asd_low']
            aops[eval_number + j, 1] += result['aop']
            aop_records.append([result['aop_gt'], result['aop_truth']])

        eval_number = eval_number + b

    dices = dices[:eval_number, 1:2]  # origin dices = dices[:eval_number, :]
    hds = hds[:eval_number, 1:2]
    asds = asds[:eval_number, 1:2]  # add asd
    dices_up = dices_up[:eval_number, 1:2]
    hds_up = hds_up[:eval_number, 1:2]
    asds_up = asds_up[:eval_number, 1:2]
    dices_low = dices_low[:eval_number, 1:2]
    hds_low = hds_low[:eval_number, 1:2]
    asds_low = asds_low[:eval_number, 1:2]
    aops = aops[:eval_number, 1:2]

    val_losses = val_losses / (batch_idx + 1)

    mean_asd = np.mean(asds[:])
    mean_dice = np.mean(dices[:])
    mean_hdis = np.mean(hds[:])
    mean_asd_up = np.mean(asds_up[:])
    mean_dice_up = np.mean(dices_up[:])
    mean_hdis_up = np.mean(hds_up[:])
    mean_asd_low = np.mean(asds_low[:])
    mean_dice_low = np.mean(dices_low[:])
    mean_hdis_low = np.mean(hds_low[:])
    mean_aop = np.mean(aops[:])
    print("---Model Inference speed", sum_time / eval_number)
    score = 0.25 * (mean_dice + mean_dice_up + mean_dice_low) / 3 + 0.125 * (
            (3 - mean_hdis_low / 100 - mean_hdis_up / 100 - mean_hdis / 100) / 3 + (
            3 - mean_asd / 100 - mean_asd_low / 100 - mean_asd_up / 100) / 3) + 0.5 * (1 - mean_aop / 180)

    return {"score": score, "loss": val_losses, "dice": mean_dice, "asd": mean_asd, "hd": mean_hdis, "aop": mean_aop,
            "dice_up": mean_dice_up,
            "asd_up": mean_asd_up, "hd_up": mean_hdis_up, "dice_low": mean_dice_low, "asd_low": mean_asd_low,
            "hd_low": mean_hdis_low, "aop_records": aop_records}


def get_eval(valloader, model, criterion, opt, args, is_val=False, has_mask_prompt=True, subnet=None):
    if opt.eval_mode == "slice":
        return myeval(valloader, model, criterion, opt, args, is_val, has_mask_prompt, subnet=subnet)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)
