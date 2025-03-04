import json
from ast import arg
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module
from sklearn.model_selection import KFold
from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D,correct_dims
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
import torch.nn.functional as F
import SimpleITK
from torch.utils.data import Dataset
from pathlib import Path
import SimpleITK as sitk


class MyDC(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        intersection = (y_pred[:, 1:2] * y_truth[:, 1:2]).sum() + (y_pred[:, 2:] * y_truth[:, 2:]).sum()
        union = y_pred[:, 1:2].sum() + y_pred[:, 2:].sum() + y_truth[:, 1:2].sum() + y_truth[:, 2:].sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_score.requires_grad_(True)
        return 1 - dice_score  # , dice1, dice2

class MyDC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        intersection1 = (y_pred[:, 1:2] * y_truth[:, 1:2]).sum()
        union1 = y_pred[:, 1:2].sum() + y_truth[:, 1:2].sum()
        intersection2 = (y_pred[:, 2:] * y_truth[:, 2:]).sum()
        union2 = y_pred[:, 2:].sum()+ y_truth[:, 2:].sum()
        dice_score1 = (2. * intersection1 + self.smooth) / (union1 + self.smooth)
        dice_score2 = (2. * intersection2 + self.smooth) / (union2 + self.smooth)
        dice_loss = (1-dice_score1 + 1-dice_score2)/2
        return dice_loss


class DCloss(nn.Module):
    def __init__(self):
        super(DCloss, self).__init__()
        self.dc = MyDC()

    def forward(self, net_output, target):
        low_res_logits = net_output['masks']
        target = F.one_hot(target.squeeze(1).long(), 3)

        low_res_logits_t = torch.flatten(low_res_logits.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        target_t = torch.flatten(target, start_dim=0, end_dim=2)

        loss_dice = self.dc(F.one_hot(torch.argmax(low_res_logits_t, dim=1), 3), target_t)
        return loss_dice


class EvalDice(nn.Module):
    def __init__(self):
        super(EvalDice, self).__init__()
        self.dc = MyDC2()

    def forward(self, net_output, target):
        low_res_logits = net_output['masks']
        target = F.one_hot(target.squeeze(1).long(), 3)

        low_res_logits_t = torch.flatten(low_res_logits.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        target_t = torch.flatten(target, start_dim=0, end_dim=2)

        loss_dice = self.dc(F.one_hot(torch.argmax(low_res_logits_t, dim=1), 3), target_t)
        return loss_dice

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, dtype='uint32')
    factor = originSize / newSize
    newSpacing = originSpacing * factor

    resampler.SetReferenceImage(itkimage)  # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    itkimgResampled.SetOrigin(itkimage.GetOrigin())
    itkimgResampled.SetSpacing(itkimage.GetSpacing())
    itkimgResampled.SetDirection(itkimage.GetDirection())
    return itkimgResampled


class Fetal_dataset(Dataset):
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [SimpleITK.GetArrayFromImage(resize_image_itk(SimpleITK.ReadImage(str(i)), (512, 512, 3))) for i
                  in list_dir[0]]
        labels = [SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(i))) for i in
                  list_dir[1]]
        self.images = np.array(images)
        self.labels = np.array(labels)
        print("-" * 20)
        print(self.images.shape)
        print(self.labels.shape)
        print("-" * 20)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, mask = correct_dims(self.images[idx].transpose((1, 2, 0)), self.labels[idx])
        sample = {}
        if self.transform:
            image, mask, low_mask = self.transform(image, mask)

        sample['image'] = image
        sample['low_res_label'] = low_mask.unsqueeze(0)
        sample['label'] = mask.unsqueeze(0)

        return sample


def main(model_fix):
    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 2023  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================

    #  ========================================================================= begin to train the model ============================================================================
    kfold = 5
    tf_val = JointTransform2D(img_size=256, low_img_size=128, ori_size=256,
                              crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True)
    root_path = Path('./testdataset')
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])

    with open("./test.txt", "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    print(len(test_index))
    db_test = Fetal_dataset(transform=tf_val,
                            list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)]))
    testloader = DataLoader(db_test, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    for fold in range(kfold):
        ckpt_path = f"./checkpoints/{model_fix}/fold{fold}"

        ckpt_file = os.listdir(ckpt_path)[-1]  # pretrained weight
        ckpt = os.path.join(ckpt_path,ckpt_file)

        parser = argparse.ArgumentParser(description='Networks')
        parser.add_argument('--modelname', default='AoP-SAM', type=str)
        parser.add_argument('-encoder_input_size', type=int, default=512)
        parser.add_argument('-low_image_size', type=int, default=128)
        parser.add_argument('--task', default='PSFH', help='task or dataset name')
        parser.add_argument('--vit_name', type=str, default='vit_h')
        parser.add_argument('--sam_ckpt', type=str, default=ckpt)
        parser.add_argument('--batch_size', type=int, default=8,
                            help='batch_size per gpu')  # SAMed is 12 bs with 2n_gpu and lr is 0.005
        parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
        parser.add_argument('--base_lr', type=float, default=0.0001,
                            help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')  # 0.0006
        parser.add_argument('--warmup', type=bool, default=True,
                            help='If activated, warp up the learning from a lower lr to the base_lr')
        parser.add_argument('--warmup_period', type=int, default=100,
                            help='Warp up iterations, only valid whrn warmup is activated')
        parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

        args = parser.parse_args()
        opt = get_config(args.task)
        device = torch.device(opt.device)
        opt.batch_size = args.batch_size * args.n_gpu

        tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                    ori_size=opt.img_size, crop=opt.crop, p_flip=0.5, p_rota=0.5, p_scale=0.0,
                                    p_gaussn=0.0,
                                    p_contr=0.0, p_gama=0.0, p_distor=0.0, color_jitter_params=None,
                                    long_mask=True)  # image reprocessing
        tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                  ori_size=opt.img_size,
                                  crop=opt.crop, p_flip=0.0, color_jitter_params=None, long_mask=True)
        # =======================  get model and check it ========================== #
        iter_num = 0
        model = get_model(ckpt=ckpt, args=args)
        model.load_state_dict(torch.load(ckpt),strict=False)
        model.to(device)

        # =======================  optimizer criterion file_fold preparation ========================== #
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr,
                                      betas=(0.9, 0.999), weight_decay=0.1)
        criterion = get_criterion(modelname=args.modelname, opt=opt)

        os.makedirs(f"./checkpoints/{model_fix}", exist_ok=True)
        os.makedirs(f"./result/{model_fix}", exist_ok=True)
        os.makedirs(f"./checkpoints/{model_fix}/fold{fold}", exist_ok=True)
        os.makedirs(f"./result/{model_fix}/fold{fold}", exist_ok=True)

        train_split = f"train_fold_{fold}"
        val_split = f"val_fold_{fold}"

        # =======================  dataloader ========================== #
        trainfold = ImageToImage2D(opt.data_path, train_split, tf_train, img_size=args.encoder_input_size)
        valfold = ImageToImage2D(opt.data_path, val_split, tf_val, img_size=args.encoder_input_size)
        trainloader = DataLoader(trainfold, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
        valloader = DataLoader(valfold, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        print(f"Total epoch: {opt.epochs}")

        model.eval()
        val_loss_dice = []
        evaldc = EvalDice()
        for batch_idx, (datapack) in enumerate(valloader):
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            label = datapack['label'].to(dtype=torch.float32, device=opt.device)
            with torch.no_grad():
                pred = model(imgs, None, None, None)

            val_loss = evaldc(pred, label)
            val_loss_dice.append(val_loss.detach().cpu().numpy())
        best_score = np.mean(val_loss_dice)
        print(f"First Test on Validation Set: {best_score}")
        

        test_loss_dice = []
        for batch_idx, (datapack) in enumerate(testloader):
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            label = datapack['label'].to(dtype=torch.float32, device=opt.device)
            with torch.no_grad():
                pred = model(imgs, None, None, None)

            test_loss = evaldc(pred, label)
            test_loss_dice.append(test_loss.detach().cpu().numpy())
        test_score = np.mean(test_loss_dice)
        print(f"First Test on Validation Set: {test_score}")

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        for epoch in range(opt.epochs):
            #  --------------------------------------------------------- training ---------------------------------------------------------
            model.train()
            train_losses = 0
            for batch_idx, (datapack) in enumerate(trainloader):
                imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
                masks = datapack['label'].to(dtype=torch.float32, device=opt.device)
                low_masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)  # 3-value mask
                # -------------------------------------------------------- forward --------------------------------------------------------
                prompt_masks = low_masks.argmax(dim=1, keepdim=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    pred = model(imgs, None, None, None)
                    print(f"Epoch[{epoch + 1}/{opt.epochs}] | Batch {batch_idx}: ", end="")
                    train_loss = criterion(pred, masks)
                # -------------------------------------------------------- backward -------------------------------------------------------
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                train_losses += train_loss.item()
                # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
                if args.warmup and iter_num < args.warmup_period:
                    lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else:
                    if args.warmup:
                        shift_iter = iter_num - args.warmup_period
                        assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                        lr_ = args.base_lr * (
                                1.0 - shift_iter / (
                                3200 * opt.epochs / opt.batch_size)) ** 0.9  # learning rate adjustment depends on the max iterations
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_

                iter_num = iter_num + 1

            #  --------------------------------------------------------- evaluation ----------------------------------------------------------
            if (epoch >= 15):
                model.eval()
                val_loss_dice = []
                evaldc = EvalDice()
                for batch_idx, (datapack) in enumerate(valloader):
                    imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
                    label = datapack['label'].to(dtype=torch.float32, device=opt.device)
                    with torch.no_grad():
                        pred = model(imgs, None, None, None)
                    val_loss = evaldc(pred, label)
                    val_loss_dice.append(val_loss.detach().cpu().numpy())
                val_loss_dice_mean = np.mean(val_loss_dice)

                test_loss_dice = []
                for batch_idx, (datapack) in enumerate(testloader):
                    imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
                    label = datapack['label'].to(dtype=torch.float32, device=opt.device)
                    with torch.no_grad():
                        pred = model(imgs, None, None, None)

                    test_loss = evaldc(pred, label)
                    test_loss_dice.append(test_loss.detach().cpu().numpy())
                test_loss_dice_mean = np.mean(test_loss_dice)

                if val_loss_dice_mean <= best_score:
                    best_score = val_loss_dice_mean
                    save_path = f'./checkpoints/{model_fix}/fold{fold}/epoch{epoch + 1}_val_{best_score:.4f}_test_{test_loss_dice_mean:.4f}'
                    torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    model_fix = "AoP-SAM"
    main(model_fix)
