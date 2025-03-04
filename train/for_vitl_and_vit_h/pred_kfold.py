import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
import random
from utils.config import get_config
from importlib import import_module
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D, correct_dims
import torch.nn.functional as F
from torch.utils.data import Dataset
import SimpleITK
from pathlib import Path
import cv2
from tqdm import tqdm
import SimpleITK as sitk


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


def main(ckpt, fix, fold):
    #  ============================================================================= parameters setting ====================================================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='AoP-SAM', type=str)
    parser.add_argument('-encoder_input_size', type=int, default=512)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--task', default='PSFH', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_h')
    parser.add_argument('--sam_ckpt', type=str, default=ckpt)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--warmup', type=bool, default=True)
    parser.add_argument('--warmup_period', type=int, default=100)
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task)
    device = torch.device(opt.device)

    seed_value = 2023  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    opt.batch_size = args.batch_size * args.n_gpu
    os.makedirs("./gt", exist_ok=True)
    os.makedirs("./pred", exist_ok=True)
    os.makedirs(f"./gt/{fix}", exist_ok=True)
    os.makedirs(f"./pred/{fix}", exist_ok=True)
    os.makedirs(f"./gt/{fix}/{fold}", exist_ok=True)
    os.makedirs(f"./pred/{fix}/{fold}", exist_ok=True)

    # ============================ dataloader ===========================#
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0.0, color_jitter_params=None, long_mask=True)
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

    # ============================= load model =======================#
    model = get_model(args=args)
    model.load_state_dict(torch.load(ckpt))
    print(f"VIT EMBEDDING: {model.image_encoder.embed_dim}")
    model.to(device)
    # ============================ pred =============================#
    with torch.no_grad():
        model.eval()
        for batch_idx, (datapack) in tqdm(enumerate(testloader)):
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['label'].to(dtype=torch.float32, device=opt.device)  # (b,1,256,256)
            preds = model(imgs, None, None, None)
            preds = preds['masks']  # (b,3,256,256)
            for i in range(preds.shape[0]):
                pred = preds[i].argmax(dim=0).detach().cpu().numpy()
                label = masks[i].squeeze(0).long().detach().cpu().numpy()
                cv2.imwrite(f"./pred/{fix}/{fold}/{batch_idx * 4 + i + 1}.png", pred)
                cv2.imwrite(f"./gt/{fix}/{fold}/{batch_idx * 4 + i + 1}.png", label)


if __name__ == '__main__':
    fix = "AoP-SAM"
    for fold in range(5):
        ckpt = os.listdir(f"./checkpoints/fold{fold}")[-1]
        path = os.path.join(f"./checkpoints/fold{fold}", ckpt)
        print(path)
        main(path, fix, f"fold{fold}")

