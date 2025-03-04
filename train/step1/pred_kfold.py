import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D, correct_dims
from torch.utils.data import Dataset
import SimpleITK
from pathlib import Path
import cv2
from tqdm import tqdm


class Fetal_dataset(Dataset):
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(i))) for i
                  in
                  list_dir[0]]
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
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('-encoder_input_size', type=int, default=256,help='256 for vit-b, 512 for vit-l and vit-h')
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--task', default='PSFH', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b',
                        help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str,default=ckpt)
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

    args = parser.parse_args()

    device = torch.device("cuda")

    seed_value = 2023
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    batch_size = 4

    os.makedirs("./gt", exist_ok=True)
    os.makedirs("./pred", exist_ok=True)
    os.makedirs(f"./gt/{fix}",exist_ok=True)
    os.makedirs(f"./pred/{fix}", exist_ok=True)


    os.makedirs(f"./gt/{fix}/{fold}", exist_ok=True)
    os.makedirs(f"./pred/{fix}/{fold}", exist_ok=True)

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=256,
                              crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True)
    root_path = Path('./testdataset')
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 701)])

    with open("./test.txt", "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    print(len(test_index))
    db_test = Fetal_dataset(transform=tf_val,list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)]))
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = get_model(args=args)
    model.load_state_dict(torch.load(ckpt))
    model.to(device)


    with torch.no_grad():
        model.eval()
        for batch_idx, (datapack) in tqdm(enumerate(testloader)):
            imgs = datapack['image'].to(dtype=torch.float32, device=device)
            masks = datapack['label'].to(dtype=torch.float32, device=device)  # (b,1,256,256)
            preds = model(imgs, None, None, None)
            preds = preds['masks']  # (b,3,256,256)
            for i in range(preds.shape[0]):
                pred = preds[i].argmax(dim=0).detach().cpu().numpy()
                label = masks[i].squeeze(0).long().detach().cpu().numpy()
                cv2.imwrite(f"./pred/{fix}/{fold}/{batch_idx * 4 + i + 1}.png", pred)
                cv2.imwrite(f"./gt/{fix}/{fold}/{batch_idx * 4 + i + 1}.png", label)


if __name__ == '__main__':
    fix = "AoP-SAM"
    for fold in [0]:
        path = os.path.join(f"./checkpoints/{fix}/fold0", "AoP-SAM_fold0_epoch29_0.029555991.pth")
        print(path)
        main(path, fix, f"fold{fold}")
