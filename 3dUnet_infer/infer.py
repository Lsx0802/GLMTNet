import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from models.my_unet import DoubleConv, ResUnet_light
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from dataload import My_dataset, Union_Dataloader
import argparse
from torch import nn, optim
import my_loss
from tqdm import tqdm
import nibabel as nib
import random
import datetime
from torch.backends import cudnn
from monai.losses import DiceCELoss
from contextlib import nullcontext
import torch.nn.functional as F
import torchio as tio
import SimpleITK as sitk

parser = argparse.ArgumentParser(description="Script to train")
parser.add_argument('--data_dir', type=str, default='./data/infer/', help='Path to data directory')
parser.add_argument('--mode', type=str, default='infer_all', help='Mode for the dataset')
parser.add_argument('--global_image_size', type=tuple, default=(256, 256, 256), help='Global image size')
parser.add_argument('--local_image_size', type=tuple, default=(320, 320, 320), help='local image size, patch size is 64, so we have (local image size/64)^3 patchs')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the DataLoader')
parser.add_argument("--ckpt", type=str, default="./runs/exp_all_2de_uncertain/last_local2global.pth", help="the path of model weight file")
parser.add_argument("--ckpt1", type=str, default="./runs/exp_all_2de_uncertain/last_global2local.pth", help="the path of model weight file")  
parser.add_argument("--save_path", type=str, default="./runs/exp1/result/", help="the path to save")
parser.add_argument("--device", type=str, default='cuda:2')
parser.add_argument("--device1", type=str, default='cuda:3')

args = parser.parse_args()

class Tester:
    def __init__(self, args):
        self.args = args
        self.unet_model = None
        self.dataloaders = None
        self.checkpoint = None 

    def get_dataloaders(self):
        dataset = My_dataset(paths=self.args.data_dir, mode=self.args.mode, global_image_size=self.args.global_image_size, local_image_size=self.args.local_image_size)
        self.dataloaders = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

    def stitch_together(self, patches, patch_locations, output_size=(320, 320, 320)):
        stitched_output = torch.zeros(output_size, dtype=torch.uint8)

        for patch, location in zip(patches, patch_locations):
            start_x, start_y, start_z, end_x, end_y, end_z = location[0][0]
            stitched_output[start_x:end_x, start_y:end_y, start_z:end_z] = patch.squeeze()

        return stitched_output

    def local2global_stitch_together(self, patches, patch_locations):
        c, d, h, w = patches.size()[1], patches.size()[2]*5, patches.size()[3]*5, patches.size()[4]*5
        stitched_output = torch.zeros(1, c, d, h, w)   # 5120
        ratio = int(320 / (5120 / c))
        for patch, location in zip(patches, patch_locations):
            start_x, start_y, start_z, end_x, end_y, end_z = location[0][0]
            stitched_output[:, :, 
                            int(start_x / ratio):int(end_x / ratio), 
                            int(start_y / ratio):int(end_y / ratio), 
                            int(start_z / ratio):int(end_z /ratio)] = patch
        resized = F.interpolate(stitched_output.float(), size=(patches.size()[2]*4, patches.size()[3]*4, patches.size()[4]*4), mode='trilinear', align_corners=False)

        return resized

    def expand_and_split(self, input_tensor, locations, ratio):
            expanded_tensor = F.interpolate(input_tensor.float(), size=(320, 320, 320), mode='trilinear', align_corners=False)
            patches = []
            for location in locations:
                start_x, start_y, start_z, end_x, end_y, end_z = location[0][0]
                start_x, start_y, start_z, end_x, end_y, end_z = [int(x / ratio) for x in [start_x, start_y, start_z, end_x, end_y, end_z]]
                patch = expanded_tensor[:, :, start_x:end_x, start_y:end_y, start_z:end_z]
                patches.append(patch)

            patches_tensor = torch.cat(patches, dim=0)
            return patches_tensor

    def build_model(self):
        self.unet_model = ResUnet_light(1, 1).to(self.args.device)
        if self.args.mode in ['infer_glb']:
            self.unet_model = ResUnet_light(1, 2).to(self.args.device)
        elif self.args.mode in ['infer_all']:
            self.unet_model = ResUnet_light(1, 1).to(self.args.device)
            self.unet_model1 = ResUnet_light(1, 1).to(self.args.device1)

    def load_checkpoint(self):
        if os.path.exists(self.args.ckpt): # load model
            self.checkpoint = torch.load(self.args.ckpt, map_location=self.args.device)
            self.unet_model.load_state_dict(self.checkpoint['model_state_dict'])
            print('load model successfully')
        elif self.args.mode in ['infer_all']:
            if os.path.exists(self.args.ckpt) and os.path.exists(self.args.ckpt1): # load model
                self.checkpoint = torch.load(self.args.ckpt, map_location=self.args.device)
                self.checkpoint1 = torch.load(self.args.ckpt1, map_location=self.args.device1)
                self.unet_model.load_state_dict(self.checkpoint['model_state_dict'])
                self.unet_model1.load_state_dict(self.checkpoint1['model_state_dict'])
                print('load model successfully')

    def save_nii_file(self, tensor_input, tensor_output, step):
        numpy_arr = tensor_input.squeeze(0).cpu().numpy()
        img = nib.Nifti1Image(numpy_arr[0], np.eye(4))
        nib.save(img, self.args.save_path + f'infer_input_{step}.nii.gz')

        numpy_arr = tensor_output.squeeze(0).cpu().detach().numpy()   # (2, 256, 256, 256)
        numpy_arr = np.where(numpy_arr >= 0.5, 1.0, 0.0)
        for i in range(numpy_arr.shape[0]):
            img = nib.Nifti1Image(numpy_arr[i], np.eye(4))
            nib.save(img, self.args.save_path + f'infer_output_{step}.nii.gz')

    def resize_image_itk(self, itkimage, newSize, origin, direction, resamplemethod=sitk.sitkNearestNeighbor):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        newSize = np.array(newSize, float)
        factor = originSize / newSize
        newSpacing = originSpacing * factor
        newSize = newSize.astype(int)  # spacing肯定不能是整数
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        return itkimgResampled

    def test(self):
        if self.args.mode == "infer_glb":
            self.infer_glb_epoch()
        elif self.args.mode == "infer_all":
            self.infer_all_epoch()
            pass

    def infer_glb_epoch(self):
        self.unet_model.eval()
        unet_model = self.unet_model
        tbar = tqdm(self.dataloaders)

        for step, (x) in enumerate(tbar):
            input = x.type(torch.FloatTensor).to(self.args.device)
            output = unet_model(input)
            self.save_nii_file(input, output, step)
            print('infer down')

    def infer_all_epoch(self):
        self.unet_model.eval()
        self.unet_model1.eval()
        unet_model = self.unet_model
        unet_model1 = self.unet_model1
        tbar = tqdm(self.dataloaders)


        for step, (x, x1, locations, image, image_size, image_origin, image_direction) in enumerate(tbar):
            input = x.type(torch.FloatTensor).to(self.args.device)
            input1 = x1.type(torch.FloatTensor).to(self.args.device1).squeeze(0)

            down0_local = unet_model1.down1(input1)
            down0 = unet_model.down1(input)
            with torch.no_grad():
                down0_ext1 = self.local2global_stitch_together(down0_local, locations)
                down0_ext = self.expand_and_split(down0, locations, 1)
            down1_local = unet_model1.down1_pool(down0_local, down0_ext.to(self.args.device1))
            down1 = unet_model.down1_pool(down0, down0_ext1.to(self.args.device))

            down1_local = unet_model1.down2(down1_local)
            down1 = unet_model.down2(down1)
            with torch.no_grad():
                down1_ext1 = self.local2global_stitch_together(down1_local, locations)
                down1_ext = self.expand_and_split(down1, locations, 2)
            down2_local = unet_model1.down2_pool(down1_local, down1_ext.to(self.args.device1))
            down2 = unet_model.down2_pool(down1, down1_ext1.to(self.args.device))

            down2_local = unet_model1.down3(down2_local)
            down2 = unet_model.down3(down2)
            with torch.no_grad():
                down2_ext1 = self.local2global_stitch_together(down2_local, locations)
                down2_ext = self.expand_and_split(down2, locations, 4)
            down3_local = unet_model1.down3_pool(down2_local, down2_ext.to(self.args.device1))
            down3 = unet_model.down3_pool(down2, down2_ext1.to(self.args.device))

            down3_local = unet_model1.down4(down3_local)
            with torch.no_grad():
                down3_ext1 = self.local2global_stitch_together(down3_local, locations)
            down3 = unet_model.down4(down3, down3_ext=down3_ext1.to(self.args.device))

            output0 = unet_model.upsample0(down3, down2, down1, down0)
            output1 = unet_model.upsample1(down3, down2, down1, down0)

            output0_np = output0.detach().cpu().numpy()
            output1_np = output1.detach().cpu().numpy()

            output = np.zeros_like(output0_np)

            output[output0_np == 1] = 1
            output[output1_np == 1] = 2

            size =[i.item() for i in image_size]
            direction = [i.item() for i in image_direction]
            origin = [i.item() for i in image_origin]

            itkimage = sitk.GetImageFromArray(output.squeeze(0).squeeze(0))
            itkimage = self.resize_image_itk(itkimage, size, origin, direction)
            sitk.WriteImage(itkimage, "./runs/result/" + f"output{step}.nii.gz")


def main_worker(args):
    init_seeds(2024)
    tester = Tester(args)
    tester.get_dataloaders()
    tester.build_model()
    tester.load_checkpoint()
    tester.test()

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True



if __name__ == "__main__":
    main_worker(args)


