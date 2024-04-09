from concurrent.futures import thread
import nibabel as nib
import torchio as tio
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import os
import numpy as np

class My_dataset(Dataset): 
    def __init__(self, paths, mode='train_glb', global_image_size=(128, 128, 128), local_image_size=(320, 320, 320)):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.global_image_size = global_image_size
        self.local_image_size = local_image_size
        self.mode = mode
    
    def _set_file_paths(self, root_dir):
        self.image_paths = []
        self.label_paths = []
        self.label1_paths = []   # 根管标签

        for patient_dir in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for file_name in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file_name)
                if 'original.nii' in file_name:  # original_preprocessed.nii.gz
                    self.image_paths.append(file_path)
                elif 'origin.mha' in file_name:
                    self.image_paths.append(file_path)
                elif 'seg.nii' in file_name:  # seg_preprocessed.nii.gz
                    self.label_paths.append(file_path)
                elif 'seg_root.nii' in file_name:  # seg_preprocessed.nii.gz
                    self.label1_paths.append(file_path)

    def __len__(self):
        return len(self.image_paths)

    def check_size(self, subject):
        if self.mode == 'train_glb' or self.mode == 'test_glb':
            if subject['image'][tio.DATA].shape[1:] != self.global_image_size:
                raise RuntimeError(f"global image shape is not {self.global_image_size} as expected.")
            if subject['label'][tio.DATA].shape[1:] != self.global_image_size:
                raise RuntimeError(f"global label shape is not {self.global_image_size} as expected.")
            if subject['label1'][tio.DATA].shape[1:] != self.global_image_size:
                raise RuntimeError(f"global label1 shape is not {self.global_image_size} as expected.")
        elif self.mode == 'train_local' or self.mode == 'test_local':
            if subject['image'][tio.DATA].shape[1:] != self.local_image_size:
                raise RuntimeError(f"local image shape is not {self.local_image_size} as expected.")
            if subject['label'][tio.DATA].shape[1:] != self.local_image_size:
                raise RuntimeError(f"local label shape is not {self.local_image_size} as expected.")
            if subject['label1'][tio.DATA].shape[1:] != self.local_image_size:
                raise RuntimeError(f"local label1 shape is not {self.local_image_size} as expected.")
        elif self.mode == 'infer_glb':
            if subject['image'][tio.DATA].shape[1:] != self.global_image_size:
                raise RuntimeError(f"local image shape is not {self.global_image_size} as expected.")

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        if self.mode in ['train_glb', 'test_glb', 'train_local', 'test_local', 'train_local2global', 'test_local2global', 'train_all', 'test_all']:
            sitk_label = sitk.ReadImage(self.label_paths[index])
            sitk_label1 = sitk.ReadImage(self.label1_paths[index])

            sitk_image.SetOrigin(sitk_label.GetOrigin())
            sitk_image.SetOrigin(sitk_label1.GetOrigin())
            sitk_image.SetDirection(sitk_label.GetDirection())
            sitk_image.SetDirection(sitk_label1.GetDirection())

            subject_glb = tio.Subject(
                image = tio.ScalarImage.from_sitk(sitk_image),
                label = tio.LabelMap.from_sitk(sitk_label),
                label1 = tio.LabelMap.from_sitk(sitk_label1),
            )

            subject_local = tio.Subject(
                image = tio.ScalarImage.from_sitk(sitk_image),
                label = tio.LabelMap.from_sitk(sitk_label),
                label1 = tio.LabelMap.from_sitk(sitk_label1),
            )   
            subject_glb = tio.Clamp(-1000,1000)(subject_glb)
            subject_local = tio.Clamp(-1000,1000)(subject_local)
        elif self.mode == 'infer_glb':
            subject_infer = tio.Subject(
                image = tio.ScalarImage.from_sitk(sitk_image),
            )
            subject_infer = tio.Clamp(-1000,1000)(subject_infer)
        elif self.mode == 'infer_all':
            subject_infer_glb = tio.Subject(
                image = tio.ScalarImage.from_sitk(sitk_image),
            )
            subject_infer_local = tio.Subject(
                image = tio.ScalarImage.from_sitk(sitk_image),
            )
            subject_infer_glb = tio.Clamp(-1000,1000)(subject_infer_glb)
            subject_infer_local = tio.Clamp(-1000,1000)(subject_infer_local)

            image = tio.ScalarImage.from_sitk(sitk_image)
            image_origin = sitk_image.GetOrigin()
            image_size = sitk_image.GetSize()
            image_direction = sitk_image.GetDirection()

        try:
            if self.mode == 'train_glb' or self.mode == 'test_glb':        # train the global branch
                padding_transform = tio.CropOrPad((640, 640, 640))   # padding to 640x640x640
                resize_transform = tio.Resize(self.global_image_size)
                subject_glb = padding_transform(subject_glb)
                subject_glb = resize_transform(subject_glb)   # 320 320 320
                self.check_size(subject_glb)

            elif self.mode == 'infer_all':        # train the global branch
                padding_transform = tio.CropOrPad((640, 640, 640))   # padding to 640x640x640
                resize_transform = tio.Resize(self.global_image_size)
                resize_transform1 = tio.Resize(self.local_image_size)
                subject_infer_glb = padding_transform(subject_infer_glb)
                subject_infer_glb = resize_transform(subject_infer_glb)   # 320 320 320
                self.check_size(subject_infer_glb)

                subject_infer_local = padding_transform(subject_infer_local)
                subject_infer_local = resize_transform1(subject_infer_local)   # 320 320 320
                self.check_size(subject_infer_local)
                
                patch_size = 64
                patch_overlap = 0
                sampler = tio.inference.GridSampler(subject_infer_local, patch_size, patch_overlap)
                patch_loader = torch.utils.data.DataLoader(sampler, batch_size=1)
                infer_image_patches = []
                patch_locations = []
                for patches_batch in patch_loader:
                    image_patch = patches_batch['image'][tio.DATA]  
                    infer_image_patches.append(image_patch)

                    patch_location = patches_batch[tio.LOCATION]
                    patch_locations.append(patch_location)
                if len(infer_image_patches) == 125:
                    stacked_tensor = torch.stack(infer_image_patches).squeeze(2)
                    # print(stacked_tensor.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")

                subject_infer_local['image'] = stacked_tensor

            elif self.mode == 'train_local' or self.mode == "test_local":        # train the global branch
                padding_transform = tio.CropOrPad((640, 640, 640))   # padding to 640x640x640
                resize_transform = tio.Resize(self.local_image_size)
                subject_local = padding_transform(subject_local)
                subject_local = resize_transform(subject_local)   # 320 320 320
                self.check_size(subject_local)
                
                patch_size = 128
                patch_overlap = 0
                sampler = tio.inference.GridSampler(subject_local, patch_size, patch_overlap)
                patch_loader = torch.utils.data.DataLoader(sampler, batch_size=1)
                infer_image_patches = []
                infer_label_patches = []
                infer_label1_patches = []
                patch_locations = []
                for patches_batch in patch_loader:
                    image_patch = patches_batch['image'][tio.DATA]  
                    label_patch = patches_batch['label'][tio.DATA]
                    label1_patch = patches_batch['label1'][tio.DATA]
                    infer_image_patches.append(image_patch)
                    infer_label_patches.append(label_patch)
                    infer_label1_patches.append(label1_patch)

                    patch_location = patches_batch[tio.LOCATION]
                    patch_locations.append(patch_location)

                if len(infer_image_patches) == 125:
                    stacked_tensor = torch.stack(infer_image_patches).squeeze(2)
                    # print(stacked_tensor.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")
                if len(infer_label_patches) == 125:
                    stacked_tensor0 = torch.stack(infer_label_patches).squeeze(2)
                    # print(stacked_tensor0.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")
                if len(infer_label1_patches) == 125:
                    stacked_tensor1 = torch.stack(infer_label1_patches).squeeze(2)
                    # print(stacked_tensor1.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")

                subject_local['image'] = stacked_tensor
                subject_local['label'] = stacked_tensor0
                subject_local['label1'] = stacked_tensor1

            elif self.mode in  ['train_local2global', 'test_local2global', 'train_all', 'test_all']:        # train the global branch
                # padding_transform = tio.CropOrPad((640, 640, 640))   # padding to 640x640x640
                resize_transform = tio.Resize(self.global_image_size)
                resize_transform1 = tio.Resize(self.local_image_size)
                # subject_glb = padding_transform(subject_glb)
                subject_glb = resize_transform(subject_glb)   
                self.check_size(subject_glb)

                # subject_local = padding_transform(subject_local)
                subject_local = resize_transform1(subject_local)   # 320 320 320
                self.check_size(subject_local)
                
                patch_size = 64
                patch_overlap = 0
                sampler = tio.inference.GridSampler(subject_local, patch_size, patch_overlap)
                patch_loader = torch.utils.data.DataLoader(sampler, batch_size=1)
                infer_image_patches = []
                infer_label_patches = []
                infer_label1_patches = []
                patch_locations = []
                for patches_batch in patch_loader:
                    image_patch = patches_batch['image'][tio.DATA]  
                    label_patch = patches_batch['label'][tio.DATA]
                    label1_patch = patches_batch['label1'][tio.DATA]
                    infer_image_patches.append(image_patch)
                    infer_label_patches.append(label_patch)
                    infer_label1_patches.append(label1_patch)

                    patch_location = patches_batch[tio.LOCATION]
                    patch_locations.append(patch_location)
                if len(infer_image_patches) == 125:
                    stacked_tensor = torch.stack(infer_image_patches).squeeze(2)
                    # print(stacked_tensor.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")
                if len(infer_label_patches) == 125:
                    stacked_tensor0 = torch.stack(infer_label_patches).squeeze(2)
                    # print(stacked_tensor0.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")
                if len(infer_label1_patches) == 125:
                    stacked_tensor1 = torch.stack(infer_label1_patches).squeeze(2)
                    # print(stacked_tensor1.shape)  # Should print: torch.Size([125, 1, 128, 128, 128])
                else:
                    print("The length of the list is not 125.")

                subject_local['image'] = stacked_tensor
                subject_local['label'] = stacked_tensor0
                subject_local['label1'] = stacked_tensor1
                

        except Exception as e:
            print(f"An error occurred: {e}")
            print(self.image_paths[index])
            raise

        # print(subject.image.data.shape)
        if self.mode == "train_glb" or self.mode == "test_glb":
            return subject_glb.image.data.clone().detach(), subject_glb.label.data.clone().detach(), subject_glb.label1.data.clone().detach()
        if self.mode == "infer_all":
            return subject_infer_glb.image.data.clone().detach(), subject_infer_local['image'].clone().detach(), patch_locations, image, image_size, image_origin, image_direction
        if self.mode == "train_local" or self.mode == "test_local":
            return subject_local['image'].clone().detach(), subject_local['label'].clone().detach(), subject_local['label1'].clone().detach(), patch_locations
        if self.mode in  ['train_local2global', 'test_local2global', 'train_all', 'test_all']:
            return subject_glb.image.data.clone().detach(), subject_glb.label.data.clone().detach(), subject_glb.label1.data.clone().detach(), \
                    subject_local['image'].clone().detach(), subject_local['label'].clone().detach(), subject_local['label1'].clone().detach(), patch_locations

class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
     
if __name__ == "__main__":
    data_dir = './data/train/'
    # Create a dataset object
    dataset = My_dataset(data_dir, mode='train_glb', global_image_size=(256, 256, 256))

    # Create a DataLoader
    batch_size = 1  # adjust as needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the DataLoader
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        numpy_arr = labels.squeeze(0).cpu().numpy()
        for i in range(numpy_arr.shape[0]):
            img = nib.Nifti1Image(numpy_arr[i], np.eye(4))
            nib.save(img, f"./label_class_{i}.nii.gz")
        unique = torch.unique(labels[:, 0])
        print(f"Unique values in labels: {unique}")



