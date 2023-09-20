# inspired from https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

import json 
import torch
from monai.transforms import (Compose,
                            LoadImaged,
                            MapTransform,
                            NormalizeIntensityd,
                            Activations, 
                            AsDiscrete,
                            RandFlipd,
                            RandRotated,
                            RandZoomd)
import numpy as np
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
import time
from monai.data import decollate_batch, CacheDataset
from datetime import datetime
from sklearn.model_selection import KFold

original_list = list()

# our custom MONAI transforms
class PreprocessDatad(MapTransform):
    def __init__(self, keys, divider):
        super().__init__(keys)
        self.divider = divider
        
    def __call__(self, x):
        for key in self.keys:
            x[key] = x[key].unsqueeze(0)
            remainder = x[key].shape[3] % self.divider
            if key == 'image':
                original_list.append(x[key].shape[3]) 
            if remainder != 0:
                _,H,W,_ = x[key].shape # 1,H,W,D
                x[key] = torch.cat([x[key],torch.zeros(1,H,W,self.divider - remainder)],dim=3)
        return x

class ConcatTwoChanneld(MapTransform):
    def __init__(self,keys):
        super().__init__(keys)
        
    def __call__(self, x):
        adc = x['image']
        z_adc = x['zmap']
        x['image'] = torch.cat([adc,z_adc],dim=0)
        return x
    
class Permuted(MapTransform):
    def __init__(self,keys):
        super().__init__(keys)
        
    def __call__(self, x):
        for key in self.keys:
            x[key] = x[key].permute(0,3,1,2) # shape C,D,H,W
        return x

class ReciprocalTransformd(MapTransform):
    def __init__(self,keys,power):
        super().__init__(keys)
        self.power = power

    def __call__(self, x):
        adc = x['image']
        z_adc = x['zmap']
        D = adc.shape[3]
        for d in range(D):
            min_data = torch.min(z_adc[:,:,:,d]).item()
            x['image'] = adc[:,:,:,d] / (1 + abs(min_data) + z_adc[:,:,:,d])**(self.power)

        return x
        
class ReciprocalTransform_Concatd(MapTransform):
    def __init__(self,keys,power):
        super().__init__(keys)
        self.power = power

    def __call__(self, x):
        adc = x['image']
        D = adc.shape[3]
        for d in range(D):
            min_data = torch.min(x['zmap'][:,:,:,d]).item()
            adc[:,:,:,d] = x['image'][:,:,:,d] / (1 + abs(min_data) + x['zmap'][:,:,:,d])**(self.power)

        x['image'] = torch.cat([x['image'],x['zmap'],adc],dim=0)
        return x
    
def RemovePadding(batch_data,original_list,index):
    batch_data = batch_data[:,:,:original_list[index],:,:] # (N,C,D,H,W)
    return batch_data     
    

now = datetime.now()
filename_time = now.strftime("%d_%m_%Y_%H_%M_%S")
work_dir = './work_dir'
checkpoint_base_path = './best_checkpoints'

with open('/home/tos_group/bonbid_challenge/bonbid_dataset_monai/dataset.json','r') as js_file:
    json_object = json.load(js_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)
resume = False

train_transform = Compose(
    [
        LoadImaged(keys=["image","zmap","label"], reader = 'ITKReader'),
        PreprocessDatad(keys=["image","zmap","label"], divider=8),
        # ReciprocalTransformd(keys=["image","zmap"], power=1.5),
        ReciprocalTransform_Concatd(keys=["image","zmap"], power=1.5),
        # ConcatTwoChanneld(keys=["image","zmap"]),
        RandZoomd(keys=["image","zmap", "label"], min_zoom=1, max_zoom=1.25, prob=0.1),
        RandFlipd(keys=["image","zmap", "label"], prob=0.1, spatial_axis=0),
        RandFlipd(keys=["image","zmap", "label"], prob=0.1, spatial_axis=1),
        RandFlipd(keys=["image","zmap", "label"], prob=0.1, spatial_axis=2),
        RandRotated(keys=["image","zmap", "label"],range_x=1.5, prob=0.1),
        NormalizeIntensityd(keys=["image","zmap"], nonzero=True, channel_wise=True),
        Permuted(keys=["image","zmap", "label"]),
        
    ]
)

val_transform = Compose(
        [LoadImaged(keys=["image","zmap","label"], reader = 'ITKReader'),
        PreprocessDatad(keys=["image","zmap","label"], divider=8),
        # ReciprocalTransformd(keys=["image","zmap"], power=1.5),
        ReciprocalTransform_Concatd(keys=["image","zmap"], power=1.5),
        # ConcatTwoChanneld(keys=["image","zmap"]),
        NormalizeIntensityd(keys=["image","zmap"], nonzero=True, channel_wise=True),
        Permuted(keys=["image","zmap", "label"]),
        ]
)

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

transform_name_list_train = list(transform.__class__.__name__ for transform in train_transform.transforms)
transform_name_list_val = list(transform.__class__.__name__ for transform in val_transform.transforms)

train_dataset = CacheDataset(json_object['training'],transform=train_transform)
val_dataset = CacheDataset(json_object['training'],transform=val_transform)

max_epochs = 500
val_interval = 5
verbose_interval = 10
scaler = torch.cuda.amp.GradScaler()

learning_rate = 1e-5
weight_decay = 1e-5
dice_metric = DiceMetric(include_background=True, reduction="mean")

ori_size_path = './original_size'

k = 5  
kf = KFold(n_splits=k, shuffle=False)
json_dict = dict()
ori_size_dict = dict()

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    total_start = time.time()

    model = SegResNet(spatial_dims=3,init_filters=32,in_channels=3,out_channels=1,
                      dropout_prob=0.2,num_groups=8,norm_name='GROUP',upsample_mode='deconv').to(device)
    
    loss_function = TverskyLoss(smooth_nr=2e-5, sigmoid=True,smooth_dr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    model_name = str(model.__class__.__name__)
    optimizer_name = str(optimizer.__class__.__name__)
    lr_scheduler_name = str(lr_scheduler.__class__.__name__)
    loss_name = str(loss_function.__class__.__name__)

    if resume:
            best_checkpoint_path = checkpoint_base_path + '/SegResNet_best_checkpoint_fold_1_15_09_2023_10_53_51.pth'
            checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict']) 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']) 
            with open(work_dir + '/' + filename_time + '_' + model_name + '_score.txt','a') as txt_file:
                txt_file.write(f"Weights loaded from: {best_checkpoint_path}\n")
    
    json_dict[f'fold_{fold+1}'] = val_idx.tolist()
    with open(work_dir + '/' + filename_time + '_' + model_name + '_val_index.json','w') as val_index_file:
        json.dump(json_dict,val_index_file)

    if fold == 0:       
        
        ori_size_dict['size'] = original_list
        with open(ori_size_path + '/' + filename_time + '_' + model_name + '_ori_size.json','w') as ori_size_file:
            json.dump(ori_size_dict,ori_size_file)

        with open(work_dir + '/' + filename_time + '_' + model_name + '_score.txt','a') as txt_file:
                        txt_file.write(f"model name: {model_name}\n"
                                    f"learning rate: {learning_rate}\n"
                                    f"weight decay: {weight_decay}\n"
                                    f"optimizer: {optimizer_name}\n"
                                    f"learning rate scheduler: {lr_scheduler_name}\n"
                                    f"loss: {loss_name}\n"
                                    f"train transforms: {transform_name_list_train}\n"
                                    f"validation transforms: {transform_name_list_val}\n"
                                    f"model summary: {model}\n")


    with open(work_dir + '/' + filename_time + '_' + model_name + '_score.txt','a') as txt_file:
                txt_file.write(f"\nFold: {fold + 1}/{k}\n")

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []

    train_dataset_subset = torch.utils.data.Subset(train_dataset,train_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset_subset, batch_size=1,shuffle=False)
    val_dataset_subset = torch.utils.data.Subset(val_dataset,val_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset_subset, batch_size=1,shuffle=False)


    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        with open(work_dir + '/' + filename_time + '_' + model_name + '_losses.txt','a') as txt_file:
                        txt_file.write(f"epoch {epoch+1}/{max_epochs}\n")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if step % verbose_interval == 0:
                print(
                    f"{step}/{len(train_loader) // train_loader.batch_size}"
                    f", train_loss: {loss.item():.4f}"
                    f", step time: {(time.time() - step_start):.4f}")
                
                with open(work_dir + '/' + filename_time + '_' + model_name + '_losses.txt','a') as txt_file:
                        txt_file.write(f"{step}/{len(train_loader) // train_loader.batch_size}"
                        f", train_loss: {loss.item():.4f}"
                        f", step time: {(time.time() - step_start):.4f}\n")

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        val_original_size = np.array(original_list)[val_idx]

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i,val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    val_labels = RemovePadding(val_labels,val_original_size,i)
                    val_outputs = RemovePadding(val_outputs,val_original_size,i) 
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict()}, checkpoint_base_path + '/' + 
                            model_name + '_best_checkpoint_fold_' + str(fold+1) + '_' + filename_time + '.pth')
                    print("saved new best metric model")
                print(
                    f"current learning rate: {lr_scheduler.get_last_lr()[0]:.7f}\n"
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

                with open(work_dir + '/' + filename_time + '_' + model_name + '_score.txt','a') as txt_file:
                    txt_file.write(
                    f"current learning rate: {lr_scheduler.get_last_lr()[0]:.7f}\n"
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}\n"
                    f"best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}\n")

        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")