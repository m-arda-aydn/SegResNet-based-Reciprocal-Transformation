import os
import json 

json_dict = dict()
train_dict_list = list()

images = sorted(os.listdir('./dataset_mha/1ADC_ss'))
z_maps = sorted(os.listdir('./dataset_mha/2Z_ADC'))
labels = sorted(os.listdir('./dataset_mha/3LABEL'))

for i in range(len(images)):
     temp_dict = dict()
     image_path = './dataset_mha/1ADC_ss/' + str(images[i])
     zmap_path = './dataset_mha/2Z_ADC/' + str(z_maps[i])
     label_path = './dataset_mha/3LABEL/' + str(labels[i])
     temp_dict['image'] = image_path
     temp_dict['zmap'] = zmap_path
     temp_dict['label'] = label_path
     train_dict_list.append(temp_dict)

json_dict['training'] = train_dict_list

with open('./bonbid_dataset_monai/dataset.json','w') as file:
     json.dump(json_dict,file)