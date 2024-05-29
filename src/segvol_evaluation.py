from transformers import AutoModel, AutoTokenizer
import os
import numpy as np
import torch
from monai import data, transforms
from torch.utils.data import Dataset, ConcatDataset
from scipy import sparse
import ast
import os
import json
import json
from monai import transforms
import os
import torch 
from tqdm import tqdm
import csv
import gc
import pickle
import argparse

from .model import SegVolGroup5
from .adapted_vit import AdaptedViT

from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

def freeze_model(model):
    # Freeze everything
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False 

    # Set new patch_embedding to trainable
    for name, param in model.model.image_encoder.patch_embedding_equiv.named_parameters():
        if 'position_embeddings' not in name:
            param.requires_grad = True 
        else:
            param.requires_grad = False

    # Set adapter module to trainable
    for param in model.model.image_encoder.adapter.parameters():
        param.requires_grad = True 
    
    return model 

def get_checkpoint_model(path):
    clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
    original = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=False)
    original.model.text_encoder.tokenizer = clip_tokenizer

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["out_proj"],
        lora_dropout=0.1,
        bias="none"
    )

    # Instantiate adapted ViT
    adapted_ViT = AdaptedViT()
    # Load the pretrained model weights and update
    pretrained_state_dict = original.model.image_encoder.state_dict()
    adapted_state_dict = adapted_ViT.state_dict()
    to_update = {module : weights for module, weights in pretrained_state_dict.items() if (module in adapted_state_dict.keys()) and (adapted_state_dict[module].size() == weights.size())}
    # Copy weights to our model
    adapted_state_dict.update(to_update)
    adapted_ViT.load_state_dict(adapted_state_dict)
    # Switch the ViT
    original.model.image_encoder = adapted_ViT
    # Freeze all weights, except the right ones
    original = freeze_model(original)
    # Inject LoRA in ViT
    ViT = original.model.image_encoder
    ViT = get_peft_model(ViT, lora_config)
    original.model.image_encoder = ViT
    # Just to be sure, freeze weights again. 
    original = freeze_model(original)

    new_model = SegVolGroup5(
        image_encoder=original.model.image_encoder,
        mask_decoder=original.model.mask_decoder,
        prompt_encoder=original.model.prompt_encoder,
        roi_size=original.config.spatial_size,
        patch_size=original.config.patch_size,
        text_encoder=original.model.text_encoder,
        test_mode=True
    )

    statedict = torch.load(path)
    model_dict = statedict['model']
    new_model.load_state_dict(model_dict,strict=False)

    for name, param in new_model.named_parameters():
        if param.requires_grad:
            print(name)
    
    original.model = new_model

    original.eval()
    print('Model loaded succesfully!')
    return original

class UnionDataset(Dataset):
    def __init__(self, concat_dataset, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = torch.cumsum(torch.tensor([0] + self.lengths), dim=0)
        self.concat_dataset = concat_dataset

        
    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

class SegVolDataset(Dataset):
    def __init__(self, data_dir, data, organ_list, transform):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform

        organ_list.remove('background')
        self.target_list = organ_list
    
    def load_uniseg_case(self, ct_npy_path, gt_npy_path):
        """ 
            https://huggingface.co/BAAI/SegVol/blob/main/model_segvol_single.py
        """
        img_array = np.load(ct_npy_path)
        allmatrix_sp= sparse.load_npz(gt_npy_path)
        if 'mask_' in gt_npy_path:
            gt_shape = ast.literal_eval(gt_npy_path.split('_')[-1].replace('.npz', ''))
        else:
            gt_shape = ast.literal_eval(gt_npy_path.split('.')[-2])
        gt_array=allmatrix_sp.toarray().reshape(gt_shape)
        return img_array, gt_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get path
        item_dict = self.data[idx]
        ct_path, gt_path = os.path.join(self.data_dir, item_dict['image']), os.path.join(self.data_dir, item_dict['label'])

        ct_npy, gt_npy = self.load_uniseg_case(ct_path, gt_path)

        item = {'image' : ct_npy, 'label' : gt_npy}
        item = self.transform(item)

        return  item
    
    

class Evaluator:
    """ 
        Experiments class that streamlines the inference and experimentation process.

        Setup an experiment as a function, define the dataset codes that are used in the experiment,
        define the type of prompts you want to use, and TODO : the organs to be used. 

        TODO : Save results, interpret results, implement experiments from paper.  
    """
    def __init__(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        print("Loading model...")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        # self.model = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=True)
        # self.model.model.text_encoder.tokenizer = self.clip_tokenizer
        self.model = get_checkpoint_model('src/medsam_30epochs_baseline.pth')
        self.model.to(device)
        self.model.eval()
        print(device)
        print('model load done')

        self.dataset_dict = {
            '0000' : 'CHAOS',
            '0001' : 'Han-Seg',
            '0002' : 'AMOS22',
            '0003' : 'AbdomenCT-1k',
            '0004' : 'KiTS23',
            '0005' : 'KiPA22',
            '0006' : 'KiTS19',
            '0007' : 'BTCV',
            '0008' : 'Pancreas-CT',
            '0009' : '3D-IRCADB',
            '0010' : 'FLARE22',
            '0011' : 'TotalSegmentator',
            '0012' : 'CT-ORG',
            '0013' : 'WORD',
            '0014' : 'VerSe19',
            '0015' : 'VerSe20',
            '0016' : 'SILVER07',
            '0017' : 'QUBIC',
            '0018' : 'MSD-Colon',
            '0019' : 'MSD-HepaticVessel',
            '0020' : 'MSD-Liver',
            '0021' : 'MSD-lung',
            '0022' : 'MSD-pancreas',
            '0023' : 'MSD-spleen',
            '0024' : 'LUNA16'
        }

    
    def build_concat_dataset(self, root_path, dataset_codes, transform):
        concat_dataset = []
        CombinationDataset_len = 0
        for dataset_code in dataset_codes:
            datalist_json = os.path.join(root_path, dataset_code, f'{dataset_code}.json')
            with open(datalist_json, 'r') as f:
                dataset_dict = json.load(f)

            datalist = dataset_dict['test']

            universal_ds = SegVolDataset(data_dir=root_path, data=datalist, organ_list=list(dataset_dict['labels'].values()), transform=transform)
            concat_dataset.append(universal_ds)
            CombinationDataset_len += len(universal_ds)

        self.categories = universal_ds.target_list

        print(f'CombinationDataset loaded, dataset size: {CombinationDataset_len}')
        return UnionDataset(ConcatDataset(concat_dataset), concat_dataset)


    def get_test_loader(self, args):
        if args['add_rotation']:
            self.test_transform = transforms.Compose(
                [
                    #transforms.RandRotated(keys=["image", "label"], range_x=0.5, range_y=0.5, range_z=0.5, prob=1.0)
                    transforms.Rotated(keys=["image", "label"], angle=(0.1, 0.1, 0.785398163))
                ]
            )
        else:
            self.test_transform = transforms.Compose(
                [
                   #transforms.RandRotated(keys=["image", "label"], range_x=30, range_y=30, range_z=30, prob=0.0)
                   transforms.Rotated(keys=["image", "label"], angle=(0.0, 0.0, 0.0))
                ]
            )

        print(f'----- test combination dataset -----')

        combination_train_ds = self.build_concat_dataset(root_path=args['data_dir'], dataset_codes=args['dataset_codes'], transform=self.test_transform)

        train_sampler = None 

        loader = data.DataLoader(
            combination_train_ds,
            batch_size=1,
            shuffle=(train_sampler is None),
            num_workers=1,
            sampler=train_sampler,
            pin_memory=False,
            persistent_workers=False
        )
        return loader
    
    def dice_score(self, preds, labels, device='cpu'):
        """"
            https://huggingface.co/BAAI/SegVol/blob/main/model_segvol_single.py
        """
        assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match\n" + str(preds.shape) + str(labels.shape)
        predict = preds.reshape(1, -1).to(device)
        target = labels.reshape(1, -1).to(device)

        predict = torch.sigmoid(predict)
        predict = torch.where(predict > 0.5, 1., 0.)
        
        tp = torch.sum(torch.mul(predict, target))
        den = torch.sum(predict) + torch.sum(target) + 1
        dice = 2 * tp / den
        return dice

    def inference(self, ct_npy, gt_npy, prompts=['text', 'bbox'], use_zoom=True, cls_idx=5):
        if ('point' in prompts) and ('bbox' in prompts):
            raise Exception('Point and bbox can not be used together!')
        
        # go through zoom_transform to generate zoomout & zoomin views
        data_item = self.model.processor.zoom_transform(ct_npy, gt_npy)

        # add batch dim manually
        data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
        data_item['image'].unsqueeze(0).to(self.device), data_item['label'].unsqueeze(0).to(self.device), data_item['zoom_out_image'].unsqueeze(0).to(self.device), data_item['zoom_out_label'].unsqueeze(0).to(self.device)

        # Create prompts
        text_prompt = [self.categories[cls_idx]]
        point_prompt, point_prompt_map = self.model.processor.point_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=self.device)   # inputs w/o batch dim, outputs w batch dim
        bbox_prompt, bbox_prompt_map = self.model.processor.bbox_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=self.device)   # inputs w/o batch dim, outputs w batch dim

        # Create argument dict
        arguments = dict()
        arguments['image'] = data_item['image']
        arguments['zoomed_image'] = data_item['zoom_out_image']
        arguments['use_zoom'] = use_zoom
        if 'text' in prompts:
            arguments['text_prompt'] = text_prompt
        if 'bbox' in prompts:
            arguments['bbox_prompt_group'] = [bbox_prompt, bbox_prompt_map]
        if 'point' in prompts:
            arguments['point_prompt_group'] = [point_prompt, point_prompt_map]
        
        logits_mask = self.model.forward_test(**arguments)
        print(data_item['label'].shape)
        dice = self.dice_score(logits_mask[0][0], data_item['label'][0][cls_idx], self.device)
        print(dice.item())
        return dice.item(), logits_mask
    

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def experiment_1(self, args, datasets=['0000'], prompts=['text', 'bbox'], use_zoom=True, add_rotation_transformation=False):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        # Select the organs you want to map
        target_organs = [
            'Liver'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp1_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp1_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores
    
    def experiment_2a(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['text', 'bbox'], use_zoom=True, add_rotation_transformation=False):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp2a_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp2a_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_2b(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['text', 'bbox'], use_zoom=True, add_rotation_transformation=True):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp2b_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp2b_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_3a(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['bbox'], use_zoom=True, add_rotation_transformation=False):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp3a_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp3a_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_3b(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['bbox'], use_zoom=True, add_rotation_transformation=True):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp3b_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp3b_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_4a(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['point', 'text'], use_zoom=True, add_rotation_transformation=False):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp4a_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp4a_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_4b(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['point', 'text'], use_zoom=True, add_rotation_transformation=True):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp4b_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp4b_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores
    

    def experiment_5a(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['point'], use_zoom=True, add_rotation_transformation=False):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp5a_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp5a_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_5b(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['point'], use_zoom=True, add_rotation_transformation=True):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp5b_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp5b_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_6a(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['text'], use_zoom=True, add_rotation_transformation=False):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }


        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp6a_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp6a_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores


    def experiment_6b(self, args, datasets=['0007', '0018', '0020', '0021', '0023'] , prompts=['text'], use_zoom=True, add_rotation_transformation=True):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                  which is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                  multiple categories in the same space. 
        """

        if datasets is None:
            datasets = ['0000', '0002', '0003', '0005', '0006', '0007', '0008', '0009', '0010', 
                        '0012', '0013', '0015', '0016', '0017', '0018', '0019', '0020', 
                        '0021', '0022', '0023', '0024']

        target_organs = [
            'Aorta', 'Colon cancer', 'Esophagus', 'Gallbladder', 'Inferior vena cava', 'Left adrenal gland', 
            'Left kidney', 'Liver', 'Liver tumor', 'Lung tumor', 'Pancreas', 'Portal/splenic vein', 
            'Right adrenal gland', 'Right kidney', 'Spleen', 'Stomach'
        ]

        organ_mapping = {
            'Aorta': ['aorta', 'Aorta', 'arota'],
            'Colon cancer': ['colon cancer', 'Colon cancer'],
            'Esophagus': ['esophagus', 'Esophagus', 'Esophagus_S', 'esophagus'],
            'Gallbladder': ['gall bladder', 'gallbladder', 'Gallbladder', 'gallbladder'],
            'Inferior vena cava': ['inferior vena cava', 'postcava', 'Inferior vena cava', 'inferior_vena_cava', 'venacava'],
            'Left adrenal gland': ['left adrenal gland', 'Left adrenal gland', 'adrenal_gland_left', 'leftsurretumor', 'leftsurrenalgland'],
            'Left kidney': ['left kidney', 'leftkidney', 'kidney_left', 'Kidney (L)'],
            'Liver': ['liver', 'Liver', 'livercyst', 'liverkyst', 'liverkyste'],
            'Liver tumor': ['livertumor', 'livertumor01', 'livertumor02', 'livertumor03', 'livertumor04', 'livertumor05', 'livertumor06', 'livertumor07', 'livertumor1', 'livertumor2', 'livertumors', 'Liver tumor'],
            'Lung tumor': ['lung tumors', 'Lung tumor', 'lung tumours', 'Lung tumours', 'lung tumours'],
            'Pancreas': ['pancreas', 'Pancreas', 'pancreatic-lesion'],
            'Portal/splenic vein': ['portal vein and splenic vein', 'portalvein', 'portalvein1', 'Portal/splenic vein', 'portal_vein_and_splenic_vein'],
            'Right adrenal gland': ['right adrenal gland', 'Right adrenal gland', 'adrenal_gland_right', 'rightsurretumor', 'rightsurrenalgland'],
            'Right kidney': ['right kidney', 'rightkidney', 'kidney_right', 'Kidney (R)'],
            'Spleen': ['spleen', 'Spleen'],
            'Stomach': ['stomach', 'Stomach'],
            'Bladder': ['bladder', 'Bladder', 'urinary_bladder'],
            'Bone': ['bone', 'Bone', 'Bone_Mandible'],
            'Brain': ['brain', 'Brain', 'Brainstem'],
            'Colon': ['colon', 'Colon'],
            'Cervical spine': ['cervical spine C1', 'cervical spine C2', 'cervical spine C3', 'cervical spine C4', 'cervical spine C5', 'cervical spine C6', 'cervical spine C7'],
            'Thoracic spine': ['thoracic spine T1', 'thoracic spine T2', 'thoracic spine T3', 'thoracic spine T4', 'thoracic spine T5', 'thoracic spine T6', 'thoracic spine T7', 'thoracic spine T8', 'thoracic spine T9', 'thoracic spine T10', 'thoracic spine T11', 'thoracic spine T12', 'additional 13th thoracic vertebra, T13'],
            'Lumbar spine': ['lumbar spine L1', 'lumbar spine L2', 'lumbar spine L3', 'lumbar spine L4', 'lumbar spine L5', 'lumbar spine L6'],
            'Coccyx': ['cocygis'],
            'Sacrum': ['sacrum', 'Sacrum'],
            'Heart': ['heart', 'Heart', 'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium', 'heart_ventricle_left', 'heart_ventricle_right'],
            'Kidney': ['kidney', 'Kidney', 'kidneys'],
            'Kidney tumor': ['kidney tumor'],
            'Lung': ['lungs', 'left lung', 'leftlung', 'right lung', 'rightlung', 'lung_lower_lobe_left', 'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right'],
            'Trachea': ['trachea', 'Trachea'],
            'Duodenum': ['duodenum'],
            'Intestine': ['smallintestin', 'small_bowel'],
            'Optic nerves': ['OpticNrv_L', 'OpticNrv_R'],
            'Liver cyst': ['livercyst', 'liverkyst', 'liverkyste'],
            'Liver vessels': ['hepatic vessels'],
            'Tumor': ['tumour', 'tumor'],
            'Adrenal': ['Adrenal'],
            'Rectum': ['Rectum'],
            'Arytenoid': ['Arytenoid'],
            'Bone_Mandible': ['Bone_Mandible'],
            'BuccalMucosa': ['BuccalMucosa'],
            'Cavity_Oral': ['Cavity_Oral'],
            'Cochlea': ['Cochlea_L', 'Cochlea_R'],
            'Cricopharyngeus': ['Cricopharyngeus'],
            'Eye': ['Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR'],
            'Glnd_Lacrimal_L': ['Glnd_Lacrimal_L'],
            'Glnd_Lacrimal_R': ['Glnd_Lacrimal_R'],
            'Glnd_Submand_L': ['Glnd_Submand_L'],
            'Glnd_Submand_R': ['Glnd_Submand_R'],
            'Glnd_Thyroid': ['Glnd_Thyroid'],
            'Glottis': ['Glottis'],
            'Larynx_SG': ['Larynx_SG'],
            'Lips': ['Lips'],
            'OpticChiasm': ['OpticChiasm'],
            'Parotid_L': ['Parotid_L'],
            'Parotid_R': ['Parotid_R'],
            'Pituitary': ['Pituitary'],
            'SpinalCord': ['SpinalCord']
        }
        

        def get_standardized_name(name):
            for standard_name, aliases in organ_mapping.items():
                if name in aliases:
                    return standard_name
            return None

        # Initialize a dictionary to hold all the dice scores for each organ and dataset
        dice_scores = {organ: {ds: 0.0 for ds in datasets} for organ in target_organs}
        all_dice_scores = {organ: {ds: [] for ds in datasets} for organ in target_organs}

        checkpoint_name = args.model_path.split('/')[-1]
        csv_file = os.path.join(args.out_dir, f'dice_scores_exp6b_{checkpoint_name.split('.')[0]}.csv')
        experiment_name = f'exp6b_{args.model_type}'

        # Write the header to the CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Organ'] + datasets
            writer.writerow(header)

        for code in tqdm(datasets):
            loader_args = {
                'data_dir' : '/scratch-shared/scur1193/M3D-Seg/M3D_Seg',
                'dataset_codes': [code],
                'add_rotation': add_rotation_transformation
            }
            loader = self.get_test_loader(loader_args)

            organ_to_idx = {organ: idx for idx, organ in enumerate(self.categories)}
            all_organs_idx = {code: organ_to_idx}

            for item in tqdm(loader, desc=f'Processing dataset {code}'):
                ct, gt = item['image'], item['label']
                torch.cuda.empty_cache()
                gc.collect()

                for raw_organ_name in organ_to_idx.keys():
                    standard_name = get_standardized_name(raw_organ_name)
                    if standard_name and standard_name in target_organs:
                        dice_score, mask = self.inference(
                            ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom, cls_idx=organ_to_idx[raw_organ_name]
                        )
                        
                        print(f"Dice score: {dice_score} from {raw_organ_name}")
                        if dice_score is not None:  # Ensure dice_score is not None
                            dice_scores[standard_name][code] += dice_score
                            all_dice_scores[standard_name][code].append(dice_score)

                            # # Save the mask with dataset code, organ name, and an index
                            # mask_dir = f'masks/{code}/{standard_name}'
                            # os.makedirs(mask_dir, exist_ok=True)
                            # mask_idx = len(os.listdir(mask_dir))
                            # mask_path = os.path.join(mask_dir, f'{experiment_name}_mask_{mask_idx}.pt')
                            # torch.save(mask, mask_path)
                            # print(f'Mask saved to {mask_path}')

                del ct, gt, item

            for organ in target_organs:
                if dice_scores[organ][code] > 0:
                    dice_scores[organ][code] /= len(loader)

            # Save all individual dice scores to a file
            os.makedirs('All dice scores', exist_ok=True)
            with open(f'All dice scores/all_dice_scores_{experiment_name}_{code}.pkl', 'wb') as f:
                pickle.dump(all_dice_scores, f)

            # Update the CSV file with new data for the current dataset
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                data = {row[0]: row[1:] for row in reader}

            for organ in target_organs:
                if organ in data:
                    data[organ][header.index(code)-1] = dice_scores[organ][code]
                else:
                    data[organ] = [0.0] * (len(datasets))
                    data[organ][header.index(code)-1] = dice_scores[organ][code]

            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for organ in target_organs:
                    writer.writerow([organ] + data[organ])

        return dice_scores

def main(args):
    eval = Evaluator()
    
    experiment_mapping = {
        1: [eval.experiment_1],
        2: [eval.experiment_2a, eval.experiment_2b],
        3: [eval.experiment_3a, eval.experiment_3b],
        4: [eval.experiment_4a, eval.experiment_4b],
        5: [eval.experiment_5a, eval.experiment_5b],
        6: [eval.experiment_6a, eval.experiment_6b],

    }
    for experiment in experiment_mapping[args.experiment]:
        experiment(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific experiments.")
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--baseline", type=bool, default=False)
    args = parser.parse_args()
    if args.baseline:
        args.model_type = 'baseline'
    else:
        args.model_type = 'finetuned'

    main(args)
