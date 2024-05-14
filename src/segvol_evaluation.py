from transformers import AutoModel, AutoTokenizer
from lib.SegVol.data_utils import BatchedDistributedSampler, MinMaxNormalization, DimTranspose
import math
import os
import numpy as np
import torch
from monai import data, transforms
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset
import os
import ast
from scipy import sparse
import random
from scipy.ndimage import binary_opening, binary_closing
from scipy.ndimage import label as label_structure
from scipy.ndimage import sum as sum_structure
import json
import json
from monai import transforms
import os
import torch 

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
    def __init__(self, data_dir, data, transform, organ_list):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform

        print(organ_list)
        organ_list.remove('background')
        self.target_list = organ_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get path
        item_dict = self.data[idx]
        ct_path, gt_path = os.path.join(self.data_dir, item_dict['image']), os.path.join(self.data_dir, item_dict['label'])

        return ct_path, gt_path
    

class Evaluator:

    def __init__(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        print("Loading model...")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=True)
        self.model.model.text_encoder.tokenizer = self.clip_tokenizer
        self.model.to(device)
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

            universal_ds = SegVolDataset(data_dir=root_path, data=datalist, transform=transform, organ_list=list(dataset_dict['labels'].values()))
            concat_dataset.append(universal_ds)
            CombinationDataset_len += len(universal_ds)

        self.categories = universal_ds.target_list

        print(f'CombinationDataset loaded, dataset size: {CombinationDataset_len}')
        return UnionDataset(ConcatDataset(concat_dataset), concat_dataset)


    def get_test_loader(self, args):
        if args['randrotate']:
            test_transform = transforms.Compose(
                [
                    transforms.RandRotate(range_x=180, range_y=180, range_z=180, prob=1.0)
                ]
            )
        else:
            test_transform = transforms.Compose(
                [
                    transforms.RandRotate(range_x=180, range_y=180, range_z=180, prob=0.0)
                ]
            )

        print(f'----- test combination dataset -----')

        combination_train_ds = self.build_concat_dataset(root_path=args['data_dir'], dataset_codes=args['dataset_codes'], transform=test_transform)

        train_sampler = None 

        loader = data.DataLoader(
            combination_train_ds,
            batch_size=1,
            shuffle=(train_sampler is None),
            num_workers=1,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True
        )
        return loader


    
    def inference(self, ct_npy, gt_npy, cls_idx=0):
        # go through zoom_transform to generate zoomout & zoomin views
        data_item = self.model.processor.zoom_transform(ct_npy, gt_npy)

        # add batch dim manually
        data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
        data_item['image'].unsqueeze(0).to(self.device), data_item['label'].unsqueeze(0).to(self.device), data_item['zoom_out_image'].unsqueeze(0).to(self.device), data_item['zoom_out_label'].unsqueeze(0).to(self.device)

        # text prompt
        text_prompt = [self.categories[cls_idx]]

        # point prompt
        point_prompt, point_prompt_map = self.model.processor.point_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=self.device)   # inputs w/o batch dim, outputs w batch dim

        # bbox prompt
        bbox_prompt, bbox_prompt_map = self.model.processor.bbox_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=self.device)   # inputs w/o batch dim, outputs w batch dim

        print('prompt done')

        # segvol test forward
        # use_zoom: use zoom-out-zoom-in
        # point_prompt_group: use point prompt
        # bbox_prompt_group: use bbox prompt
        # text_prompt: use text prompt
        logits_mask = self.model.forward_test(image=data_item['image'],
            zoomed_image=data_item['zoom_out_image'],
            # point_prompt_group=[point_prompt, point_prompt_map],
            # bbox_prompt_group=[bbox_prompt, bbox_prompt_map],
            text_prompt=text_prompt,
            use_zoom=True
            )

        # cal dice score
        dice = self.model.processor.dice_score(logits_mask[0][0], data_item['label'][0][cls_idx], self.device)
        print(dice)


    
    def experiment_1(self, datasets=['0007', '0023', '0021', '0018', '0020'], prompts=['text', 'bbox', 'point'], use_zoom=True):
        """ 
            Internal validation experiment in which task-specific segmentation models are compared
            with the generally trained model SegVol. 

            "The 10 internal segmentation tasks are selected from BTCV [32] and MSD- spleen [58] datasets, which focus on organ segmentation
            and from MSD-lung, MSD-colon, and MSD-liver datasets, which focus on lesion segmentation."

            Claims : 
                - SegVol trained on 25 datasets outperforms task-specific segmentation models.
                - Exhibits narrow DSC distribution, indicating robustness and generalization ability.
                - Massive generative pretraining on unlabeled data endows SegVol with a complete understanding of the volumetric structures,
                whcih is superior to learning from a small number of samples. 
                - Learning from joint datasets with semantic prompts makes SegVol generalize better to unseen data (can learn from kidney, and left-kidney)
                - Spatial point/bbox prompts provide a precise spatial reference and help disambiguate the overlap of 
                multiple categories in the same space. 

            \cite{du2024}

        """

        # dice = self.model.processor.dice_score(logits_mask[0][0], data_item['label'][0][cls_idx], device)
                    

    
    def experiment_2(self, datasets=['0002'], prompts=['text', 'bbox'], use_zoom=True):
        """ 
            External validation experiment where SegVol is compared with interactive methods such as SAMMED-3D

            "To compare with these interactive segmentation models, we per- formed external validation experiments 
            on 1,738 cases from the validation set of AMOS22 [26] and the whole novel annotated set of Universal
            Lesion Segmentation Challenge 23(ULS23) [9]."

            NOTE : The ULS dataset is not included in the opensource data provided, so we only validate 0002. 

            Claims : 
                - 
                - 
                - 

        """

        args = {
            'data_dir' : os.path.join(os.curdir, 'data', 'datasets'),
            'dataset_codes' : datasets,
            'randrotate' : True

        }
        loader = self.get_test_loader(args)

        print(self.categories)
        for batch in loader:
            ct, gt = batch 
            print(ct, gt)
            ct_npy, gt_npy = self.model.processor.load_uniseg_case(ct[0], gt[0])

            self.inference(ct_npy, gt_npy)


    def experiment_3(self, datasets=['0001'], prompts=[], use_zoom=True):
        """ 
            Additionally, the authors discuss the generalization performance of SegVol by 
            applying it on an external MRI dataset (CHAOS).

            NOTE : This dataset is contained as CT scan data, but not as MRI data. So if we want to perform this experiment, 
            we would have to generate the dataset ourselves. 

            Claims :
                - "It achieves median Dice scores of 85.70%, 80.09%, 80.04%, and 81.46% for liver, spleen, left kidney, and right kidney, respectively."
                - This demonstrates robustness of SegVol in the face of completely unseen modality data. 
        """
        pass 

    
    def experiment_4(self, datasets=[], prompts=[], use_zoom=True):
        """ 
            Experiment that analyzes the relationship between spatial-prompt, and semantic prompts. 

            NOTE : It is not entirely clear which datasets are used in this experiment

            "In Fig. 5 a, we quantita- tively analyze the mutually supportive relationship between semantic-prompt and spatial-prompt in 
            19 internal segmen- tation tasks."

            We can best select a few organs to perform this analysis. 

            Claims :
                - Semantic prompts help mititage the multiple plausible outputs problem in the spatial prompt setting. 
        
        """
        pass

    def experiment_5(self, datasets=[], prompts=[], use_zoom=True):
        """ 
            Experiment that studies the possibility of SegVol to reflect spatial prompts to semantic categories. 

            "We implement this reflection experiment by decoding the semantic prompts from 
            a category set and applying the softmax function among the logits of semantic 
            prompts on the predicted mask voxels to get the prediction probabilities of different categories."

            Claims : 
                - SegVol can give accurate semantic categories based on the spatial prompts, 

        
        """
        pass

    def experiment_6(self, datasets=[], prompts=[], use_zoom=True):
        """ 
            Experiment that expands the scale of the training set, showing that model performance 
            scales with dataset size. 

            Claims : 
                - 


        
        """
        pass




def main():
    eval = Evaluator()
    eval.experiment_2()


if __name__ == '__main__':
    main()