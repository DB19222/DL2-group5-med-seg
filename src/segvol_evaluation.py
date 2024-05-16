from transformers import AutoModel, AutoTokenizer
from lib.SegVol.data_utils import BatchedDistributedSampler, MinMaxNormalization, DimTranspose
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
        self.model = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=True)
        self.model.model.text_encoder.tokenizer = self.clip_tokenizer
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
        if args['randrotate']:
            self.test_transform = transforms.Compose(
                [
                    transforms.RandRotated(keys=["image", "label"], range_x=0.5, range_y=0.5, range_z=0.5, prob=1.0)
                ]
            )
        else:
            self.test_transform = transforms.Compose(
                [
                   transforms.RandRotated(keys=["image", "label"], range_x=30, range_y=30, range_z=30, prob=0.0)
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

    
    def experiment_1(self, datasets=['0007', '0023', '0021', '0018', '0020'], prompts=['text', 'bbox'], use_zoom=True):
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
                 

    
    def experiment_2(self, datasets=['0014'], prompts=['text', 'point'], use_zoom=True):
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
            'randrotate' : False
        }
        loader = self.get_test_loader(args)

        print(self.categories)
        for item in loader:
            ct, gt = item['image'], item['label']
            print(ct.squeeze(0).shape, gt.squeeze(0).shape)
            self.inference(ct.squeeze(0), gt.squeeze(0), prompts=prompts, use_zoom=use_zoom)

        # for dataset in datasets:
        #     data_path = os.path.join(os.curdir, 'data', 'datasets', dataset, f'{dataset}.json')
        #     with open(data_path, 'r') as rf:
        #         data_json = json.loads(rf.read())

        #     test_data = data_json['test']

        #     dataset_path = os.path.join(os.curdir, 'data', 'datasets')
        #     for item in test_data:
        #         ct_path = os.path.join(dataset_path, item['image'])
        #         gt_path = os.path.join(dataset_path, item['label'])

        #         categories_dict = data_json['labels']
        #         self.categories = [x for _, x in categories_dict.items() if x != "background"]

        #         ct_npy, gt_npy = self.model.processor.load_uniseg_case(ct_path, gt_path)

        #         self.inference(ct_npy, gt_npy)


            



    def experiment_3(self, datasets=['0001'], prompts=[], use_zoom=True):
        """ 
            Additionally, the authors discuss the generalization performance of SegVol by 
            applying it on an external MRI dataset (CHAOS).

            NOTE : The huggingface dataset contains the CT data, but not the MRI data. So if we want to perform this experiment, 
            we would have to pre-process the dataset ourselves. 

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
