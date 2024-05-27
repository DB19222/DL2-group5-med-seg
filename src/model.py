import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
import random
from .utils.monai_inferers_utils import select_points, generate_box
from .utils.loss import BCELoss, BinaryDiceLoss

from typing import Tuple, List
from types import MethodType


# Coppied remote code and fixed bugs to make code work
def predict_masks_group5(
    self,
    image_embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts masks. See 'forward' for more details."""
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(
        sparse_prompt_embeddings.size(0), -1, -1
    )
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
    # Expand per-image data in batch direction to be per-mask
    if image_embeddings.shape[0] != tokens.shape[0]:
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    else:
        src = image_embeddings

    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w, d = src.shape

    # Run the transformer
    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w, d)
    upscaled_embedding = self.output_upscaling(src)
    hyper_in_list: List[torch.Tensor] = []
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
        )
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w, d = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d)

    if text_embedding is not None:
        text_embedding_down = self.txt_align_upscaled_embedding(
            text_embedding
        ).unsqueeze(dim=1)
        upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
        sim = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)
        sim = sim.repeat(1, masks.shape[1], 1, 1, 1)
        masks = masks + sim
    iou_pred = self.iou_prediction_head(iou_token_out)

    return masks, iou_pred


# Create new model that includes the fixed bugs
class SegVolGroup5(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        roi_size,
        patch_size,
        text_encoder,
        test_mode,
        processor=None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        # Include the function above, with fixed bugs
        self.mask_decoder.predict_masks = MethodType(
            predict_masks_group5, self.mask_decoder
        )
        self.prompt_encoder = prompt_encoder
        self.text_encoder = text_encoder
        self.feat_shape = np.array(roi_size) / np.array(patch_size)
        self.test_mode = test_mode
        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()
        self.decoder_iter = 6

    def forward(self, image, text=None, boxes=None, points=None, **kwargs):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)
        image_embedding = image_embedding.transpose(1, 2).reshape(
            bs,
            -1,
            int(self.feat_shape[0]),
            int(self.feat_shape[1]),
            int(self.feat_shape[2]),
        )
        # test mode
        if self.test_mode:
            return self.forward_decoder(image_embedding, img_shape, text, boxes, points)

        # train mode
        ## sl
        sl_loss = self.supervised_forward(
            image,
            image_embedding,
            img_shape,
            kwargs["train_organs"],
            kwargs["train_labels"],
        )
        ## ssl
        # ssl_loss = self.unsupervised_forward(image, image_embedding, kwargs['pseudo_seg_cleaned'], img_shape)
        return sl_loss

    def forward_decoder(
        self, image_embedding, img_shape, text=None, boxes=None, points=None
    ):
        device = image_embedding.device
        with torch.no_grad():
            if boxes is not None:
                if len(boxes.shape) == 2:
                    boxes = boxes[:, None, :]  # (B, 1, 6)
            if text is not None:
                text_embedding = self.text_encoder(text, device)  # (B, 768)
            else:
                text_embedding = None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=text_embedding,
        )

        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding=text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        logits = F.interpolate(
            low_res_masks, size=img_shape, mode="trilinear", align_corners=False
        )
        return logits

    def supervised_forward(
        self, image, image_embedding, img_shape, training_organs, train_labels
    ):
        device = image_embedding.device
        iter_points, iter_bboxes, iter_organs = self.build_prompt_label(
            image.shape[0], training_organs, train_labels, device
        )
        # select prompt
        prompt_options = [
            [None, iter_points, iter_organs],
            [iter_bboxes, None, iter_organs],
            [None, None, iter_organs],
            [iter_bboxes, None, None],
            [None, iter_points, None],
            [iter_bboxes, iter_points, None],
        ]
        sl_loss = 0
        for prompt in prompt_options:
            bboxes, points, organs = prompt
            logits = self.forward_decoder(
                image_embedding, img_shape, text=organs, boxes=bboxes, points=points
            )
            # cal loss
            sl_loss_dice = self.dice_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss_bce = self.bce_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss

    def build_prompt_label(self, bs, training_organs, train_labels, device):
        # generate prompt & label
        iter_organs = []
        iter_bboxes = []
        iter_points_ax = []
        iter_point_labels = []
        for sample_idx in range(bs):
            # organ prompt
            iter_organs.append(training_organs)
            # box prompt
            box = generate_box(train_labels[sample_idx], bbox_shift=10)
            iter_bboxes.append(box)
            # point prompt
            num_positive_extra_max, num_negative_extra_max = 10, 10
            num_positive_extra = random.randint(0, num_positive_extra_max)
            num_negative_extra = random.randint(0, num_negative_extra_max)
            point, point_label = select_points(
                train_labels[sample_idx],
                num_positive_extra=num_positive_extra,
                num_negative_extra=num_negative_extra,
                fix_extra_point_num=num_positive_extra_max + num_negative_extra_max,
            )
            iter_points_ax.append(point)
            iter_point_labels.append(point_label)
        # batched prompt
        iter_points_ax = torch.stack(iter_points_ax, dim=0).to(device)
        iter_point_labels = torch.stack(iter_point_labels, dim=0).to(device)
        iter_points = (iter_points_ax, iter_point_labels)
        iter_bboxes = torch.stack(iter_bboxes, dim=0).float().to(device)
        return iter_points, iter_bboxes, iter_organs


# %% set up model
class SegVol(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        roi_size,
        patch_size,
        text_encoder,
        # clip_model,
        test_mode=False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.mask_decoder.predict_masks = MethodType(
            predict_masks_group5, self.mask_decoder
        )
        self.prompt_encoder = prompt_encoder
        self.text_encoder = text_encoder
        self.feat_shape = np.array(roi_size) / np.array(patch_size)
        self.test_mode = test_mode
        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()
        self.decoder_iter = 6

    def forward(self, image, text=None, boxes=None, points=None, **kwargs):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)
        image_embedding = image_embedding.transpose(1, 2).reshape(
            bs,
            -1,
            int(self.feat_shape[0]),
            int(self.feat_shape[1]),
            int(self.feat_shape[2]),
        )
        # test mode
        if self.test_mode:
            return self.forward_decoder(image_embedding, img_shape, text, boxes, points)

        # train mode
        ## sl
        sl_loss = self.supervised_forward(
            image,
            image_embedding,
            img_shape,
            kwargs["train_organs"],
            kwargs["train_labels"],
        )
        ## ssl
        # ssl_loss = self.unsupervised_forward(image, image_embedding, kwargs['pseudo_seg_cleaned'], img_shape)
        return sl_loss

    def forward_decoder(
        self, image_embedding, img_shape, text=None, boxes=None, points=None
    ):
        device = image_embedding.device
        with torch.no_grad():
            if boxes is not None:
                if len(boxes.shape) == 2:
                    boxes = boxes[:, None, :]  # (B, 1, 6)
            if text is not None:
                text_embedding = self.text_encoder(text, device)  # (B, 768)
            else:
                text_embedding = None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=text_embedding,
        )

        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding=text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        logits = F.interpolate(
            low_res_masks, size=img_shape, mode="trilinear", align_corners=False
        )
        return logits

    def supervised_forward(
        self, image, image_embedding, img_shape, training_organs, train_labels
    ):
        device = image_embedding.device
        iter_points, iter_bboxes, iter_organs = self.build_prompt_label(
            image.shape[0], training_organs, train_labels, device
        )
        # select prompt
        prompt_options = [
            [None, iter_points, iter_organs],
            [iter_bboxes, None, iter_organs],
            [None, None, iter_organs],
            [iter_bboxes, None, None],
            [None, iter_points, None],
            [iter_bboxes, iter_points, None],
        ]
        sl_loss = 0
        for prompt in prompt_options:
            bboxes, points, organs = prompt
            logits = self.forward_decoder(
                image_embedding, img_shape, text=organs, boxes=bboxes, points=points
            )
            # cal loss
            sl_loss_dice = self.dice_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss_bce = self.bce_loss.forward(
                logits.squeeze().float(), train_labels.squeeze().float()
            )
            sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss

    def build_prompt_label(self, bs, training_organs, train_labels, device):
        # generate prompt & label
        iter_organs = []
        iter_bboxes = []
        iter_points_ax = []
        iter_point_labels = []
        for sample_idx in range(bs):
            # organ prompt
            iter_organs.append(training_organs)
            # box prompt
            box = generate_box(train_labels[sample_idx], bbox_shift=10)
            iter_bboxes.append(box)
            # point prompt
            num_positive_extra_max, num_negative_extra_max = 10, 10
            num_positive_extra = random.randint(0, num_positive_extra_max)
            num_negative_extra = random.randint(0, num_negative_extra_max)
            point, point_label = select_points(
                train_labels[sample_idx],
                num_positive_extra=num_positive_extra,
                num_negative_extra=num_negative_extra,
                fix_extra_point_num=num_positive_extra_max + num_negative_extra_max,
            )
            iter_points_ax.append(point)
            iter_point_labels.append(point_label)
        # batched prompt
        iter_points_ax = torch.stack(iter_points_ax, dim=0).to(device)
        iter_point_labels = torch.stack(iter_point_labels, dim=0).to(device)
        iter_points = (iter_points_ax, iter_point_labels)
        iter_bboxes = torch.stack(iter_bboxes, dim=0).float().to(device)
        return iter_points, iter_bboxes, iter_organs


class TextEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names):
        text_list = [
            "A computerized tomography of a {}.".format(organ_name)
            for organ_name in organ_names
        ]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].cuda()
        return tokens

    def forward(self, text):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        tokens = self.organ2tokens(text)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)
        return text_embedding
