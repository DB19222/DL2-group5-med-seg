import os
import torch
import argparse
from datetime import datetime
from .model import SegVolGroup5
from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from .utils.data_utils import get_loader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .adapted_vit import AdaptedViT, AdaptedViTBaseline

from torchsummary import summary
import time
from peft import LoraConfig, get_peft_model

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_model(model):
    # Freeze everything
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    # Set new patch_embedding to trainable
    for (
        name,
        param,
    ) in model.model.image_encoder.patch_embedding_equiv.named_parameters():
        if "position_embeddings" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Set adapter module to trainable
    for param in model.model.image_encoder.adapter.parameters():
        param.requires_grad = True

    return model


def altered_forward(self, image, text=None, boxes=None, points=None, **kwargs):
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
    sl_loss = self.supervised_forward(
        image,
        image_embedding,
        img_shape,
        kwargs["train_organs"],
        kwargs["train_labels"],
    )
    return sl_loss


def setup_model(pretrained_model, lora_config, input_size, baseline=False):
    # Instantiate Adapted Vision Transformer
    if baseline:
        adapted_ViT = AdaptedViTBaseline()
        print("USING BASELINE!!")
    else:
        adapted_ViT = AdaptedViT()

    print("Initital ViT : ")
    Initial_ViT = pretrained_model.model.image_encoder
    for param in Initial_ViT.parameters():
        param.requires_grad = True
    print(summary(Initial_ViT.to(global_device), input_size))

    # Get the parameters, and check which of the parameters coincide with our model's parameters.
    pretrained_state_dict = pretrained_model.model.image_encoder.state_dict()
    adapted_state_dict = adapted_ViT.state_dict()
    to_update = {
        module: weights
        for module, weights in pretrained_state_dict.items()
        if (module in adapted_state_dict.keys())
        and (adapted_state_dict[module].size() == weights.size())
    }

    print(f"Modules that are being updated : {to_update.keys()}")

    # Copy weights to our model
    adapted_state_dict.update(to_update)
    adapted_ViT.load_state_dict(adapted_state_dict)

    # Switch the image_encoders
    pretrained_model.model.image_encoder = adapted_ViT

    pretrained_model = freeze_model(pretrained_model)

    ViT = pretrained_model.model.image_encoder
    print("ViT before LoRA injection : ")
    print(summary(ViT.to(global_device), input_size))
    print(sum(p.numel() for p in ViT.parameters() if p.requires_grad))
    # Inject LoRA modules and insert in pretrained architecture
    ViT = get_peft_model(ViT, lora_config)
    pretrained_model.model.image_encoder = ViT

    # Just to be sure, freeze weights again.
    pretrained_model = freeze_model(pretrained_model)
    ViT = pretrained_model.model.image_encoder
    print("ViT after LoRA injection : ")
    # print(summary(ViT, input_size))
    print(sum(p.numel() for p in ViT.parameters() if p.requires_grad))
    print(
        [
            (name, param.numel())
            for name, param in ViT.named_parameters()
            if param.requires_grad
        ]
    )
    # Use our own copy of the model to be able to debug within the code used (remote code full of bugs)
    output_model = SegVolGroup5(
        image_encoder=pretrained_model.model.image_encoder,
        mask_decoder=pretrained_model.model.mask_decoder,
        prompt_encoder=pretrained_model.model.prompt_encoder,
        roi_size=pretrained_model.config.spatial_size,
        patch_size=pretrained_model.config.patch_size,
        text_encoder=pretrained_model.model.text_encoder,
        test_mode=False,
    )

    for name, param in output_model.named_parameters():
        if param.requires_grad:
            print(name)

    return output_model


def set_parse():
    """
    Parser that sets all the parameters (we use default)
    """
    parser = argparse.ArgumentParser()
    # %% set up parser
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--dataset_codes", type=list, default=["0007", "0023", "0021", "0018", "0020"]
    )
    # config
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument(
        "-infer_overlap",
        default=0.5,
        type=float,
        help="sliding window inference overlap",
    )
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple)
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple)
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("--clip_ckpt", type=str, default="./config/clip")
    parser.add_argument(
        "--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability"
    )
    parser.add_argument(
        "--RandScaleIntensityd_prob",
        default=0.1,
        type=float,
        help="RandScaleIntensityd aug probability",
    )
    parser.add_argument(
        "--RandShiftIntensityd_prob",
        default=0.1,
        type=float,
        help="RandShiftIntensityd aug probability",
    )
    parser.add_argument("-num_workers", type=int, default=8)
    parser.add_argument("--usecheckpoint", type=bool, default=True)
    parser.add_argument("--use_original", type=bool, default=False)
    parser.add_argument("--baseline", type=bool, default=False)

    # dist
    parser.add_argument(
        "--dist",
        dest="dist",
        type=bool,
        default=False,
        help="distributed training or not",
    )
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument("--init_method", type=str, default="env://")
    parser.add_argument(
        "--bucket_cap_mb",
        type=int,
        default=25,
        help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)",
    )
    # key params
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-weight_decay", type=float, default=1e-5)
    parser.add_argument("-warmup_epoch", type=int, default=10)
    parser.add_argument("-num_epochs", type=int, default=4)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("--use_pseudo_label", default=True, type=bool)
    args = parser.parse_args()

    return args


def train_epoch(
    args, segvol_model, train_dataloader, optimizer, scheduler, epoch, iter_num
):

    start = time.time()
    epoch_loss = 0
    epoch_sl_loss = 0

    epoch_iterator = tqdm(train_dataloader, desc="[RANK]", dynamic_ncols=True)

    for batch in epoch_iterator:
        image, gt3D = batch["image"].cuda(), batch["post_label"].cuda()
        organ_name_list = batch["organ_name_list"]

        loss_step_avg = 0
        sl_loss_step_avg = 0

        for cls_idx in range(len(organ_name_list)):
            optimizer.zero_grad()

            organs_cls = organ_name_list[cls_idx]
            labels_cls = gt3D[:, cls_idx]

            if torch.sum(labels_cls) == 0:
                print(f"[RANK] ITER-{iter_num} --- No object, skip iter")
                continue

            segvol_model = segvol_model.to("cuda")

            input_dict = {
                "image": image,
                "train_organs": organs_cls,
                "train_labels": labels_cls,
            }

            sl_loss = segvol_model(**input_dict)

            if args.use_pseudo_label:
                loss = sl_loss
                sl_loss_step_avg += sl_loss.item()

            loss_step_avg += loss.item()

            loss.backward()
            optimizer.step()
            print(
                f"[RANK] ITER-{iter_num} --- loss {loss.item()}, sl_loss, {sl_loss.item()}."
            )
            iter_num += 1

        loss_step_avg /= len(organ_name_list)
        sl_loss_step_avg /= len(organ_name_list)
        print(f"[RANK] AVG loss {loss_step_avg}, sl_loss, {sl_loss_step_avg}.")

        epoch_loss += loss_step_avg
        epoch_sl_loss += sl_loss_step_avg

    scheduler.step()

    epoch_loss /= len(train_dataloader) + 1e-12
    epoch_sl_loss /= len(train_dataloader) + 1e-12
    print("epoch_loss: {}".format(epoch_loss))
    print(f"{epoch} took {time.time() - start} seconds.")
    return epoch_loss, iter_num


def main_worker(args):
    # Load the original model
    clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
    original_model = AutoModel.from_pretrained(
        "BAAI/SegVol", trust_remote_code=True, test_mode=False
    )
    original_model.model.text_encoder.tokenizer = clip_tokenizer

    # Get LoRA configuration.
    # We use the traditional LoRA config, and only finetune the feedforward dense layer due to ease of implementations.
    # We also apply a slight dropout due to risk of overfitting
    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["out_proj"], lora_dropout=0.1, bias="none"
    )
    segvol_model = original_model
    segvol_model = setup_model(
        segvol_model, lora_config, (1, 32, 256, 256), args.baseline
    )

    optimizer = torch.optim.AdamW(
        segvol_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.num_epochs
    )

    if args.usecheckpoint:
        statedict = torch.load(args.model_path)
        scheduler_dict = statedict["scheduler"]
        model_dict = statedict["model"]
        optimizer_dict = statedict["optimizer"]

        segvol_model.load_state_dict(model_dict, strict=False)
        # segvol_model = freeze_model(segvol_model)

        for name, param in segvol_model.named_parameters():
            if param.requires_grad:
                print(name)

        optimizer.load_state_dict(optimizer_dict)
        # scheduler.load_state_dict(scheduler_dict)

        print("Succesfully loaded everything!")

    if args.use_original:
        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        segvol_model = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=False
        )
        segvol_model.model.text_encoder.tokenizer = clip_tokenizer
        segvol_model.train()
        print("Loaded original SegVol model")

    # %% train
    num_epochs = args.num_epochs
    iter_num = 0

    train_dataloader = get_loader(args)

    start_epoch = 0
    segvol_model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss, iter_num = train_epoch(
            args, segvol_model, train_dataloader, optimizer, scheduler, epoch, iter_num
        )

        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        # save the model checkpoint
        if (epoch + 1) % 4 == 0:
            checkpoint = {
                "model": segvol_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scheduler": scheduler.state_dict(),
                "epoch_loss": epoch_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(args.model_save_path, f"medsam_e{epoch+1}_LrReinit.pth"),
            )


def main():
    # set seeds
    torch.manual_seed(2023)
    torch.cuda.empty_cache()
    args = set_parse()
    args.resume = "../Data/SegVol/weights/SegVol_v1.pth"
    args.num_workers = 1
    args.batch_size = 1
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")

    print(args.dataset_codes)
    # segvol model load
    model_save_path = os.path.join(args.work_dir)
    args.model_save_path = model_save_path

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12222"
    if args.use_pseudo_label:
        print("----- use pseudo_label -----")
    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    print("Spwaning processces, ngpus_per_node={}".format(ngpus_per_node))
    print(torch.cuda.device_count())
    print(args)
    print(f"=====> project save at {args.model_save_path}")
    # mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))
    main_worker(args)


if __name__ == "__main__":
    main()
