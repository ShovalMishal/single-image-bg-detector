import os
from enum import Enum
import torch
import torchvision
from PIL import Image
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import single_image_bg_detector.dino.vision_transformer as vits
from torchvision import transforms as pth_transforms
import skimage.io
import cv2
from .bg_subtractor_utils import pad_image_to_divisible, calculate_patches_alg_heat_maps_for_k_values, plot_k_heat_maps, \
    plot_heat_map, ViTMode, ViTModelType, normalize_array, create_gt_attention
from .dino.visualize_attention import display_instances

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BGSubtractionWithDinoVit:
    def __init__(self, target_dir, vit_patch_size, vit_arch, vit_image_size, dota_obj, threshold,
                 pretrained_weights="", checkpoint_key="", model_mode=ViTMode.CLS_SELF_ATTENTION.value,
                 model_type=ViTModelType.DINO_MC.value):
        self.model = None
        self.model_type = model_type
        self.model_mode = model_mode
        self.vit_patch_size = vit_patch_size
        self.vit_image_size = vit_image_size
        self.target_dir = os.path.join(target_dir, model_type)
        os.makedirs(self.target_dir, exist_ok=True)
        if self.model_mode == ViTMode.CLS_SELF_ATTENTION.value:
            self.target_dir = os.path.join(self.target_dir, ViTMode.CLS_SELF_ATTENTION.value)
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            self.target_dir = os.path.join(self.target_dir, ViTMode.LAST_BLOCK_OUTPUT.value)
            os.makedirs(self.target_dir, exist_ok=True)
        self.dota_obj = dota_obj
        self.threshold = threshold
        self.init_model(vit_arch, vit_patch_size, model_type, pretrained_weights=pretrained_weights,
                        checkpoint_key=checkpoint_key)

    def init_model(self, vit_arch, vit_patch_size, model_type, pretrained_weights="", checkpoint_key=""):
        self.model = vits.__dict__[vit_arch](patch_size=vit_patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(device)
        if model_type == ViTModelType.DINO_VIT.value:
            url = None
            if vit_arch == "vit_small" and vit_patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif vit_arch == "vit_small" and vit_patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif vit_arch == "vit_base" and vit_patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif vit_arch == "vit_base" and vit_patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                self.model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")
        else:
            # dino-mc case
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            # print(state_dict['teacher']['module.head.last_layer.weight_v'])
            # print(state_dict.keys())
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # print(state_dict.keys())
            # if model_name != 'vit_small':
            #     state_dict = {k.replace("head", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    def run_on_image_tensor(self, img:torch.tensor):
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % self.vit_patch_size, img.shape[2] - img.shape[2] % self.vit_patch_size
        img = img[:, :w, :h].unsqueeze(0)
        w_featmap = img.shape[-2] // self.vit_patch_size
        h_featmap = img.shape[-1] // self.vit_patch_size
        if self.model_mode == ViTMode.CLS_SELF_ATTENTION.value:
            with torch.no_grad():
                attentions = self.model.get_last_selfattention(img.to(device))
            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = \
                nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.vit_patch_size, mode="bilinear")[
                    0].cpu().numpy()

            # add average attention
            attentions = np.concatenate((attentions, np.mean(attentions, axis=0)[None, :]), axis=0)
            return attentions[-1]
        elif self.model_mode == ViTMode.LAST_BLOCK_OUTPUT.value:
            output = self.model.get_last_block(img.to(device))[1:, :].cpu()
            k_to_heat_map = calculate_patches_alg_heat_maps_for_k_values(heat_map_width=w_featmap,
                                                                         heat_map_height=h_featmap,
                                                                         flattened_pathces_matrix=output)


            returned_heat_map = k_to_heat_map[20].reshape((w_featmap, h_featmap)) / k_to_heat_map[20].reshape(
                (w_featmap, h_featmap)).max()
            returned_heat_map = \
                nn.functional.interpolate(torch.tensor(returned_heat_map).unsqueeze(dim=0).unsqueeze(dim=0),
                                          scale_factor=self.vit_patch_size,
                                          mode="nearest")[0][0]


            return returned_heat_map.numpy()

    def run_on_image_path(self, imgid, plot_result=False):
        anns = self.dota_obj.loadAnns(imgId=imgid)
        # gt_attention = self.create_gt_attention(anns=anns)
        image_path = os.path.join(self.dota_obj.imagepath, imgid + '.png')
        if os.path.isfile(image_path):
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                orig_img = img.convert('RGB')
        if self.model_type == ViTModelType.DINO_VIT.value:
            transform = pth_transforms.Compose([
                pth_transforms.Resize(self.vit_image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            transform = pth_transforms.Compose([
                pth_transforms.Resize(256, interpolation=3),
                # pth_transforms.CenterCrop(224),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        scale_factor = self.vit_image_size[0] / orig_img.size[0]
        gt_attention = create_gt_attention(anns=anns, scale_factor=scale_factor, image_size=self.vit_image_size)
        transform_to_view_original = pth_transforms.Compose([
            pth_transforms.Resize(self.vit_image_size),
            pth_transforms.ToTensor()
        ])
        img = transform(orig_img)
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % self.vit_patch_size, img.shape[2] - img.shape[2] % self.vit_patch_size
        img = img[:, :w, :h].unsqueeze(0)
        orig_img = transform_to_view_original(orig_img)[:, :w, :h].permute(1, 2, 0)
        # RGB to BGR
        orig_img = orig_img[:, :, torch.tensor([2, 1, 0])]

        w_featmap = img.shape[-2] // self.vit_patch_size
        h_featmap = img.shape[-1] // self.vit_patch_size
        if self.model_mode == ViTMode.CLS_SELF_ATTENTION.value:
            attentions = self.model.get_last_selfattention(img.to(device))
            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            if self.threshold is not None:
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - self.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = \
                    nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=self.vit_patch_size, mode="nearest")[
                        0].cpu().numpy()

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = \
                nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.vit_patch_size, mode="nearest")[
                    0].cpu().numpy()
            # add average attention
            attentions = np.concatenate((attentions, np.mean(attentions, axis=0)[None, :]), axis=0)
            # save attentions heatmaps
            if plot_result:
                os.makedirs(os.path.join(self.target_dir, imgid), exist_ok=True)
                torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                             os.path.join(self.target_dir, imgid, "img.png"))
                for j in range(nh + 1):
                    fname = os.path.join(self.target_dir, imgid, "attn-head" + str(j) + ".png")
                    if j == nh:
                        fname = os.path.join(self.target_dir, imgid, "average_attention.png")
                    plt.imsave(fname=fname, arr=attentions[j], format='png')
                    plot_heat_map(heatmap=attentions[j],
                                  imgid=imgid, target_dir=self.target_dir, dota_obj=self.dota_obj, anns=anns,
                                  orig_image_size=orig_img.shape, orig_image=np.flipud(orig_img),
                                  filenames=[f"attention_map_head_{j}",
                                             f"attention_map_head_{j}_overlaid_on_image_{imgid}"])
                    print(f"{fname} saved.")

            if self.threshold is not None and plot_result:
                image = skimage.io.imread(os.path.join(self.target_dir, imgid, "img.png"))
                for j in range(nh):
                    display_instances(image, th_attn[j], fname=os.path.join(self.target_dir, imgid,
                                                                            "mask_th" + str(
                                                                                self.threshold) + "_head" + str(
                                                                                j) + ".png"), blur=False)
            return attentions[-1], gt_attention

        elif self.model_mode == ViTMode.LAST_BLOCK_OUTPUT.value:
            output = self.model.get_last_block(img.to(device))[1:, :].cpu()
            k_to_heat_map = calculate_patches_alg_heat_maps_for_k_values(heat_map_width=w_featmap,
                                                                         heat_map_height=h_featmap,
                                                                         flattened_pathces_matrix=output)
            if plot_result:
                os.makedirs(os.path.join(self.target_dir, imgid), exist_ok=True)
                plot_k_heat_maps(k_to_heat_map=k_to_heat_map, heat_map_width=w_featmap, heat_map_height=h_featmap,
                                 target_dir=self.target_dir, imgid=imgid, anns=anns, dota_obj=self.dota_obj,
                                 orig_image=np.flipud(orig_img))
            returned_heat_map = k_to_heat_map[20].reshape((w_featmap, h_featmap)) / k_to_heat_map[20].reshape(
                (w_featmap, h_featmap)).max()
            returned_heat_map = \
                nn.functional.interpolate(torch.tensor(returned_heat_map).unsqueeze(dim=0).unsqueeze(dim=0),
                                          scale_factor=self.vit_patch_size,
                                          mode="nearest")[0][0]
            return returned_heat_map.numpy(), gt_attention

    def run_on_tensor(self, image: torch.tensor) -> torch.tensor:
        """Receives a single image and outputs a heatmap tensor in the spatial dims of the original image."""
        pass
