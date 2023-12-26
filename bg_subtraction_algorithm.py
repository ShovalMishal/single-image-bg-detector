import argparse
from enum import Enum
import numpy as np
from DOTA_devkit.DOTA import DOTA
from bg_subtraction_accord_pixels import bg_detector_alg_pixels_ver
from bg_subtraction_with_dino_vit import bg_subtraction_with_dino_vit


class AlgorithmType(Enum):
    PIXELS = "pixels"
    VIT_CLS_HEADS = "vit"


def parse_args():
    parser = argparse.ArgumentParser(description="apply single image BG detection for the images in source dir and"
                                                 "output results to the target dir.")
    parser.add_argument("--source", help="Path to the source directory", default='examples')
    parser.add_argument("--target", help="Path to the target directory", default='results')
    parser.add_argument('--bg_subtraction_alg_type', default='pixels', type=str,
                        choices=['pixels', 'vit'], help='Optional background subtraction algorithms.')
    parser.add_argument('--patch_size_in_meters', nargs=2, default=(5, 5), type=int, help='A tuple of two integers representing patch '
                                                                          'size in meters')
    parser.add_argument('--vit_patch_size', default=8, type=int, help='Patch resolution of the vit model.')
    parser.add_argument('--vit_arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument("--vit_image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--vit_threshold", type=float, default=None, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()
    return args


def bg_detector_alg(dota_obj, imgid, target_dir, patch_size_in_meter, vit_patch_size, vit_arch, vit_image_size,
                    vit_threshold, alg_type: AlgorithmType=AlgorithmType.PIXELS):
    if alg_type == AlgorithmType.PIXELS.value:
        bg_detector_alg_pixels_ver(imgid=imgid, target_dir=target_dir, dota_obj=dota_obj,
                                   patch_size_in_meter=patch_size_in_meter)
    elif alg_type == AlgorithmType.VIT_CLS_HEADS.value:
        bg_subtraction_with_dino_vit(imgid=imgid, target_dir=target_dir,vit_patch_size=vit_patch_size, vit_arch=vit_arch,
                                     vit_image_size=vit_image_size, dota_obj=dota_obj, threshold=vit_threshold, patch_size_in_meter=patch_size_in_meter)


def main():
    args = parse_args()
    source_dir = args.source
    target_dir = args.target
    patch_size_in_meters = args.patch_size_in_meters
    vit_patch_size = args.vit_patch_size
    vit_arch = args.vit_arch
    vit_image_size = args.vit_image_size
    vit_threshold = args.vit_threshold
    bg_subtraction_alg_type = args.bg_subtraction_alg_type
    dota_obj = DOTA(basepath=source_dir)
    imgids = dota_obj.getImgIds()
    for imgid in imgids:
        bg_detector_alg(dota_obj=dota_obj, imgid=imgid, target_dir=target_dir, patch_size_in_meter=patch_size_in_meters,
                        vit_patch_size=vit_patch_size, vit_arch=vit_arch, vit_image_size=vit_image_size,
                        vit_threshold=vit_threshold, alg_type=bg_subtraction_alg_type)


if __name__ == '__main__':
    main()
