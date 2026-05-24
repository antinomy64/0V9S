# -*- coding: utf-8 -*-
"""
Coarse PascalPart116 taxonomy for 0V9S / Talk2DINO.

Single source of truth for both:
  - Stage2/GW training via src.dataset_joint.DinoClipJointDataset
  - OVSS/MMseg evaluation via PascalPart116_PART_COARSE

Fine ids are exactly the original PascalPart116 part ids:
  id = index in FINE_PART_CLASSES
Ignore label 255 is preserved when remapping masks.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


FINE_PART_CLASSES = (
    "aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail",
    "aeroplane's engine", "aeroplane's wheel",
    "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight",
    "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot",
    "bottle's body", "bottle's cap",
    "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window",
    "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window",
    "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear",
    "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn",
    "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle",
    "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", "horse's muzzle", "horse's hoof",
    "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight",
    "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand",
    "pottedplant's pot", "pottedplant's plant",
    "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn",
    "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof",
    "train's coach", "tvmonitor's screen",
)

FINE_PART_NAME_TO_ID = {name: i for i, name in enumerate(FINE_PART_CLASSES)}

# Conservative parent-level grouping: tiny parts -> immediate parent part.
# Total coarse classes: 58.
COARSE_PART_GROUPS: Dict[str, List[str]] = {
    "aeroplane's body": ["aeroplane's body", "aeroplane's engine", "aeroplane's wheel"],
    "aeroplane's wing": ["aeroplane's wing"],
    "aeroplane's tail": ["aeroplane's stern", "aeroplane's tail"],

    "bicycle's wheel": ["bicycle's wheel", "bicycle's chainwheel"],
    "bicycle's saddle": ["bicycle's saddle"],
    "bicycle's handlebar": ["bicycle's handlebar", "bicycle's headlight"],

    "bird's head": ["bird's head", "bird's eye", "bird's beak"],
    "bird's torso": ["bird's torso", "bird's neck"],
    "bird's wing": ["bird's wing"],
    "bird's tail": ["bird's tail"],
    "bird's leg": ["bird's leg", "bird's foot"],

    "bottle's body": ["bottle's body", "bottle's cap"],

    "bus's wheel": ["bus's wheel"],
    "bus's front": ["bus's front", "bus's headlight", "bus's license plate"],
    "bus's side": ["bus's side", "bus's mirror", "bus's door", "bus's window"],
    "bus's back": ["bus's back"],
    "bus's roof": ["bus's roof"],

    "car's wheel": ["car's wheel"],
    "car's front": ["car's front", "car's headlight", "car's license plate"],
    "car's side": ["car's side", "car's mirror", "car's door", "car's window"],
    "car's back": ["car's back"],
    "car's roof": ["car's roof"],

    "cat's head": ["cat's head", "cat's eye", "cat's nose", "cat's ear"],
    "cat's torso": ["cat's torso", "cat's neck"],
    "cat's leg": ["cat's leg", "cat's paw"],
    "cat's tail": ["cat's tail"],

    "cow's head": ["cow's head", "cow's eye", "cow's ear", "cow's muzzle", "cow's horn"],
    "cow's torso": ["cow's torso", "cow's neck"],
    "cow's leg": ["cow's leg"],
    "cow's tail": ["cow's tail"],

    "dog's head": ["dog's head", "dog's eye", "dog's nose", "dog's ear", "dog's muzzle"],
    "dog's torso": ["dog's torso", "dog's neck"],
    "dog's leg": ["dog's leg", "dog's paw"],
    "dog's tail": ["dog's tail"],

    "horse's head": ["horse's head", "horse's eye", "horse's ear", "horse's muzzle"],
    "horse's torso": ["horse's torso", "horse's neck"],
    "horse's leg": ["horse's leg", "horse's hoof"],
    "horse's tail": ["horse's tail"],

    "motorbike's wheel": ["motorbike's wheel"],
    "motorbike's saddle": ["motorbike's saddle"],
    "motorbike's handlebar": ["motorbike's handlebar", "motorbike's headlight"],

    "person's head": ["person's head", "person's eye", "person's nose", "person's ear", "person's eyebrow", "person's mouth", "person's hair"],
    "person's torso": ["person's torso", "person's neck"],
    "person's leg": ["person's leg", "person's foot"],
    "person's arm": ["person's lower arm", "person's upper arm", "person's hand"],

    "pottedplant's pot": ["pottedplant's pot"],
    "pottedplant's plant": ["pottedplant's plant"],

    "sheep's head": ["sheep's head", "sheep's eye", "sheep's ear", "sheep's muzzle", "sheep's horn"],
    "sheep's torso": ["sheep's torso", "sheep's neck"],
    "sheep's leg": ["sheep's leg"],
    "sheep's tail": ["sheep's tail"],

    "train's head": ["train's head", "train's headlight"],
    "train's front": ["train's front"],
    "train's side": ["train's side"],
    "train's back": ["train's back"],
    "train's roof": ["train's roof"],
    "train's coach": ["train's coach"],

    "tvmonitor's screen": ["tvmonitor's screen"],
}

COARSE_PART_CLASSES = tuple(COARSE_PART_GROUPS.keys())
COARSE_PART_NAME_TO_ID = {name: i for i, name in enumerate(COARSE_PART_CLASSES)}

COARSE_ID_TO_FINE_IDS: Dict[int, List[int]] = {}
FINE_ID_TO_COARSE_ID: Dict[int, int] = {}

for coarse_name, fine_names in COARSE_PART_GROUPS.items():
    coarse_id = COARSE_PART_NAME_TO_ID[coarse_name]
    fine_ids = []
    for fine_name in fine_names:
        if fine_name not in FINE_PART_NAME_TO_ID:
            raise KeyError(f"Unknown fine PascalPart116 part: {fine_name}")
        fine_id = FINE_PART_NAME_TO_ID[fine_name]
        fine_ids.append(fine_id)
        FINE_ID_TO_COARSE_ID[fine_id] = coarse_id
    COARSE_ID_TO_FINE_IDS[coarse_id] = sorted(set(fine_ids))

_missing = sorted(set(range(len(FINE_PART_CLASSES))) - set(FINE_ID_TO_COARSE_ID.keys()))
if _missing:
    raise ValueError(f"Missing fine part ids in coarse mapping: {_missing}")

# LUT for mask remap. 255 is ignore/background.
FINE_TO_COARSE_LUT = np.full((256,), 255, dtype=np.uint8)
for fine_id, coarse_id in FINE_ID_TO_COARSE_ID.items():
    FINE_TO_COARSE_LUT[fine_id] = coarse_id
FINE_TO_COARSE_LUT[255] = 255


# Palette copied/generated in the same Pascal VOC style; length >= 58.
def _voc_palette(n=256):
    palette = []
    for j in range(n):
        lab = j
        r = g = b = 0
        i = 0
        while lab > 0:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        palette.append([r, g, b])
    return palette

COARSE_PALETTE = _voc_palette(256)[:len(COARSE_PART_CLASSES)]


IMAGENET_TEMPLATES = (
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
)


def build_prompts(name: str) -> List[str]:
    return [template.format(name) for template in IMAGENET_TEMPLATES]


def coarse_parts_for_object(obj_name: str) -> List[str]:
    prefix = f"{obj_name}'s "
    return [name for name in COARSE_PART_CLASSES if name.startswith(prefix)]


def remap_fine_mask_to_coarse(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max(initial=0) > 255:
        raise ValueError("Expected uint8 PascalPart116 mask with values in [0, 255].")
    return FINE_TO_COARSE_LUT[mask.astype(np.uint8)]


def fine_ids_for_coarse_id(coarse_id: int) -> List[int]:
    return list(COARSE_ID_TO_FINE_IDS[int(coarse_id)])


def print_summary() -> None:
    print(f"fine parts: {len(FINE_PART_CLASSES)}")
    print(f"coarse parts: {len(COARSE_PART_CLASSES)}")
    for i, name in enumerate(COARSE_PART_CLASSES):
        fine_names = [FINE_PART_CLASSES[j] for j in COARSE_ID_TO_FINE_IDS[i]]
        print(f"{i:02d} {name}: {fine_names}")


if __name__ == "__main__":
    print_summary()
