from typing import Union, List
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


# =====================================================================
# 1. Color Interventions (Manual Section: 2. Color Bias)
# =====================================================================
def apply_grayscale(
    image: Union[Image.Image, torch.Tensor],
) -> Union[Image.Image, torch.Tensor]:
    """Converts image to grayscale while retaining 3 channels for backbone compatibility."""
    if isinstance(image, torch.Tensor):
        return F.rgb_to_grayscale(image, num_output_channels=3)
    return F.to_grayscale(image, num_output_channels=3)


def apply_hue_rotation(
    image: Union[Image.Image, torch.Tensor], hue_factor: float = 0.5
) -> Union[Image.Image, torch.Tensor]:
    """Rotates hue by a fixed factor in [-0.5, 0.5].

    hue_factor=0.5 corresponds to a 180-degree rotation in HSV space.
    """
    return F.adjust_hue(image, hue_factor=hue_factor)


# =====================================================================
# 2. Spatial Translation (Manual Section: 4. Spatial Translation)
# =====================================================================
def apply_translation(
    image: Union[Image.Image, torch.Tensor],
    shift_px: int,
    direction: str,
) -> Union[Image.Image, torch.Tensor]:
    """Displaces image by shift_px in direction ('north', 'south', 'east', 'west')

    using reflection padding and shifted crops.
    """
    if shift_px == 0:
        return image

    valid_directions = {"north", "south", "east", "west"}
    direction = direction.lower()
    if direction not in valid_directions:
        raise ValueError(
            f"Direction must be one of {valid_directions}, got '{direction}'"
        )

    if isinstance(image, torch.Tensor):
        _, h, w = image.shape[-3:]
    else:
        w, h = image.size

    padded = F.pad(
        image,
        padding=[shift_px, shift_px, shift_px, shift_px],
        padding_mode="reflect",
    )

    if direction == "north":
        top = 2 * shift_px
        left = shift_px
    elif direction == "south":
        top = 0
        left = shift_px
    elif direction == "west":
        top = shift_px
        left = 2 * shift_px
    elif direction == "east":
        top = shift_px
        left = 0

    return F.crop(padded, top=top, left=left, height=h, width=w)


# =====================================================================
# 3. Patch Shuffling (Manual Section: 5. Patch Shuffling)
# =====================================================================
def apply_patch_shuffle(
    image: Union[Image.Image, torch.Tensor],
    grid_size: int = 4,
    seed: int = 6304,
) -> Union[Image.Image, torch.Tensor]:
    """Divides image into grid_size x grid_size grid (16 patches for 4x4)

    and shuffles their positions deterministically with the specified seed.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        tensor_img = F.to_tensor(image)
    else:
        tensor_img = image

    c, h, w = tensor_img.shape[-3:]
    assert h % grid_size == 0 and w % grid_size == 0, (
        f"Image dimensions ({h}x{w}) must be divisible by grid_size ({grid_size})"
    )
    patch_h = h // grid_size
    patch_w = w // grid_size

    patches: List[torch.Tensor] = []
    for r in range(grid_size):
        for c_idx in range(grid_size):
            patch = tensor_img[
                ...,
                r * patch_h : (r + 1) * patch_h,
                c_idx * patch_w : (c_idx + 1) * patch_w,
            ]
            patches.append(patch)

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(patches))
    shuffled_patches = [patches[idx] for idx in shuffled_indices]

    rows = []
    for r in range(grid_size):
        row_patches = shuffled_patches[r * grid_size : (r + 1) * grid_size]
        rows.append(torch.cat(row_patches, dim=-1))

    shuffled_tensor = torch.cat(rows, dim=-2)

    if is_pil:
        return F.to_pil_image(shuffled_tensor)
    return shuffled_tensor