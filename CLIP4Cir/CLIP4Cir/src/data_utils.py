import json
from pathlib import Path
from typing import List
import PIL
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Base directory path
base_path = Path(__file__).absolute().parents[2].absolute()

# Utility function to convert images to RGB
def _convert_image_to_rgb(image):
    return image.convert("RGB")

# Simple preprocessing transform for resizing and normalization
def preprocess_transform(dim: int):
    """
    Simple preprocessing transform:
        - Resize to the desired dimension
        - CenterCrop for consistent size
        - Normalize using CLIP's default mean and std values
    :param dim: Target dimension for resizing and cropping
    :return: Torchvision Compose object
    """
    return Compose([
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# Dataset class for FashionIQ
class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class to manage FashionIQ data.
    Dataset Modes:
        - 'classic': Yields (image_name, image) tuples
        - 'relative': Yields:
            - (reference_image, target_image, captions) for 'train' split
            - (reference_name, target_name, captions) for 'val' split
            - (reference_name, reference_image, captions) for 'test' split
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        Initialize the FashionIQ dataset.
        :param split: Dataset split ('train', 'val', 'test')
        :param dress_types: List of fashion categories (e.g., ['dress', 'toptee', 'shirt'])
        :param mode: Dataset mode ('relative' or 'classic')
        :param preprocess: Image preprocessing function
        """
        self.split = split
        self.dress_types = dress_types
        self.mode = mode
        self.preprocess = preprocess

        # Validate arguments
        if mode not in ['relative', 'classic']:
            raise ValueError("Mode should be 'relative' or 'classic'")
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split should be 'train', 'val', or 'test'")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'toptee', 'shirt']:
                raise ValueError("Dress type should be 'dress', 'toptee', or 'shirt'")

        # Load triplets for 'relative' mode
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # Load image names for 'classic' mode
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    # Load and preprocess reference image
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))

                    # Load and preprocess target image
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))

                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    # Load and preprocess reference image
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("Mode should be 'relative' or 'classic'")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        return len(self.triplets) if self.mode == 'relative' else len(self.image_names)


# Example preprocess pipeline for FashionIQ
def get_preprocess(dim: int):
    return preprocess_transform(dim)
