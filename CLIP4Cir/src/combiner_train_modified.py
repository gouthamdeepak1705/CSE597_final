from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_utils import base_path, FashionIQDataset, CIRRDataset, targetpad_transform, squarepad_transform
from combiner import Combiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics

def combiner_training_fiq(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozed the CLIP model
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_fiq_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    # Custom augmentation pipeline
    preprocess = transforms.Compose([
        preprocess,
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    if save_best:
        best_avg_recall = 0

    print('Training loop started')
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16
        combiner.train()
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)

        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            step = len(train_bar) * epoch + idx
            images_in_batch = reference_images.size(0)

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            flattened_captions = generate_randomized_fiq_caption(np.array(captions).T.flatten().tolist())
            text_inputs = clip.tokenize(flattened_captions, truncate=True).to(device, non_blocking=True)

            with torch.no_grad():
                reference_features = clip_model.encode_image(reference_images).float()
                target_features = clip_model.encode_image(target_images).float()
                text_features = clip_model.encode_text(text_inputs).float()

            with torch.cuda.amp.autocast():
                logits = combiner(reference_features, text_features, target_features)
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = crossentropy_criterion(logits, ground_truth)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        train_epoch_loss = train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']
        scheduler.step()  # Adjust learning rate

        if epoch % validation_frequency == 0:
            clip_model.float()
            combiner.eval()
            recalls_at10 = []

            for val_dress_type in val_dress_types:
                val_dataset = FashionIQDataset('val', [val_dress_type], 'relative', preprocess)
                recall_at10, _ = compute_fiq_val_metrics(val_dataset, clip_model, combiner.combine_features)
                recalls_at10.append(recall_at10)

            avg_recall = mean(recalls_at10)
            if save_best and avg_recall > best_avg_recall:
                best_avg_recall = avg_recall
                save_model('combiner_best', epoch, combiner, training_path)

            print(f"Epoch {epoch}: Avg Recall @10: {avg_recall}")

        if save_training and not save_best:
            save_model(f'combiner_epoch_{epoch}', epoch, combiner, training_path)
