from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List
import clip
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from data_utils import base_path, FashionIQDataset
from combiner import Combiner
from utils import update_train_running_results, set_train_bar_description, save_model, device
from validate import compute_fiq_val_metrics


def combiner_training_fiq(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozen the CLIP model
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

    print("Preprocessing pipeline applied successfully.")

    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    print(f"Train dataset size: {len(relative_train_dataset)}")

    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True
    )
    print(f"Number of batches in training loader: {len(relative_train_loader)}")

    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if save_best:
        best_avg_recall = 0

    print('Training loop started')
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        combiner.train()
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)

        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            step = len(train_bar) * epoch + idx
            images_in_batch = reference_images.size(0)

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            with torch.no_grad():
                reference_features = clip_model.encode_image(reference_images).float()
                target_features = clip_model.encode_image(target_images).float()
                text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
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
        print(f"Epoch {epoch + 1} training loss: {train_epoch_loss:.4f}")
        scheduler.step()

        if epoch % validation_frequency == 0:
            clip_model.float()
            combiner.eval()
            recalls_at10 = []

            for val_dress_type in val_dress_types:
                val_dataset = FashionIQDataset('val', [val_dress_type], 'relative', preprocess)
                recall_at10, _ = compute_fiq_val_metrics(val_dataset, clip_model, combiner.combine_features)
                recalls_at10.append(recall_at10)

            avg_recall = mean(recalls_at10)
            print(f"Validation Avg Recall @10 for epoch {epoch + 1}: {avg_recall:.4f}")
            if save_best and avg_recall > best_avg_recall:
                best_avg_recall = avg_recall
                save_model('combiner_best', epoch, combiner, training_path)

        if save_training and not save_best:
            save_model(f'combiner_epoch_{epoch}', epoch, combiner, training_path)

    print("Training complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the Combiner model on FashionIQ dataset.")
    
    # Arguments for dataset and training
    parser.add_argument("--dataset", type=str, default="FashionIQ", help="Dataset to use (default: FashionIQ)")
    parser.add_argument("--clip-model-name", type=str, required=True, help="CLIP model name (e.g., RN50, RN50x4)")
    parser.add_argument("--clip-model-path", type=str, required=True, help="Path to the CLIP model weights")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--projection-dim", type=int, default=2560, help="Projection dimension of the combiner")
    parser.add_argument("--hidden-dim", type=int, default=5120, help="Hidden dimension of the combiner")
    parser.add_argument("--combiner-lr", type=float, default=0.0001, help="Learning rate for the combiner")
    parser.add_argument("--save-training", action="store_true", help="Save the training model at each epoch")
    parser.add_argument("--save-best", action="store_true", help="Save the best model based on validation metrics")
    parser.add_argument("--validation-frequency", type=int, default=1, help="Validate the model every N epochs")
    parser.add_argument("--transform", type=str, default="clip", help="Preprocessing transform to use (default: clip)")
    parser.add_argument("--target-ratio", type=float, default=1.25, help="Target ratio for targetpad transform")

    # Parse the arguments
    args = parser.parse_args()

    # Define training hyperparameters
    training_hyper_params = {
        "train_dress_types": ['dress', 'toptee', 'shirt'],  # Categories for training
        "val_dress_types": ['dress', 'toptee', 'shirt'],    # Categories for validation
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.batch_size,  # Batch size for CLIP (same as training batch size)
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "target_ratio": args.target_ratio,
    }

    # Call the training function
    combiner_training_fiq(**training_hyper_params)

