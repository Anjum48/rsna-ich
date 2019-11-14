import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import apex
import glob
from shutil import copyfile
from datetime import datetime
from colorama import init, Fore, Back, Style
from collections import OrderedDict
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/anjum/PycharmProjects/kaggle")
# sys.path.append("/home/anjum/rsna_code")  # GCP
from rsna_intracranial_hemorrhage_detection.datasets import ICHDataset, BalancedRandomSampler
from rsna_intracranial_hemorrhage_detection.utils import *
from rsna_intracranial_hemorrhage_detection.networks import model_builder
torch.multiprocessing.set_sharing_strategy('file_system')
init(autoreset=True)

INPUT_DIR = "/mnt/storage_dimm2/kaggle_data/rsna-intracranial-hemorrhage-detection/"
OUTPUT_DIR = "/mnt/storage/kaggle_output/rsna-intracranial-hemorrhage-detection/"
# INPUT_DIR = "/home/anjum/rsna_data/"  # GCP
# OUTPUT_DIR = "/home/anjum/rsna_output/"  # GCP
N_WORKERS = 4
FOLDS = 5


def competition_metric(target, predictions):
    target = target.numpy()
    predictions = torch.sigmoid(predictions).numpy()
    return log_loss(target.flatten(), predictions.flatten(), sample_weight=[2, 1, 1, 1, 1, 1] * target.shape[0])


# See https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109995#636452
def weighted_multi_label_logloss(criterion, prediction, target, weights):
    assert target.size() == prediction.size()
    assert weights.shape[0] == target.size(1)

    loss = 0
    for i in range(target.size(1)):
        loss += weights[i] * criterion(prediction[:, i], target[:, i])
    return loss


def train_model(args, device="cpu"):
    start = time.time()
    metadata = pd.read_parquet(os.path.join(INPUT_DIR, 'train_metadata.parquet.gzip'))
    # metadata = metadata.sample(1024*10, random_state=48)  # For testing

    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=15),
                                           transforms.RandomResizedCrop(size=(args.img_size, args.img_size),
                                                                        scale=(0.85, 1.0), ratio=(0.8, 1.2)),
                                           transforms.ToTensor(),
                                           # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           #                      std=[0.229, 0.224, 0.225])
                                           ])

    test_loaders = build_tta_loaders(args.img_size, dataset=args.stage, batch_size=args.batch, num_workers=N_WORKERS,
                                     image_folder=args.img_folder)

    # out-of-fold predictions on train data and averaged predictions on test data
    n_images = metadata.shape[0]
    categories = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    oof_df = pd.DataFrame({c: np.zeros(n_images) for c in categories}, index=metadata['SOPInstanceUID'])
    prediction = np.zeros((len(test_loaders[0].dataset), len(categories)))

    # list of scores on folds
    scores, fold_data = [], OrderedDict()

    # kfold = GroupKFold(n_splits=FOLDS)

    # for fold_n, (train_index, valid_index) in enumerate(kfold.split(metadata, groups=metadata["PatientID"])):
    #     train_image_ids = metadata.iloc[train_index]['SOPInstanceUID'].values
    #     valid_image_ids = metadata.iloc[valid_index]['SOPInstanceUID'].values

    # Using the same fold definition as the rest of the team
    for fold_n in range(5):
        train_image_ids = pd.read_csv(os.path.join(INPUT_DIR, args.cv_scheme, f"train_{fold_n}.csv"))["sop_instance_uid"]
        valid_image_ids = pd.read_csv(os.path.join(INPUT_DIR, args.cv_scheme, f"valid_{fold_n}.csv"))["sop_instance_uid"]

        train_dataset = ICHDataset("train", phase=0, image_filter=train_image_ids, transforms=train_transforms,
                                   image_folder=args.img_folder)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=N_WORKERS, shuffle=True,
                                  drop_last=True)
        valid_loaders = build_tta_loaders(args.img_size, dataset="train", batch_size=args.batch,
                                          phase=0, image_filter=valid_image_ids, num_workers=N_WORKERS,
                                          image_folder=args.img_folder)

        # Create fresh network. optimiser etc.
        torch.manual_seed(48 + fold_n)
        model = model_builder(args.architecture)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
        #                                                           patience=5, min_lr=0.00001, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        checkpoint_file = os.path.join(args.output_dir, f"phase0_fold_{fold_n + 1}.pt")
        early_stopping = EarlyStopping(patience=3, verbose=True, file_path=checkpoint_file, parallel=args.parallel)

        train_loss, valid_loss, valid_metric = [], [], []

        weights = np.array([2, 1, 1, 1, 1, 1])
        weights_norm = weights / weights.sum()
        criterion = torch.nn.BCEWithLogitsLoss()

        # Resume training
        if args.checkpoint_path is not None:
            try:
                checkpoint_file = max(glob.glob(args.checkpoint_path + f'/*fold_{fold_n + 1}*.pt'),
                                      key=os.path.getctime)
                print("Resumed from", checkpoint_file)
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                train_loss, valid_loss = checkpoint["train_loss"], checkpoint["valid_loss"]
                valid_metric = checkpoint['valid_metric']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
                epoch_start = checkpoint["epoch"] + 1
                early_stopping.load_state_dict(checkpoint["stopping_params"])
            except ValueError:
                print(f"Fold {fold_n + 1} checkpoint not found. Starting from scratch")
                epoch_start = 0
        else:
            epoch_start = 0

        if torch.cuda.is_available():
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
            if args.parallel:
                model = torch.nn.DataParallel(model)

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        for epoch in range(epoch_start, args.epochs):
            model.train()
            running_loss = 0
            for i, (image, target) in enumerate(train_loader):
                # # https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/losses/sigmoid_cross_entropy
                # if args.label_smoothing > 0:
                #     target = target * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                #     # target[:, 1:] = target[:, 1:] * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

                image, target = image.to(device), target.to(device)

                optimizer.zero_grad()
                y_hat = model(image)
                loss = weighted_multi_label_logloss(criterion, y_hat, target, weights_norm)

                if torch.cuda.is_available():
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 0 or i+1 == len(train_loader):
                    print(f"Fold {fold_n+1} [{epoch:2d}, {i+1:5d}/{len(train_loader):5d}] Loss: {loss.item():.4f}")

            valid_predictions, valid_target = test_time_augmentation(model, valid_loaders, device, "Valid")
            score = roc_auc_score(valid_target.numpy(), valid_predictions.numpy())
            validation_loss = weighted_multi_label_logloss(criterion, valid_predictions,
                                                           valid_target, weights_norm).item()
            # validation_metric = competition_metric(valid_target, valid_predictions)
            validation_metric = validation_loss
            running_loss = running_loss / len(train_loader)
            print(Fore.GREEN + f'Train Loss: {running_loss:.5f}, '
                               f'Validation Loss: {validation_loss:.5f}, '
                               f'Validation Metric: {validation_metric:.5f}, '
                               f'ROC AUC: {score:.5f}')

            train_loss.append(running_loss)
            valid_loss.append(validation_loss)
            valid_metric.append(validation_metric)
            # scheduler.step(validation_metric)
            scheduler.step()

            early_stopping(validation_metric, model, epoch=epoch, args=args, optimizer=optimizer.state_dict(),
                           train_loss=train_loss, valid_loss=valid_loss, valid_metric=valid_metric,
                           lr_scheduler=scheduler.state_dict())

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print(Fore.GREEN + f"Best score: {-early_stopping.best_score:.5f}")
        fold_data[f"train_loss_{fold_n + 1}"] = pd.Series(train_loss)
        fold_data[f"valid_loss_{fold_n + 1}"] = pd.Series(valid_loss)
        fold_data[f"valid_metric_{fold_n + 1}"] = pd.Series(valid_metric)

        # Predict on test and OOF sets using the best checkpoint on a fresh model
        # For some reason loading weights to an existing model was doing weird things with the submission ¯\_(ツ)_/¯
        print("Loading checkpoint and creating submission & OOFs")
        model = model_builder(args.architecture)
        model = model.to(device)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        if args.parallel:
            model = torch.nn.DataParallel(model)

        # scores.append(score)
        valid_predictions, valid_target = test_time_augmentation(model, valid_loaders, device, "Valid")
        oof_df.loc[valid_image_ids, categories] = torch.sigmoid(valid_predictions).numpy()

        plot_multiclass_roc_curve(valid_target, torch.sigmoid(valid_predictions).numpy(),
                                  os.path.join(args.output_dir, f"roc_fold_{fold_n + 1}.png"),
                                  -early_stopping.best_score)

        test_predictions, _ = test_time_augmentation(model, test_loaders, device, "Test ")
        prediction += test_predictions.detach().numpy()

        # Save results
        test_df = pd.DataFrame(torch.sigmoid(torch.tensor(prediction / (fold_n + 1))).numpy(), columns=categories)
        test_df["ImageID"] = test_loaders[0].dataset.image_ids
        test_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

        oof = wide_to_long(oof_df.reset_index(), id_var="SOPInstanceUID")
        sub = reindex_submission(wide_to_long(test_df), args.stage)
        oof.to_csv(os.path.join(args.output_dir, f"{args.output_dir[-15:]}_oof_predictions.csv.gz"),
                   index=False, compression="gzip")
        sub.to_csv(os.path.join(args.output_dir, f"{args.output_dir[-15:]}_submission.csv.gz"),
                   index=False, compression="gzip")

        # break  # HACK: Single fold

    metric_values = [v.iloc[-1] for k, v in fold_data.items() if 'valid_metric' in k]
    for i, (ll, roc) in enumerate(zip(metric_values, scores)):
        print(Fore.GREEN + f'Fold {i+1} LogLoss: {ll:.4f}, ROC AUC: {roc:.4f}')

    print(Fore.CYAN + f'CV mean ROC AUC score: {np.mean(scores):.4f}, std: {np.std(scores):.4f}')
    print(Fore.CYAN + f'CV mean Log Loss: {np.mean(metric_values):.4f}, std: {np.std(metric_values):.4f}')

    # Make a plot
    fold_data = pd.DataFrame(fold_data)
    fig, ax = plt.subplots()
    plt.title(f"CV Mean score: {np.mean(metric_values):.4f} +/- {np.std(metric_values):.4f}")
    valid_curves = fold_data.loc[:, fold_data.columns.str.startswith('valid_loss')]
    train_curves = fold_data.loc[:, fold_data.columns.str.startswith('train_loss')]
    valid_curves.plot(ax=ax, colormap='Blues_r')
    train_curves.plot(ax=ax, colormap='Reds_r')
    # ax.set_ylim([np.min(train_curves.values), np.max(valid_curves.values)])
    ax.tick_params(labelleft=True, labelright=True, left=True, right=True)
    plt.savefig(os.path.join(args.output_dir, f"phase0.png"))
    print("Done in", (time.time() - start) // 60, "minutes")


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='store', dest='config', help='Configuration scheme')
    args = parser.parse_args()

    # Lookup the config from the YAML file
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        if args.config is None:
            settings = cfg['default_run']
            # settings = cfg['default_run_1080']
        else:
            print("Using", args.config, "configuration")
            settings = cfg[args.config]

        args.epochs = settings["epochs"]
        args.batch = settings["batch"]
        args.lr = settings["lr"]
        args.img_size = settings["img_size"]
        args.checkpoint = settings["checkpoint"]
        args.parallel = settings["parallel"]
        args.img_folder = settings["img_folder"]
        args.architecture = settings["architecture"]
        args.stage = settings["stage"]
        args.cv_scheme = settings["cv_scheme"]

    if args.checkpoint is None:
        args.checkpoint = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(OUTPUT_DIR, "phase0", args.checkpoint)
        args.checkpoint_path = None
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    else:
        args.checkpoint_path = os.path.join(OUTPUT_DIR, "phase0", args.checkpoint)
        args.output_dir = os.path.join(OUTPUT_DIR, "phase0", datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    print("Saving output to", args.output_dir)
    copyfile("config.yml", os.path.join(args.output_dir, "config.yml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(args, device)
