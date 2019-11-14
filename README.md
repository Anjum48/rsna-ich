# RSNA Intracranial Hemorrhage Detection
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

# Hardware used
* 2x NVidia RTX 2080Ti GPUs
* 128GB RAM
* 16-core AMD CPU
* Data stored on NVMe drives in RAID configuration
* OS: Ubuntu 19.04

# Steps to reproduce results
1. Modify the data paths at the top of `data_prep.py`, `datasets.py` & `model.py`

2. Run `data_prep.py`. This takes around 12-15 hours for each set of images and will create:
    * `train_metadata.parquet.gzip`
    * `stage_1_test_metadata.parquet.gzip`
    * `stage_2_test_metadata.parquet.gzip`
    * `train_triplets.csv`
    * `stage_1_test_triplets.csv`
    * `stage_1_test_triplets.csv`
    * A folder called `png` for all 3 stages containing the preprocessed & cropped images
    
3. Run `batch_run.sh`. This will train (using 5 fold CV) and make submission files (as well as 
out-of fold predictions) using:
    * EfficientNet-B0 (224x224 model for fast experimentation)
    * EfficientNet-B5 (456x456, 3-slice model)
    * EfficientNet-B3 (300x300, 3-window model)
    * ~~DenseNet-169~~
    * ~~SE-ResNeXt101_32x4d~~
    
    This will create a timestamped folder in the `OUTPUT_DIR` containing:
    * A submission file
    * Out-of-fold (OOF) predictions (for later stacking models)
    * Model checkpoints for each fold
    * QC plots (e.g. ROC curves, train/validation loss curves)
    
4. To infer on different datasets:
    * In `config.yml` set `stage` to either `test1` or `test2`
    * For the `checkpoint` eneter the name of the timestamped folder containing the model checkpoints
    * Set epochs to 1 (this will skip the training part)
    * Re-run the models using `batch_run.sh`. A new output directory will be created with the 
    predictions
    * If a completely new dataset is being used, the file paths in `ICHDataset` found in 
    `datasets.py` will need to be modified
    
# Model summary

## Primary model (3-slice model)
1. First the metadata is collected from the individual DICOM images. This allows the studies to be
grouped by `PatientID` which is important for a stable cross validation due to overlapping patients

2. Based on `StudyInstanceUID` and sorting on `ImagePositionPatient` it is possible to reconstruct
3D volumes for each study. However since each study contained a variable number of axial slices
(between 20-60) this makes it difficult to create a architecture that implements 3D convolutions. 
Instead, triplets of images were created from the 3D volumes to represent the RGB channels of an 
image, i.e. the green channel being the target image and the red & blue channels being the adjacent
images. If an image was at the edge of the volume, then the green channel was repeated. This is 
essentially a 3D volume but only using 3 axial slices. At this stage no windowing was applied 
and the image is retained in Hounsfield units.

3. The images then had the objects labeled using `scipy.ndimage.label` which looks for groups of 
connected pixels. The group with the second largest number of pixels was assumed to be the head 
(the first largest group is the background). This removes most of the dead space and the headrest
of the CT scanner. A 10 pixel border was retained to keep some space for rotation augmentations

4. The images were clipped between 0-255 Hounsfield units and saved as 8-bit PNG files to pass to 
the PyTorch dataset object. The reason for this was a) most of the interesting features are 
between 0-255 HU so we shouldn't be losing too much detail and b) this makes it easier to try 
different windows without recreating the images. I also found that processing DICOM images on the
fly was too slow to keep 2 GPUs busy when small images/large batch sizes were used, (224x224, 
batch size=256) which is why I went down the PNG route.

5. The images are then windowed once loaded into the dataset. A subdural window with 
`window_width, window_length = 200, 80` was used on all 3 channels. 

## Alternative model (3-window model)
An alternative model using different CT windowing for each channel is also used. Here the windows
are applied when the PNG images are made according to the channels in `prepare_png` found in 
`data_prep.py`. The images are cropped in the same way above. The windows used were :
* Brain - `window_width, window_length = 80, 40`
* Subdural - `window_width, window_length = 200, 80`
* Bone - `window_width, window_length = 2000, 600`

## Model details

1. Image augmentations: 
    1. `RandomHorizontalFlip(p=0.5)`
    2. `RandomRotation(degrees=15)`
    3. `RandomResizedCrop(scale=(0.85, 1.0), ratio=(0.8, 1.2))`
    
2. The models were then trained as follows:
    * 5 folds defined by grouping on `PatientID`. See the section below on the CV scheme.
    * All the data used (no down/up sampling)
    * 10 epochs with early stopping (patience=3)
    * AdamW optimiser with default parameters
    * Learning rate decay using cosine annealing
    * Loss function: custom weighted multi-label log loss with `weights=[2, 1, 1, 1, 1, 1]`
    * Image size: 512x512. No pre-normalisation (i.e. ImageNet stats were not used)
    * Batch size: as large as possible depending on the architecture
    
3. Postprocessing:
    * Test time augmentation (TTA): Identity, horizontal flip, rotate -10 degrees & +10 degrees 
    * Take the mean of all 5 folds
    * A prediction smoothing script based on the relative positions of the axial slices 
    (this script was used by all teammates)
    
# Cross validation scheme
The CV scheme was fixed using 5 pairs of CSV files agreed by the team in the format `train_n.csv` 
& `valid_n.csv`. The first version of these files were designed to prevent the same patient appearing 
in train & validation sets. A second version of these files were made removing patients that were present 
in the train & stage 1 test sets to prevent fitting to the overlapping patients on the stage 1 public LB. 
Some of these models are trained with the V1 scheme and others with the V2 scheme (most of the team used 
the latter)
 
These CSV files are included in a file called `team_folds.zip` and should be in the same folder
as the the rest of the input data. The scheme is selected using the `cv_scheme` value in the config file