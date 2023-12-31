# ESRT
Efficient Transformer for Single Image
Super-Resolution

## Update
#######22.03.17########

The result images of our method are collected in fold "/result".



## Environment
*  pytorch >=1.0
* python 3.6
* numpy


## Model
<p align="center">
    <img src="figs/esrt.png" width="960"> <br />
    <em> The overall architecture of the proposed Efficient SR Transformer (ESRT). </em>
</p>
<p align="center">
    <img src="figs/EMHA.png" width="960"> <br />
    <em> Efficient Transformer and Efficient Multi-Head Attention. </em>
</p>

## Train
* dataset: DIV2K
* prepare

  Like [IMDN](https://github.com/Zheng222/IMDN), convert png files in  DIV2K to npy files:
  ```python
  python scripts/png2npy.py --pathFrom /path/to/DIV2K/ --pathTo /path/to/DIV2K_decoded/
  ```
* Training
```shell
python train.py --scale 2 --patch_size 96
python train.py --scale 3 --patch_size 144
python train.py --scale 4 --patch_size 192
```
If you want a better result, use 128/192/256 patch_size for each scale.

## Test
Example:

* test B100 X4
```shell
python test.py --is_y --test_hr_folder dataset/benchmark/B100/HR/ --test_lr_folder dataset/benchmark/B100/LR_bicubic/X4/ --output_folder results/B100/x4 --checkpoint experiment/checkpoint/x4/epoch_990.pth --upscale_factor 4
```

## Visual comparison
<p align="center">
    <img src="figs/visual images-v2.png" width="960"> <br />
    <em> The visual comparison. </em>
</p>
# asds
# asds
“# asdf”

python3 trainprism.py --scale 4 --patch_size 96 --root ‘dataset’ --threads 0

##python3 test.py  --is_y --test_hr_folder dataset/original_prism --test_lr_folder dataset/downscaled4 --output_folder result_of_model/  --checkpoint  experiment/checkpoint_ESRTprism_x4/prism epoch_60.pth --upscale_factor 4

## clean up log
use mask file to calculate MSE
remove hardcoded test data path  
more moves for old *.py files to util/old  
move unused *.py to util/old  
move data to project_data or project_data/old, part 2 
move data to project_data or project_data/old  
add ipynb files that compute PSNR and MSE  
calculate PSNR, MSE, SSIE in test.py

### test and training after clean up

python3 test.py  --is_y \
--test_hr_folder project_data/original \
--test_lr_folder project_data/downscaledx4/ \
--output_folder ./result_of_model2/ \
--checkpoint "experiment/checkpoint_ESRTprism_x4/prism epoch_80.pth" --upscale_factor 4

* for fast test, change line "set max_test_item = 2000"

python3 trainprism.py --scale 4 --patch_size 96 --threads 0 \
--test_hr_folder "project_data/train_npy_files/high_resolution" \
--test_lr_folder "project_data/train_npy_files/low_resolution"