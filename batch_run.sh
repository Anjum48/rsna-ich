#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1

# Stage 1 training
python model.py --config=efficientnetb0 | tee output_efficientnetb0.txt
python model.py --config=efficientnetb5 | tee output_efficientnetb5.txt
python model.py --config=efficientnetb3 | tee output_efficientnetb3.txt

# These were never run
#python model.py --config=densenet169 | tee output_densenet169.txt
#python model.py --config=seresnext | tee output_seresnext.txt
#python model.py --config=vgg19 | tee output_vgg19.txt

# Stage 2 inference
python model.py --config=efficientnetb0-stage2 | tee output_efficientnetb0-stage2.txt
python model.py --config=efficientnetb5-stage2 | tee output_efficientnetb5-stage2.txt
python model.py --config=efficientnetb3-stage2 | tee output_efficientnetb3-stage2.txt
