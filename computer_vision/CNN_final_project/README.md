# Before you begin
Please make sure you are running the python scripts in the correct working directory.
This directory contains all the scripts:

`cd CNN_final_project/`

# Steps to run the pipeline 
1. Please install requirements.txt by running
`pip install -r requirements.txt`

2. If you are using my script to CREATE a model, then you must provide datasets in the `train/` and `test/` directory.
The standford SVHN Dataset is provided here: http://ufldl.stanford.edu/housenumbers/
If you are creating the models run these commands, otherwise skip to step 3:
`python build_model.py --vgg16_pretrained` for the pretrained vgg16 model
`python build_model.py --vgg16_scratch` for the vgg16 model without weights
`python build_model.py --sequential` for a sequential keras model

3. To replicate my output images, please use my VGG16 Pretrained Model `vgg16_pretrained.h5`:
https://www.dropbox.com/sh/137rywvch6ukmg6/AACLsUG6PyF_3VPyptwn3643a?dl=0
My other models are located inside the dropbox folder `other models`
Make sure the `vgg16_pretrained.h5` is in the `model/` directory before proceeding to step 4.

4. The 5 input images are in the `input/` folder and the output images are generated in the `output/` directory.
Run this command to generate the output images found in the report:
`python classify_images.py`