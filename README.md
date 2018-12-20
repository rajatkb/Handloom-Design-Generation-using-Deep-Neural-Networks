# Handloom-Design-Generation-using-Deep-Neural-Networks-
This is our 4th year Engineering final year project. It aims to use techniques like Conditional GAN , Image to Image translation , Texture and content transfer for aiding as a design tool for handloom weavers and also designers and industry as a whole.

### How to use the data_builder  
```
$ python data_builder.py -h
usage: data_builder.py [-h] --tfx TARGET_FOLDERX --tfy TARGET_FOLDERY
                       [--augx AUGMENTX] [--augy AUGMENTY]

data builder

optional arguments:
  -h, --help            show this help message and exit
  --tfx TARGET_FOLDERX
  --tfy TARGET_FOLDERY
  --augx AUGMENTX
  --augy AUGMENTY

```  

Run the the script at any target folder to create a dataset out of all the images inside the folder. Augmentation of image includes current colour shifting and inverting of axis and both combined.  

# 1. Simple GAN 

Attempt at achieving simple designs using very low amount of data of Meckala Sador dataset and also Saree Dataset

# 2. Variation AutoEncoder 

A simple attempt at genrative model to using very low amount of data for generating design after learning from Saree dataset. To be quoted "Miserable results" is what we we got. more updates on the methods later on but the very early version is not working

# 3. Fast Neural Style Transfer using Perceptual loss  

This will work as a image inpaniting method of various cases of probelm.  
### Use natural images and artistic image to train Network, Use the network to appy style on existing Saree (Normal and Mekhala data)  
### Use saree data to train to apply style and color of Mekhala dataset on the normal saree dataset
### Use artistic dataset to train , and use masking technique on hand drawn image for Image-inpainting
### Use Mekhala sador dataset for image inpainting

# 4. Image to Image translation problem i.e Pix2Pix

We treat generating handloom design as an image to image translation problem where take normal saree dataset will be treated as the input image and the Mekhala dataset as the target distribution which the normal dataset must be converted to. This can be tackled by following  
### CycleGAN  
### DiscoGAN  
### PixtoPix paper  


### Todo  
* ~~add option for resolution change~~
* ~~Find proper paramters for Faste Neural Style transfer for global learning~~
* ~~Use natural images and artistic image to train Network, Use the network to appy style on existing Saree (Normal and Mekhala data)~~  
* ~~Use saree data to train to apply style and color of Mekhala dataset on the normal saree dataset~~
* ~~Use artistic dataset to train , and use masking technique on hand drawn image for Image-inpainting~~
* ~~Use Mekhala sador dataset for image inpainting~~
* ~~Test CycleGAN on normal dataset prepare a testing script for our experiments~~  
* ~~Test DiscoGAN on normal dataset prepare a testing script for our experiments~~
* ~~Test both network for domain transfer~~
* ~~make web based pipeline for all operation~~
* ~~make python server for getting image and serving the processed image back
* make frontend Angular application
* have the python server interact with the model.predict to get output
