Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features

##Step 1: Setting up the environment and getting the pre-requisites ready before the running of the code.

Download the FashionIQ dataset: [Link](https://drive.google.com/drive/folders/1DVb606AG1aP9QZ-VIy9qRGaSz-VnbrYq?usp=sharing)<br>
FashionIQ dataset should be in the following path ./CLIP4Cir<br>
1st Download CLIP4Cir models : [LINK](https://drive.google.com/drive/folders/1-ULTy3jocqbxPgw6cOG4FM8jgXojm8kJ?usp=sharing)<br>
1st CLIP4Cir model should be in the following path ./CLIP4Cir/model<br>
2nd Download CLIP4Cir models : [LINK](https://drive.google.com/drive/folders/1-IE_O_lSupP-k1FqMVU8NBRmBSUHs5jI?usp=sharing)<br>
2nd CLIP4Cir model should be in the following path ./CLIP4Cir/CLIP4Cir/model <br>

##Step 2: Run the main.ipynb 

# [CVPR 2024] Improved MeaCap: Memory-Augmented Zero-shot Image Captioning

**Authors**:
[Vedant Sawant](linkedin.com/in/vedantsawant6900)
<br/>
Improved implementation of  MeaCap.


<br/>

<div align = center>
<img src="./assets/demo1.png">
</div>

<br/>

# Steps to Replicate
## Step 1: 
### Clone the repository
## Step 2: 
### Setup Python Environment. Execute cells under setup environment
## Step 3: 
### Go into the environment/lib/python3.8/sentence_transformer/Sentence_Transformer.py and remove cache_download. Do the same with utils.py files in same folder
## Step 4:
### Download external memory for image captioning. 
### CC3M: [Link](https://huggingface.co/JoeyZoZ/MeaCap/tree/main/memory)
### SS1M: [Link](https://huggingface.co/JoeyZoZ/MeaCap/tree/main/memory)
### COCO: [Link](https://huggingface.co/JoeyZoZ/MeaCap/tree/main/memory)
### Flickr30k: [Link](https://huggingface.co/JoeyZoZ/MeaCap/tree/main/memory)
### After downloading put them in data folder
![img.png](img.png)<br>
### you can also preprocess a new textual memory bank, for example:
### python prepare_embedding.py --memory_id coco --memory_path data/memory/coco/memory_captions.json
## Step 5: Download the openai/clip-vit-base-patch32 model which extracts the image content. [Click here to download.](https://huggingface.co/openai/clip-vit-base-patch32)
## Step 6: Download the SceneGraphParser lizhuang144/flan- t5-base-VG-factual-sg which helps us understand the objects and relationship. [Click here to download.](https://huggingface.co/lizhuang144/flan-t5-base-VG-factual-sg)
## Step 7: Download the SentenceBERT which helps in connecting different context into sentence. [Click here to download.](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 
## Step 8: Modify paths of all these downloaded files in get_args file.
## Step 9: Download Flickr30k Dataset with images and its annotations.
## Step 10: Split the dataset using karpathy split and put all images in ./image_example
## Step 11: Change annotation format as given in image.![img_1.png](img_1.png)
## Step 12: As MeaCapTOT is the SOTA method use memory id as coco which is by default. Use Language model as CBART_COCO. 
## Step 13: Run python inference.py --memory_id coco --img_path ./image_example -- lm_model_path ./checkpoints/CBART_COCO
## Step 14: Check the output file in ./outputs and check the captions.
## Step 15: Finally Run the cocoeval.py to get the metrics and compare the Result.
#Acknowledgements
Thank You [MeaCap](https://github.com/joeyz0z/MeaCap?tab=readme-ov-file#inference)
