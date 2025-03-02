# Asymmetric Mutual Alignment for Unsupervised Zero-Shot Sketch-Based Image Retrieval
This repository provides a PyTorch implementation of the method presented in the AAAI 2024 paper titled "Asymmetric Mutual Alignment for Unsupervised Zero-Shot Sketch-Based Image Retrieval" (AAAI 2024). 

## Introduction
In recent years, many methods have been developed for zero-shot sketch-based image retrieval (ZS-SBIR). However, challenges arise due to the lack of training data that matches the test distribution and the absence of labels. We address this with unsupervised zero-shot sketch-based image retrieval (UZS-SBIR), where training data is unlabeled and training/testing categories do not overlap. We propose a novel **asymmetric mutual alignment** method (AMA) that includes a self-distillation module and a cross-modality mutual alignment module. This approach extracts feature embeddings from unlabeled data and aligns them across image and sketch domains, enhancing feature representations and improving generalization to unseen classes.

![UZS-SBIR](uzs-sbir.png)


## Get started
Clone this repository and create a virtual environment as the follows:

```bash
  conda create -n uzs-sbir python=3.7.13
  conda activate uzs-sbi
  conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge


