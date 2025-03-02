# Asymmetric Mutual Alignment for Unsupervised Zero-Shot Sketch-Based Image Retrieval

In recent years, many methods have been developed for zero-shot sketch-based image retrieval (ZS-SBIR). However, challenges arise due to the lack of training data that matches the test distribution and the absence of labels. We address this with unsupervised zero-shot sketch-based image retrieval (UZS-SBIR), where training data is unlabeled and training/testing categories do not overlap. We propose a novel **asymmetric mutual alignment** method (AMA) that includes a self-distillation module and a cross-modality mutual alignment module. This approach extracts feature embeddings from unlabeled data and aligns them across image and sketch domains, enhancing feature representations and improving generalization to unseen classes.

![UZS-SBIR](uzs-sbir.png)
