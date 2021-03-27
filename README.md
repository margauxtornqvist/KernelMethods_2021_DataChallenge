# Kernel Methods For Machine Learning - 2021 - DataChallenge

Transcription factors (TFs) are proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. This data challenge consisted in predicting whether a DNA sequences is a binding site for a given TF or not. The classification into bound (label 1) or unbound (label 0) was done on 3 different datasets each composed of 2000 DNA sequences of length 101 nucleotides belonging to {A,C,G,T\}. This challenge brought us to explore kernel methods for classification and therefore implement several kernels and typical classifiers detailed in the following sections.

The **save_kernels.py** pyhton script allows to compute and save Gram matrixes for different kernels and is highly recommended if you are looking to do paramter optimization and thus use several times the different Gram matrixes. 

The **submit.py** python script allows to reproduce one of our best results.

**Note:**
We couldn't upload the kernels used in submit.py in github due to their high volume. Please download them from this [drive link](https://drive.google.com/drive/folders/13GxnsUzbkxCF-PRXS0QhUcWTifSfUWRl?usp=sharing) and add the corresponding to the repository before running submit.py (Please note that once you click the link, left click on mykernels and choose download. You will get a zipped folder, unzip it and you should get a folder untitled "mykernels" that you should add to the repository to make sure the code will run correctly).
Please note also that we provide directly the kernels because some of them are very time and computing power consuming (e.g. k=11).
