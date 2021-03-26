# Kernel Methods For Machine Learning - 2021 - DataChallenge

Transcription factors (TFs) are proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes. This data challenge consisted in predicting whether a DNA sequences is a binding site for a given TF or not. The classification into bound (label 1) or unbound (label 0) was done on 3 different datasets each composed of 2000 DNA sequences of length 101 nucleotides belonging to {A,C,G,T\}. This challenge brought us to explore kernel methods for classification and therefore implement several kernels and typical classifiers detailed in the following sections.

The **save_kernels.py** pyhton script allows to compute and save Gram matrixes for different kernels and is highly recommended if you are looking to do paramter optimization and thus use several times the different Gram matrixes. 

The **submit.py** python script allows to reproduce one of our best results.
