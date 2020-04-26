# Struct2Graph
A PyTorch implementation of GCN with mutual attention for protein-protein interaction prediction

![Proposed GCN architecture with Mutual Attention Mechanism](approach.png)

We use SMILES representation of query molecules to generate relevant fingerprints, which are then fed to the GCN/RF architecture for producing binary labels corresponding to each of the 11 metabolic pathway classes. The details of the GCN and RF architectures are described in our paper (currently under review).

A dataset of 6669 compounds belonging to one or more of these 11 constituent pathway classes was downloaded (February 2019) from the KEGG database: https://www.genome.jp/kegg/pathway.html.

### Requirements
* PyTorch
* scikit-learn
* RDKit
* Jupyter Notebook

### Usage
We provide two notebook files, one each for the multi-class GCN classifier and the multi-class RF classifier. The notebooks are self-sufficient and various relevant details have been marked in the files themselves.

### Contact
Contact: <a href="https://web.eecs.umich.edu/~mayankb/">Mayank Baranwal, Postdoctoral Fellow, University of Michigan at Ann Arbor</a>

### Acknowledgements
Part of the code was adopted from [1], and suitably modified for the pathway prediction task.

### References
1. Baranwal, Mayank, Abram Magner, Paolo Elvati, Jacob Saldinger, Angela Violi, and Alfred Hero. "A deep learning architecture for metabolic pathway prediction." Bioinformatics (2019)
