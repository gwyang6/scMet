# scMet
A package for deconvoluting Bulk RNAseq, using sc RNAseq to train CVAE models to generate new sc RNA seq data, and finally fitting Bulk RNAseq to obtain representative sc RANseq data
=========
scMet
=========
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT


ZsMet is a CVAE model trained using sc RNA seq data and generating new and reasonable sc RNA seq. Then, the generated data is used to fit Bulk RNA seq to obtain sc RNA seq data that can represent Bulk RNA seq data

Installation
============

1.Clone the repository.

```
git clone https://github.com/gwyang6/scMet.git
```

2.Navigate to the project directory.

```
cd zsMet
```
3.Set up a virtual environment.

```
python -m zsMet
```

```
source venv/bin/activate  # For Linux/Mac
```

```
venv\Scripts\activate  # For Windows
```

4.Install the dependencies.

```
pip install -r requirements.txt
```

5.Run

'''
python main.py
```

Dependencies
------------
- 'scanpy',
- 'pandas',
- 'combat',
- 'numpy',
- 'warnings',
- 'matplotlib',
- 'sklearn',
- 'tqdm',
- 'scipy',
- 'tensorflow',
- 'random',
- 'anndata'

Steps
=====

1. Reading and Merging Single-cell Gene Expression Data
------------------------------------------------------

Read multiple single-cell gene expression data files and corresponding cell type annotation files. Merge them into a large sample and randomly select cells to generate multiple simulated Bulk RNA-seq data.

2. Batch Correction of Simulated and Real Bulk RNA-seq Data
----------------------------------------------------------

Apply the Combat algorithm to perform batch correction on the simulated Bulk RNA-seq data and real Bulk RNA-seq data. Reduce the technical differences between scRNA-seq data and Bulk RNA-seq data. Obtain batch-corrected real Bulk RNA-seq data for deconvolution.

3. Preprocessing of Single-cell Gene Expression Data for Deconvolution
---------------------------------------------------------------------

Preprocess single-cell gene expression data. Use cell type annotation files to identify cell type-specific expressed genes and their expression levels. Solve the deconvolution problem by applying NNLS (Non-Negative Least Squares) on Bulk RNA-seq data using cell type-specific expressed genes and their expression levels, thus obtaining cell type proportions.

4. Training CVAE-GAN Model for Generating Single-cell Data
--------------------------------------------------------

Train a CVAE-GAN (Conditional Variational Autoencoder - Generative Adversarial Network) model using single-cell metabolic gene expression profiles and corresponding cell annotations. Use cell annotations as conditional input and randomize batch inputs. Record the training loss at each iteration. Add Adamw optimizer for backpropagation. Save a well-performing model for generating single-cell data.

5. Generating and Filtering Simulated Single-cell Data
-----------------------------------------------------

Use the trained CVAE-GAN model to generate a large number of new single-cell gene expression data and corresponding cell annotations. Filter the generated data based on the correlation with original single-cell data of different cell types. Select cells with correlation above a certain threshold as the source for fitting Bulk RNA-seq data in the next step.

6. Fitting Bulk RNA-seq Data using Selected Single-cell Data
-----------------------------------------------------------

Merge the selected simulated single-cell gene expression data and corresponding cell annotations with the original single-cell gene expression data. This merged dataset will be used as the source for fitting Bulk RNA-seq data. Use heuristic search algorithms like Simulated Annealing or Genetic Algorithm to select a subset of cells to reduce computational complexity. Consider weighting the evaluation metrics to quickly find cell subsets that satisfy various criteria. The fitting process includes:

   a. Select different cell subsets based on the cell type proportions obtained in Step 3. Each subset contains a specific number of cells (e.g., 10,000 cells).
   b. Calculate simulated Bulk RNA-seq data using the selected cell subsets and fit it with the real Bulk RNA-seq data. Evaluate the fitting quality by computing correlation, Euclidean distance, or other similarity metrics between the two datasets.
   c. Identify the Differentially Expressed Genes (DEGs) between the simulated and real Bulk RNA-seq data. Use differential expression analysis methods such as edgeR, DESeq2, etc. Calculate the proportion of DEGs among all genes between the two datasets.
   d. Perform Metabolic Functional Enrichment Analysis. Compare the consistency in metabolic activity pathways between the simulated and real Bulk RNA-seq data using KEGG metabolic gene sets.
   e. Use dynamic Flux Balance Analysis (dFBA) to assess whether the fitted sample's metabolic flux is approximately balanced.
