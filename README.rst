
scMet
=========
A package for deconvoluting Bulk RNAseq, using sc RNAseq to train CVAE models to generate new sc RNA seq data, and finally fitting Bulk RNAseq to obtain representative sc RANseq data

=========
.. image:: https://github.com/gwyang6/scMet/blob/main/Resource/scMet.png

scMet is a CVAE model trained using sc RNA seq data and generating new and reasonable sc RNA seq. Then, the generated data is used to fit Bulk RNA seq to obtain sc RNA seq data that can represent Bulk RNA seq data

Installation
============

1. Clone the repository.

   .. code-block:: bash

      git clone https://github.com/gwyang6/scMet.git

2. Navigate to the project directory.

   .. code-block:: bash

      cd scMet

3. Set up a virtual environment.

   .. code-block:: bash

      conda create -n scMET python=3.7
      conda activate scMET

4. Install the dependencies.

   .. code-block:: bash

      pip install -r requirements.txt

5. Help Activate

   .. code-block:: bash

      cd code
      python main.py -h

6. Run example data

   .. code-block:: bash

      python main.py

Dependencies
------------
- scanpy
- pandas
- combat
- numpy
- warnings
- matplotlib
- sklearn
- tqdm
- scipy
- tensorflow
- random
- anndata

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

4. Training CVAE Model for Generating Single-cell Data
--------------------------------------------------------

Train a CVAE (Conditional Variational Autoencoder) model using single-cell metabolic gene expression profiles and corresponding cell annotations. Use cell annotations as conditional input and randomize batch inputs. Record the training loss at each iteration. Add Adamw optimizer for backpropagation. Save a well-performing model for generating single-cell data.

5. Generating and Filtering Simulated Single-cell Data
-----------------------------------------------------

Use the trained CVAE-GAN model to generate a large number of new single-cell gene expression data and corresponding cell annotations. Filter the generated data based on the correlation with original single-cell data of different cell types. Select cells with correlation above a certain threshold as the source for fitting Bulk RNA-seq data in the next step.

6. Fitting Bulk RNA-seq Data using Selected Single-cell Data
-----------------------------------------------------------
