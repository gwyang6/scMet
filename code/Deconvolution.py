import os
import scanpy as sc
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from tqdm import tqdm
from scipy.optimize import nnls
import anndata as ad
def scale_data(exp):
    """
    Parameter:
    - exp (pd.DataFrame): FPKM Expression

    Return:
    - scaled_matrix (pd.DataFrame)
    """

    # Expression data were log10 transformed
    exp_log = np.log10(exp + 1)

    # Standardization
    print("Starting with standardized data...")
    scaled_matrix = exp_log.apply(lambda x: scale(x.values), axis=1, result_type='expand')
    scaled_matrix[np.isnan(scaled_matrix)] = 0

    # Set the names of the rows and columns
    scaled_matrix.index = exp.index
    scaled_matrix.columns = exp.columns

    print("Data normalization is complete!")

    return scaled_matrix

def compute_cell_type_proportions(adata, bulk_data, num_top_genes=50):
    """
    Specific expression genes were solved based on each cell type, and the proportion of each cell type in the sample was calculated.

    Parameter:
    - adata (AnnData): Annddata objects containing single-cell data
    - bulk_data (DataFrame): Dataframe containing bulk data, with each column representing one sample
    - num_top_genes (int, optional): Number of specific genes selected per cell type, default 50

    Return:
    - cell_type_proportions_df (DataFrame): The proportion of each cell type in each sample, with the sample name as the row and the cell type as the column of the dataframe
    """

    # Specifically expressed genes were solved based on each cell type
    print('Solve for each cell type specific expression gene...')
    sc.tl.rank_genes_groups(adata, groupby='celltype', method='t-test')
    all_genes = []
    for ct in adata.obs['celltype'].unique():
        top_genes = adata.uns['rank_genes_groups']['names'][ct][:num_top_genes]
        all_genes.extend(top_genes)

    # Screen for specific expression genes of each cell type, retaining genes that are only specific to that cell type and not specific to other types of cells
    specific_genes = np.unique(all_genes)
    num_specific_genes = len(specific_genes)
    print(f"Retained {num_specific_genes} specific genes")
    # Obtain the gene expression matrix of the reference cell type from the AnaData object
    ref_data = adata[:, specific_genes].X.T
    ref_cell_types = adata.obs['celltype'].cat.categories.values
    ref_cell_types_data = np.zeros((len(specific_genes), len(ref_cell_types)))
    for i, cell_type in enumerate(ref_cell_types):
        cell_type_data = adata[adata.obs['celltype'] == cell_type, specific_genes].X.mean(axis=0)
        ref_cell_types_data[:, i] = cell_type_data

    # Initialize the DataFrame that saves the proportion of cell types
    cell_type_proportions_df = pd.DataFrame(index=bulk_data.columns, columns=ref_cell_types)

    # Cycle to calculate the proportion of cell types for each sample
    for col in bulk_data.columns:
        bulk_gene_counts = bulk_data.loc[specific_genes][col].values

        # NNLS deconvolution of reference gene expression data
        cell_type_proportions, _ = nnls(ref_cell_types_data, bulk_gene_counts)

        # Save cell type ratio to DataFrame
        cell_type_proportions_df.loc[col] = cell_type_proportions

    # Normalize the cell proportion data so that the sum of the proportions for each sample is one
    cell_type_proportions_df = cell_type_proportions_df.div(cell_type_proportions_df.sum(axis=1), axis=0)
    visualize_cell_type_proportions(cell_type_proportions_df.astype(float))
    return cell_type_proportions_df

def visualize_cell_type_proportions(cell_type_proportions_df):
    """
    Visualize the proportion of each cell type in each sample

    Parameters:
    - cell_type_proportions_df (DataFrame): The proportion of each cell type in each sample, DataFrame with sample name as row and cell type as column
    """

    # Set Drawing Style
    plt.style.use('seaborn-dark')

    # Obtain the number of samples and cell types
    num_samples, num_cell_types = cell_type_proportions_df.shape

    # Set Color Mapping
    colors = plt.cm.get_cmap('tab20c', num_cell_types)

    # Create drawings and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw Stacked bar chart
    bottom = np.zeros(num_samples)
    for i, cell_type in enumerate(cell_type_proportions_df.columns):
        proportions = cell_type_proportions_df[cell_type]
        ax.bar(range(num_samples), proportions, bottom=bottom, color=colors(i), label=cell_type)
        bottom += proportions

    # Set graphic titles and labels
    ax.set_title("Cell Type Proportions")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Proportions")

    # add legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Automatically adjust subgraph layout
    fig.tight_layout()

    # display graphics
    plt.show()
