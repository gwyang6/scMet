import os
import scanpy as sc
import pandas as pd
from combat.pycombat import pycombat
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from tqdm import tqdm
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers
import random
import anndata as ad

def simulate_bulk_data_pre(data, bulk_data, num_simulations=10, num_cells=600):
    bulk_sim_list = []
    for i in range(num_simulations):
        selected_cells = data.sample(n=num_cells, axis=1)
        bulk_sim = pd.DataFrame(np.sum(selected_cells, axis=1), columns=['Counts'])
        bulk_sim_list.append(bulk_sim)

    dataset_1 = bulk_data
    dataset_2 = pd.concat(bulk_sim_list,axis=1)
    df_expression = pd.concat([dataset_1,dataset_2],join="inner",axis=1)
    plt.boxplot(df_expression)
    plt.show()
    batch = []
    datasets = [dataset_1,dataset_2]
    for j in range(len(datasets)):
         batch.extend([j for _ in range(len(datasets[j].columns))])
    df_corrected = pycombat(df_expression,batch)
    # visualise results
    plt.boxplot(df_corrected)
    plt.show()
    sample=(dataset_1.shape[1])
    df_corrected_output = df_corrected.iloc[:,:sample]
    return df_corrected_output

def normalize_columns_to_sum_one(data,common_genes_metabolism):
    normalized_data = np.zeros_like(data)

    for col in range(data.shape[1]):
        column_sum = data.iloc[:, col].sum()
        normalized_data[:, col] = data.iloc[:, col] / column_sum

    normalized_data = pd.DataFrame(normalized_data)
    normalized_data.index = data.index
    normalized_data.columns = data.columns
    normalized_data = normalized_data.loc[common_genes_metabolism,:]
    return normalized_data


def generate_synthetic_data(best_model, cell_type_proportions_df, common_genes_metabolism, data_metabolism, anno,
                            num_samples=20000, latent_dim=64, z_std=2.5, save_path=None):
    """
    Generate synthetic data samples.

    Parameter:
    - best_model (CVAE): A trained CVAE model
    - cell_type_proportions_df (DataFrame): The proportion of each cell type in each sample, represented by the DataFrame with the sample name as the row and the cell type as the column.
    - common_genes_metabolism (list): List of Common Metabolic Genes
    - metabolism_df (DataFrame): DataFrame representing metabolic genes
    - num_samples (int, optional): Generate sample quantity, default to 20000
    - latent_dim (int, optional): The dimension of latent variables, default to 64

    Return:
    - generated_data (DataFrame): Generated synthetic data samples, including metabolic gene information and cell type labels
    """

    # Randomly generated latent variables and conditions
    z_mean = np.zeros(latent_dim)
    z = np.random.normal(loc=z_mean, scale=z_std, size=(num_samples, latent_dim))

    # Calculate the number of each cell type
    cell_type_counts = anno.groupby('celltype').size()
    cell_type_proportions_sc = cell_type_counts / cell_type_counts.sum()
    cell_type_proportions_df_sc = cell_type_proportions_sc.to_frame(name='proportions').reset_index()
    cell_type_proportions_df_sc = cell_type_proportions_df_sc.set_index('celltype')
    cell_type_proportions_df_transposed_sc = cell_type_proportions_df_sc.T
    cell_type_proportions_df_transposed_sc.reset_index(drop=True, inplace=True)
    cell_types_list_sc = cell_type_proportions_df_transposed_sc.columns
    cell_types = np.random.choice(cell_types_list_sc, size=num_samples, p=cell_type_proportions_sc.values)

    # Encoding cell types with OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    c = encoder.fit_transform(cell_types.reshape(-1, 1))

    # Merge latent variables and conditions into a tensor
    z_c = np.hstack([z, c])

    # Input CVAE model to generate sample data and cell type annotations
    x_pred, y_pred = best_model.decoder.predict(z_c), np.argmax(c, axis=1)
    print(f"Cell generation completed, total generation: {num_samples} cells")

    # Save the generated data and labels as DataFrame
    df = pd.DataFrame(x_pred)
    df['celltype'] = y_pred
    labels = cell_type_proportions_df.columns.tolist()
    label_dict = {i: label for i, label in enumerate(labels)}
    df['celltype'] = df['celltype'].replace(label_dict)
    generated_data = pd.DataFrame(x_pred, columns=common_genes_metabolism)
    print("Continue processing the generated data")
    # Process the generated data
    value_proportions = data_metabolism.T.apply(lambda x: x.value_counts(normalize=True))

    for col in generated_data.columns:
        # Filter out data with a ratio of 0 and sort in descending order
        proportions = value_proportions[col][value_proportions[col] > 0.0001].sort_index(ascending=False)
        values = proportions.index.values
        cum_proportions = np.cumsum(proportions.values)

        # Calculate the number of elements that should be assigned to each value based on proportion
        num_elements_per_value = (cum_proportions * generated_data.shape[0]).astype(int)
        num_elements_diff = np.diff(np.concatenate(([0], num_elements_per_value)))

        # Correct the number of the last element to match the number
        num_elements_diff[-1] = generated_data.shape[0] - np.sum(num_elements_diff[:-1])

        # Generate a list of all values based on the number of elements and arrange them in descending order
        all_values = np.repeat(values, num_elements_diff)

        # For generated_ Sort the current column of data in descending order and obtain the sorted index
        sorted_indices = np.argsort(-generated_data.loc[:, col])

        # Assign the arranged values to generated_ Data, keep row order unchanged
        generated_data.loc[sorted_indices, col] = all_values
    print(
        "Processing completed, conducting simulation data quality analysis. Please try different standard deviation values based on the UMAP clustering diagram")
    adata = ad.AnnData(X=generated_data, obs={'cell_type': df['celltype'].values})
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata)
    # Save UMAP image
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "Simulation_UMAP_Metabolism_std_" + str(z_std) + ".png")
    else:
        save_path = os.path.join(os.getcwd(), save_path)

    print(f"Total generated: {num_samples} cells")
    print(f"Metabolism_ UMAP image will be saved to: {save_path}")
    print(f"The current standard deviation value is: {z_std},Please adjust according to the UMAP clustering effect")

    sc.pl.umap(adata, color=['cell_type'], legend_loc='on data', title='', frameon=True, show=False)
    plt.savefig(save_path)
    plt.close()
    sc.pl.umap(adata, color=['cell_type'], legend_loc='on data', title='', frameon=True)
    generated_data = pd.DataFrame(x_pred, columns=common_genes_metabolism)
    value_proportions = data_metabolism.T.apply(lambda x: x.value_counts(normalize=True))

    for col in generated_data.columns:
        # Filter out data with a ratio of 0 and sort in descending order
        proportions = value_proportions[col][value_proportions[col] > 0.0001].sort_index(ascending=False)
        values = proportions.index.values
        cum_proportions = np.cumsum(proportions.values)

        # Calculate the number of elements that should be assigned to each value based on proportion
        num_elements_per_value = (cum_proportions * generated_data.shape[0]).astype(int)
        num_elements_diff = np.diff(np.concatenate(([0], num_elements_per_value)))

        # Correct the number of the last element to match the number
        num_elements_diff[-1] = generated_data.shape[0] - np.sum(num_elements_diff[:-1])

        # Generate a list of all values based on the number of elements and arrange them in descending order
        all_values = np.repeat(values, num_elements_diff)

        # Sort the current column of generated_data in descending order and obtain the sorted index
        sorted_indices = np.argsort(-generated_data.loc[:, col])

        # Assign the arranged values to XX, keeping the row order unchanged
        generated_data.loc[sorted_indices, col] = all_values

    return generated_data, df

def calculate_gene_expression_correlations(data_metabolism, common_genes_metabolism,anno,generated_data,generated_labels,cell_type_proportions_df):
    # Convert raw data and generated data to DataFrame
    adata_raw_data = pd.DataFrame(data_metabolism.T, columns=common_genes_metabolism)
    adata_raw_cell_types = pd.DataFrame(anno, columns=['celltype'])
    adata_raw_cell_types.index = adata_raw_data.index

    generated_data_df = pd.DataFrame(generated_data, columns=common_genes_metabolism)
    generated_cell_types_df = pd.DataFrame(generated_labels, columns=['celltype'])

    # Merge raw data and generate data
    adata_raw = pd.concat([adata_raw_data, adata_raw_cell_types], axis=1)
    adata_generated = pd.concat([generated_data_df, generated_cell_types_df], axis=1)

    # Calculate the correlation of gene expression profiles for various cell types
    cell_types = cell_type_proportions_df.columns
    correlations = {}

    for cell_type in cell_types:
        # Extract raw data and generate data for specific cell types
        raw_cell_type_data = adata_raw[adata_raw['celltype'] == cell_type].drop('celltype', axis=1)
        generated_cell_type_data = adata_generated[adata_generated['celltype'] == cell_type].drop('celltype', axis=1)

        # Calculate the average of gene expression profiles between raw and generated data
        raw_mean = raw_cell_type_data.mean(axis=0)
        generated_mean = generated_cell_type_data.mean(axis=0)

        # Calculate Spearman correlation coefficient
        corr, _ = spearmanr(raw_mean, generated_mean)

        correlations[cell_type] = corr

    return correlations

