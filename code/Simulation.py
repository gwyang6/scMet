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
def simulate_bulk_data(cell_type_proportions_df, df, bulk_metabolism_ratio,common_genes_metabolism, num_simulations=20, num_cells=5000 ):
    selected_gene_expression_df_total = []
    selected_celltype = []
    selected_samplename = []

    for sample in range(cell_type_proportions_df.shape[0]):
        cell_type_proportions_df_tem = pd.DataFrame(cell_type_proportions_df.iloc[sample, :]).T
        col_sum = cell_type_proportions_df_tem.sum(axis=0)
        zero_cols = col_sum[col_sum == 0].index.tolist()
        cell_type_proportions_df_tem = cell_type_proportions_df_tem.loc[:,
                                       ~cell_type_proportions_df_tem.columns.isin(zero_cols)]
        print(f"Simulate {cell_type_proportions_df.index[sample]} {num_simulations} times")
        min_distance = float('inf')
        selected_gene_expression_df = None
        bulk_metabolism_ratio_tem = bulk_metabolism_ratio.iloc[:, sample]
        for i in range(num_simulations):
            cells_list = []
            seed = random.randint(1, 1000)

            for j in range(cell_type_proportions_df_tem.shape[1]):
                n_cells = (cell_type_proportions_df_tem.iloc[:, j] * num_cells).astype(int)
                cells = df[df['celltype'] == cell_type_proportions_df_tem.columns[j]].sample(n=n_cells.values)
                cells_list.append(cells)

            cells_df = pd.concat(cells_list)
            cells_label = cells_df.iloc[:, -1]
            cells_df = cells_df.iloc[:, :-1]

            gene_expression_df = cells_df.sum(axis=0)
            corr, pval = spearmanr(gene_expression_df, bulk_metabolism_ratio_tem)
            print(f"Round {i + 1} Spearman Correlation: {corr.mean():.5f}")

            gene_expression_df = gene_expression_df / gene_expression_df.sum()
            distance = np.linalg.norm(gene_expression_df.values - bulk_metabolism_ratio_tem.values)

            if distance < min_distance:
                min_distance = distance
                selected_gene_expression_df = cells_df
                selected_cells_label = cells_label
                selected_sample_name = pd.DataFrame([cell_type_proportions_df.index[sample]] * len(cells_df))
                Item = i
                best_corr = corr

            print(f"Round {i + 1} Euclidean Distance: {distance:.5f}")

        plt.scatter(selected_gene_expression_df.sum(axis=0).rank(), bulk_metabolism_ratio_tem.rank(), alpha=0.5)
        save_path = os.path.join(os.getcwd(),'output',f"{cell_type_proportions_df.index[sample]}_corr.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Save the correlation scatter plot to:{save_path}")

        selected_gene_expression_df_total.append(selected_gene_expression_df)
        selected_celltype.append(selected_cells_label)
        selected_samplename.append(selected_sample_name)
        print(
            f"Select the fitting result of the {Item + 1}th order,with a sample distance of {min_distance},and a correlation of {best_corr}")

    selected_gene_expression_df_total = pd.concat(selected_gene_expression_df_total, axis=0)
    selected_gene_expression_df_total.columns = common_genes_metabolism
    selected_gene_expression_df_total.index = ['Cell' + str(i) for i in
                                               range(1, selected_gene_expression_df_total.shape[0] + 1)]

    selected_celltype = pd.concat(selected_celltype, axis=0)
    selected_celltype = pd.DataFrame(selected_celltype)
    selected_celltype.index = ['Cell' + str(i) for i in range(1, selected_celltype.shape[0] + 1)]

    selected_samplename = pd.concat(selected_samplename, axis=0)
    selected_samplename.index = ['Cell' + str(i) for i in range(1, selected_samplename.shape[0] + 1)]

    return selected_gene_expression_df_total, selected_celltype, selected_samplename

def generate_umap_plots(selected_gene_expression_df_total, selected_celltype, selected_samplename, save_path=None):
    adata = ad.AnnData(X=selected_gene_expression_df_total)
    adata.obs['celltype'] = selected_celltype['celltype']
    adata.obs['sample'] = selected_samplename.iloc[:, 0]
    return adata