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
import argparse

from Dataloader import load_data,filter_genes,preprocess_and_plot_umap
from CVAEmodel import preprocess_and_plot_umap_metabolism,prepare_train_val_data,CVAE,get_condition_dim,train_cvae
from Deconvolution import scale_data,compute_cell_type_proportions
from GenerateData import simulate_bulk_data_pre,normalize_columns_to_sum_one,generate_synthetic_data,calculate_gene_expression_correlations
from Simulation import simulate_bulk_data,generate_umap_plots
from Std_choose import loop_choose_std,select_image


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='ScMet: A package used to generate sc RNA seq data that can represent Bulk RNA seq data, involving NNLS and CVAE')

parser.add_argument('--sc_data_dir', nargs='?',  type=str, default=None,
                    help='List of file names for storing single cell data.')
parser.add_argument('--sc_data_anno', nargs='?', type=str, default=None, metavar='<input_directory>',
                    help='List of cell annotation information file locations for storing single cell data pairing.')
parser.add_argument('--bulk_data', nargs='?', type=str, default=None, metavar='<data_directory>',
                    help='Bulk RNA seq data file location.')
parser.add_argument('--Meta_gene_dir', nargs='?', type=str, default= None,
                    help='Metabolism-related gene information file storage location.')
parser.add_argument('--learning_rate', nargs='?', type=float, default=0.0001,
                    help='Controlling the magnitude of parameter updates at each step.')
parser.add_argument('--epochs', nargs='?', type=int, default=50,
                    help='User defined EPOCH (training iteration).')
parser.add_argument('--batch_size', nargs='?', type=int, default=1440,
                    help='Number of cells per training.')
parser.add_argument('--Rm_batch_cell_number', nargs='?', type=int, default=2000,
                    help='Generate simulated Bulk RNA seq data and use it to remove the number of cells from the batch.')
parser.add_argument('--Rm_batch_sample_number', nargs='?', type=int, default=10,
                    help='Generate simulated Bulk RNA seq data and use it for sample size removal from batches.')
parser.add_argument('--Number_gene_deconvolution', nargs='?', type=int, default=250,
                    help='Number of genes retained by various cells for deconvolution.')
parser.add_argument('--Number_sm', nargs='?', type=int, default=3000,
                    help='Number of cells producing simulated small sample single-cell data.')
parser.add_argument('--Number_Nr', nargs='?', type=int, default=100000,
                    help='Number of cells producing simulated normal sample single-cell data.')
parser.add_argument('--latent_dim', nargs='?', type=int, default=64,
                    help='Refers to the dimensionality of the latent space in a machine learning model.')
parser.add_argument('--Std', nargs='?', type=int, default=1,
                    help='Standard deviation for single-cell data generation, multiple adjustments required.')
parser.add_argument('--num_simulations', nargs='?', type=int, default=20,
                    help='Number of times to fit Bulk RNA seq data using single cell data.')
parser.add_argument('--num_simulations_cell', nargs='?', type=int, default=5000,
                    help='The number of cells used to fit Bulk RNA seq data using single cell data.')

arg = parser.parse_args()

sc_data_dir = arg.sc_data_dir
if sc_data_dir is None:
    current_path = os.getcwd()
    parent_directory = os.path.dirname(current_path)
    resource_directory = os.path.join(parent_directory, 'Resource')
    sc_data_dir = os.path.join(resource_directory, 'CS1.csv')

sc_data_anno = arg.sc_data_anno
if sc_data_anno is None:
    current_path = os.getcwd()
    parent_directory = os.path.dirname(current_path)
    resource_directory = os.path.join(parent_directory, 'Resource')
    sc_data_anno = os.path.join(resource_directory, 'AN1.csv')

bulk_data_dir = arg.bulk_data
if bulk_data_dir is None:
    current_path = os.getcwd()
    parent_directory = os.path.dirname(current_path)
    resource_directory = os.path.join(parent_directory, 'Resource')
    bulk_data_dir = os.path.join(resource_directory, 'Bulk_1.csv')


Meta_gene = arg.Meta_gene_dir
if Meta_gene is None:
    current_path = os.getcwd()
    parent_directory = os.path.dirname(current_path)
    resource_directory = os.path.join(parent_directory, 'Resource')
    Meta_gene = os.path.join(resource_directory, 'metabolism_gene_total.csv')

Rm_batch_cell_number = arg.Rm_batch_cell_number
Rm_batch_sample_number = arg.Rm_batch_sample_number
Number_gene_deconvolution = arg.Number_gene_deconvolution
batch_size = arg.batch_size
epochs = arg.epochs
latent_dim = arg.latent_dim
learning_rate = arg.learning_rate
Number_sm = arg.Number_sm
Number_Nr = arg.Number_Nr
Std = arg.Std
num_simulations = arg.num_simulations
num_simulations_cell = arg.num_simulations_cell

print('Step 1: Import and preprocess the data')
sc_data,anno,bulk_data= load_data(sc_data_dir,sc_data_anno,bulk_data_dir)
bulk_data,sc_data= filter_genes(bulk_data,sc_data)
adata = preprocess_and_plot_umap(sc_data,anno)

print('Step 2: Solve for cell specific expression genes and deconvolute the proportion of cells')
scaled_matrix = scale_data(bulk_data)
cell_type_proportions_df = compute_cell_type_proportions(adata,scaled_matrix,Number_gene_deconvolution)

print('Step 3: Preserve metabolic related genes, generate training data, and train the CAVE model')
bulk_metabolism,data_metabolism,common_genes_metabolism,layers_num = preprocess_and_plot_umap_metabolism(bulk_data,sc_data,anno,Meta_gene)
train_data_batches,val_data_batches = prepare_train_val_data(data_metabolism,anno,batch_size)
batch_rm_data = simulate_bulk_data_pre(sc_data,bulk_data,Rm_batch_sample_number,Rm_batch_cell_number)
sc_normalized_data = normalize_columns_to_sum_one(batch_rm_data,common_genes_metabolism)
condition_dim = get_condition_dim(cell_type_proportions_df)
train_loss_results, val_loss_results, best_model = train_cvae(train_data_batches, val_data_batches, latent_dim, condition_dim, epochs, learning_rate,layers_num)

print('Step 4: Generate single cell data using the trained CVAE model')
print('Now we will generate small sample data for you based on different standard deviations. Please choose the existing standard deviation or input a new standard deviation based on the quality of the UMAP graph clustered by the generated data to achieve the best generation effect')
selected_index = Std
selected_type = 'std'
while selected_type is not None:
    img, z_std_list = loop_choose_std(best_model, cell_type_proportions_df, common_genes_metabolism, data_metabolism,
                                      anno, Number_sm, latent_dim, selected_index, None)
    selected_index, selected_type = select_image(img)
Std = z_std_list[selected_index]
generated_data, df = generate_synthetic_data(best_model, cell_type_proportions_df, common_genes_metabolism, data_metabolism, anno, Number_Nr, latent_dim, Std)

print('Step 5: Using generated data for fitting')
selected_gene_expression_df_total,selected_celltype,selected_samplename = simulate_bulk_data(cell_type_proportions_df,df, sc_normalized_data,common_genes_metabolism,num_simulations,num_simulations_cell)
adata= generate_umap_plots(selected_gene_expression_df_total, selected_celltype, selected_samplename)





