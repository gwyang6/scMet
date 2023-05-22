from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import ipywidgets as widgets
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



def loop_choose_std(best_model, cell_type_proportions_df, common_genes_metabolism, data_metabolism, anno,
                            num_samples=3000, latent_dim=64, z_std=1, save_path=None):

    z_std_list = [(z_std/2),(z_std),(z_std+z_std/2)]
    img = []
    for z_std in z_std_list:
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
        print( "Processing completed, conducting simulation data quality analysis. Please try different standard deviation values based on the UMAP clustering diagram")
        adata = ad.AnnData(X=generated_data, obs={'cell_type': df['celltype'].values})
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=1000)
        sc.pp.scale(adata)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata)
        save_path = os.path.join(os.getcwd(), "Simulation_UMAP_Metabolism_std_" + str(z_std) + ".png")
        print(f"Total generated: {num_samples} cells")
        print(f"Metabolism_ UMAP image will be saved to: {save_path}")
        print(f"The current standard deviation value is: {z_std},Please adjust according to the UMAP clustering effect")
        std_str = str(z_std)
        sc.pl.umap(adata, color=['cell_type'], legend_loc='on data',title='The current std is '+std_str,frameon=True, show=False)
        img.append(save_path)
        plt.savefig(save_path)
        plt.close()

    return img,z_std_list


def select_image(img_list):
    images = []
    for img_path in img_list:
        image = Image.open(img_path)
        images.append(image)

    fig, axs = plt.subplots(1, len(images), figsize=(10, 6))  # 调整子图大小
    fig.tight_layout()

    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')

    plt.show()

    selected_index = None
    selected_type = None
    while selected_index is None:
        user_input = input("Please select the image serial number (input serial number) or enter a new standard deviation (input s) for the graph: ")
        if user_input == "s":
            new_std = float(input("Please enter a new standard deviation value: "))
            selected_index = new_std
            selected_type = "std"
        else:
            try:
                selected_index = int(user_input) - 1
                if selected_index < 0 or selected_index >= len(images):
                    print("Invalid image sequence number, please re-enter")
                    selected_index = None
            except ValueError:
                print("Invalid input, please re-enter")

    if selected_type == "std":
        print(f"The new standard deviation you have selected is {selected_index}")
        return selected_index, selected_type
    else:
        print(f"The serial number of the image you selected is {selected_index + 1}")
        return selected_index, selected_type
