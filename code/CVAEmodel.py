import os
import scanpy as sc
import pandas as pd
from combat.pycombat import pycombat
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import layers
import random
import anndata as ad
def preprocess_and_plot_umap_metabolism(bulk_data, data, anno, metabolism_genes, save_path=None):
    """
    Preprocess gene expression data and generate UMAP images containing metabolic related genes

    Parameter:
    - bulk_data (DataFrame): DataFrame for batch data, with each column representing one sample
    - data (DataFrame): DataFrame for single cell data, where each row represents a gene and each column represents a cell
    - anno (DataFrame): DataFrame containing cell annotation information
    - metabolism_genes (DataFrame): DataFrame containing metabolic related genes, at least containing the 'gene name' column
    - save_path (str, optional): The path and file name for saving UMAP images. If not provided, it will be saved in the current path by default
    """
    print("Reading metabolic related genes")
    metabolism_genes = pd.read_csv(metabolism_genes)
    print("Successfully read")
    # Extracting metabolic related genes
    common_genes_metabolism = list(set(bulk_data.index).intersection(set(metabolism_genes['genename'])))
    bulk_metabolism = bulk_data.loc[common_genes_metabolism, :]
    data_metabolism = data.loc[common_genes_metabolism, :]
    num_common_genes_metabolism = len(common_genes_metabolism)
    print(f"Retained {num_common_genes_metabolism } metabolic related genes")
    # Create an AnaData object
    adata = sc.AnnData(X=data_metabolism.T)

    # Add cell annotation information to AnaData
    adata.obs['celltype'] = ''
    for idx, row in anno.iterrows():
        adata.obs.loc[row['cellname'], 'celltype'] = row['celltype']

    # pre-processing
    print("Start preprocessing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1500)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    print("Preprocessing completed...")

    # Save UMAP image
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "Orig_UMAP_Metabolism.png")
    else:
        save_path = os.path.join(os.getcwd(), save_path)

    print(f"Metabolism_ UMAP image will be saved to: {save_path}")
    sc.pl.umap(adata, color=['celltype'], legend_loc='on data', title='', frameon=True, show=False)
    plt.savefig(save_path)
    plt.close()
    layers_num = len(common_genes_metabolism)
    return bulk_metabolism,data_metabolism,common_genes_metabolism,layers_num


def prepare_train_val_data(data_metabolism, anno, batch_size):
    """
    Prepare training and validation set data, and generate batch data

    Parameter:
    - data_metabolism (DataFrame): DataFrame containing metabolic data, row name is cell name
    - anno (DataFrame): DataFrame containing cell annotation information needs to include the 'celltype' column
    - batch_size (int, optional): Batch size, default to 300

    Return:
    - train_data_batches (list): A list of training set batch data, each batch containing(batch, decoder_inputs)。
    - val_data_batches (list): A list of validation set batch data, each batch containing(batch, decoder_inputs)。
    """

    # Encoding cell types with unique heat
    enc = OneHotEncoder()
    cell_annotation_encoded = enc.fit_transform(anno[['celltype']])
    cell_annotation_encoded = pd.DataFrame(cell_annotation_encoded.toarray(),
                                           columns=enc.categories_[0])
    print('Divide the dataset into training and testing sets')
    # Assign row names to cells_ annotation_ encoded
    cell_annotation_encoded.index = data_metabolism.T.index

    # Merge metabolic data with cell type information encoded by unique heat
    data_metabolism_train = pd.concat([data_metabolism.T, cell_annotation_encoded], axis=1)

    # Divide training and validation sets
    train_data, val_data = train_test_split(data_metabolism_train, test_size=0.2, random_state=30)

    # Generate training set batch data
    train_data_batches = []
    for i in range(0, train_data.shape[0], batch_size):
        batch = train_data.iloc[i:i + batch_size].values
        decoder_inputs = batch[:, :-cell_annotation_encoded.shape[1]]
        train_data_batches.append((batch, decoder_inputs))

    # Generate validation set batch data
    val_data_batches = []
    for i in range(0, val_data.shape[0], batch_size):
        batch = val_data.iloc[i:i + batch_size].values
        decoder_inputs = batch[:, :-cell_annotation_encoded.shape[1]]
        val_data_batches.append((batch, decoder_inputs))
    return train_data_batches, val_data_batches

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, condition_dim,layers_num):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(layers_num,)),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu')
        ])

        # decoder
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim + condition_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(layers_num, activation='sigmoid')
        ])

    # Reparameterization
    def sampling(self, z_mean, z_log_var):
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        x = inputs[0]
        c = inputs[1]

        # Encoder output z_ Mean and z_ log_ var
        h = self.encoder(x)
        z_mean = layers.Dense(self.latent_dim)(h)
        z_log_var = layers.Dense(self.latent_dim)(h)

        # Reparameterization to obtain hidden vector z
        z = self.sampling(z_mean, z_log_var)

        # Splice hidden vectors and conditional vectors as decoder input
        z_cond = tf.concat([z, c], axis=1)

        # Decoder outputs reconstruction results
        recon_x = self.decoder(z_cond)

        return recon_x, z_mean, z_log_var


def get_condition_dim(cell_type_proportions_df):
    return len(cell_type_proportions_df.columns.tolist())


def train_cvae(train_data_batches, val_data_batches, latent_dim, condition_dim, epochs, learning_rate,layers_num):
    # Definition model
    cvae = CVAE(latent_dim, condition_dim,layers_num)
    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    # Training model
    train_loss_results = []
    val_loss_results = []
    best_val_loss = float("inf")
    best_model = None

    # Create progress bar
    progress_bar = tqdm(total=epochs, desc="Training Progress")

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()

        # At the beginning of each epoch, disrupt the order of training data
        np.random.shuffle(train_data_batches)

        for step, batch in enumerate(train_data_batches):
            x = batch[1]
            c = batch[0][:, -condition_dim:]

            # Calculate gradient and update parameters
            with tf.GradientTape() as tape:
                # Calculate reconstruction error
                recon_x, z_mean, z_log_var = cvae((x, c,layers_num))
                mse_loss = mse_loss_fn(x, recon_x)

                # Calculate KL divergence error
                kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                total_loss = mse_loss + 0.25 * kl_loss

            gradients = tape.gradient(total_loss, cvae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, cvae.trainable_variables))

            # Record the loss value for each step
            epoch_loss_avg.update_state(total_loss)

        train_loss_results.append(epoch_loss_avg.result().numpy())

        # Calculate loss values on the validation set
        val_loss_avg = tf.keras.metrics.Mean()
        for val_batch in val_data_batches:
            val_x = val_batch[1]
            val_c = val_batch[0][:, -condition_dim:]

            val_recon_x, val_z_mean, val_z_log_var = cvae((val_x, val_c,layers_num))
            val_mse_loss = mse_loss_fn(val_x, val_recon_x)
            val_kl_loss = -0.5 * tf.reduce_mean(1 + val_z_log_var - tf.square(val_z_mean) - tf.exp(val_z_log_var))
            val_total_loss = val_mse_loss + 0.25 * val_kl_loss

            val_loss_avg.update_state(val_total_loss)

        val_loss = val_loss_avg.result().numpy()
        val_loss_results.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = cvae

        # Update progress bar
        progress_bar.set_postfix({"Train Loss": epoch_loss_avg.result().numpy(), "Val Loss": val_loss})
        progress_bar.update(1)

    progress_bar.close()

    print("Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved best model to best_model.h5")

    return train_loss_results, val_loss_results, best_model
