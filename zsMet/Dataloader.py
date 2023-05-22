import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt

def load_data(sc_file_path,anno_files,bulk_file_path):
    print("""Loading data from the specified path, please wait...""")
    sc_file_path = [sc_file_path]
    data_list = []
    for file in sc_file_path:
        data = pd.read_csv(file, index_col=0)
        data_list.append(data)
    data = pd.concat(data_list, axis=1)
    num_rows, num_columns = data.shape
    print(f"Loaded data from {file}")
    print(f"Gene number: {num_rows}")
    print(f"Cell number: {num_columns}")
    anno_list = []
    anno_files = [anno_files]
    for file in anno_files:
        anno = pd.read_csv(file)
        anno_list.append(anno)
    anno = pd.concat(anno_list)
    bulk_data = pd.read_csv(bulk_file_path, index_col=0)
    num_rows, num_columns = bulk_data.shape
    print("Loaded bulk data:")
    print(f"Gene number:{num_rows}")
    print(f"Sample number:{num_columns}")
    print("""Successfully loaded data from the specified path""")

    return data,anno,bulk_data


def filter_genes(bulk_data, sc_data, min_cells=20):
    print('Genes being screened Waiting...')
    common_genes = list(set(bulk_data.index).intersection(set(sc_data.index)))
    bulk_data = bulk_data.loc[common_genes]
    sc_data = sc_data.loc[common_genes]

    expr_counts = sc_data.astype(bool).sum(axis=1)
    expr_genes = expr_counts[expr_counts >= min_cells].index
    bulk_data = bulk_data.loc[expr_genes]
    sc_data = sc_data.loc[expr_genes]

    print(f'Genes retained as expressed in more than {min_cells} cells：{len(expr_genes)} genes')

    return bulk_data, sc_data

def preprocess_and_plot_umap(adata, anno, save_path=None):
    """
    Preprocessing of Annddata objects and generation of umap images。

    Parameters:
    - adata (AnnData): Annddata objects containing single-cell data
    - anno (DataFrame): Dataframe containing cellular annotation information
    - save_path (str, optional): Path and file name where umap images were saved. If not provided, save under the current path by default.
    """

    # Conversion to anddata
    adata = sc.AnnData(X=adata.T)

    # Adding cell annotation information to annddata
    adata.obs['celltype'] = ''
    for idx, row in anno.iterrows():
        adata.obs.loc[row['cellname'], 'celltype'] = row['celltype']

    # pre-processing
    print("Started preprocessing the data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    print("Preprocessing complete...")

    # Save UAMP images
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "Orig_UMAP.png")
    else:
        save_path = os.path.join(os.getcwd(), save_path)

    print(f"The umap image will be saved to: {save_path}")
    sc.pl.umap(adata, color=['celltype'], legend_loc='on data', title='', frameon=True, show=False)
    plt.savefig(save_path)
    plt.close()

    # Returns preprocessed Annddata objects
    return adata



