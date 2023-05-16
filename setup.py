from setuptools import setup

def readme_file():
    with open("README.rst") as rf:
        return rf

setup(
    name='zsMet',
    version='1.0.0',
    author='Gw yang',
    author_email='stu_gwyang@163.com',
    description='Fits bulk RNA-seq data using single-cell RNA-seq (scRNA-seq) data, enabling accurate estimation and analysis of gene expression profiles and metabolic flux balance',
    long_description=readme_file(),
    url='https://github.com/your_username/your_package',
    packages=['zsMet'],
    install_requires=[
        'scanpy',
        'pandas',
        'combat',
        'numpy',
        'warnings',
        'matplotlib',
        'sklearn',
        'tqdm',
        'scipy',
        'tensorflow',
        'random',
        'anndata'
    ],
    python_requires='>=3.8',
)
