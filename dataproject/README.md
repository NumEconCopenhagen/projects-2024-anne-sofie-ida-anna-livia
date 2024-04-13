# Data analysis project

Our project is titled **An Ingvestigation of Danish Fertility**. We investigate the development of the crude birth rate in Denmark from 1967 to 2023 as well as its relationship to GDP-growth and the level of education among Danish women in the fertile age. 

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We apply the **following datasets**:

1. HISB3
2. NAN1
3. HFU1
4. HFUDD10
5. HFUDD11

The datasets are all fetched from Statistikbanken (Statistics Denmark) using an API. In the py-file functions are defined which define the fetching, the parameters and the storage of each of the datasets, we work with. 

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``, ``pip install mpl-axes-aligner`` & ``pip install git+https://github.com/alemartinello/dstapi``
