import itertools
import sys
import time
from bisect import bisect
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy
from datasets import IterableDataset
from torch.utils.data import Dataset, DataLoader

sys.path.append('hest-src/src/hest')

import datasets
import pandas as pd
from hest import load_hest, HESTData

#local_dir='hest_data' # hest will be dowloaded to this folder
local_dir = '/fast/home/cosmo/hest/hest_data'                                 # hest will be dowloaded to this folder

#meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_0_0.csv")
meta_df = pd.read_csv('/fast/home/cosmo/hest/HEST_v1_0_0.csv')



PATCHES_DIRECTORY = Path('/fast/home/cosmo/hest/hest_data/patches/')

class SpatialPathReader:

    def __init__(self, hest_data: HESTData):
        self.hest_data = hest_data

        self.gene_filter = None             # will be set to an array of indexes once the gene set is selected
        self.gene_mean_counts = None        # for each gene in the gene_filter list, stores mean count for this given slide
        self.gene_count_medians = None

        '''
        v.var_vector('AAACCGTTCGTCCAGG-1')          -> returns array of shape (33538,)
        d.adata.obs_vector('MYL9')                  -> returns array length=num_spots
        '''

        # have the patches been dumped yet? Find the H5 file where we expect it
        self.h5_path = PATCHES_DIRECTORY / f'{self.id}.h5'
        self.num_patches = 0
        self.read_count = 0
        self.h5 = None
        if not self.h5_path.exists():
            print(f'Cannot find {str(self.h5_path)}')
            return

        with h5py.File(str(self.h5_path)) as h5:
            self.num_patches, H, W, c = h5['img'].shape

        #self.cycle = itertools.cycle(iter(self))

        #### Standardize gene symbols


        # some datasets prefix gene symbols with "GRCh38______"
        # some other datasets don't use all-caps symbols, so we uppercase
        df = hest_data.adata.var.copy()
        df['index'] = df.index.str.replace('GRCh38______', '').str.upper()
        df = df.set_index('index')
        #print(df)
        if 'level_0' in df: del df['level_0']
        self.genes_df = df

        #return df

    def open_h5(self):
        if self.read_count > 128:
            self.h5.close()
            self.h5 = None
            self.read_count = 0
            print('closing')

        if not self.h5:
            self.h5 = h5py.File(str(self.h5_path))

        self.read_count += 1
        return self.h5

    def __getitem__(self, item):
        h5 = self.open_h5()
        if self.gene_filter is None:
            raise ValueError('Gene filter has not been specified yet')
        patch = h5['img'][item]
        barcode = h5['barcode'][item].astype('U')[0]           # numpy object arrays sure do suck ...
        gene_counts = self.get_gene_vector(barcode)
        #return patch, gene_counts

        gene_present = np.clip(gene_counts, 0, 1).astype(np.float32)
        return patch, gene_present

        #gene_high_low = 1*(gene_counts > self.gene_count_medians)    # should be elementwise
        #gene_high_low = gene_high_low.astype(np.float32)
        #gene_high_low = 2*gene_high_low - 1
        #print(f'Expression vector : {gene_high_low.sum()}/{gene_high_low.size}')
        #return patch, gene_high_low

        gene_counts = gene_counts.astype(np.float32)
        return patch, gene_counts

    def __len__(self):
        '''makes sense to define it as the number of spots, IE, tiles'''
        #return len(self.spots_df)
        return self.num_patches

    def __repr__(self):
        return f'SpatialPathology reader class with {len(self)} tiles'

    #def __next__(self):
    #    return next(self.cycle)

    def select_genes(self, gene_list):
        '''Given a list of genes, in order, this will look each up and create the gene filter'''

        self.genes_df['idx'] = range(len(self.genes_df))            # map gene_symbol to its position in the dataframe
        T = type(self.hest_data.adata.X)
        if T==np.ndarray:
            median_counts = np.median(self.hest_data.adata.X, axis=0)         # median counts for each gene
        elif T==scipy.sparse._csr.csr_matrix or T==scipy.sparse._csc.csc_matrix:
            median_counts = np.median(self.hest_data.adata.X.toarray(), axis=0)         # median counts for each gene
        else:
            raise ValueError(f'Unexpected array type : {T}')

        '''
        self.gene_filter = np.zeros(len(gene_list), dtype=np.int32)
        self.gene_mean_counts = np.zeros(len(gene_list), dtype=np.float32)
        self.gene_count_medians = np.zeros(len(gene_list), dtype=np.float32)

        for i, gene_symbol in enumerate(gene_list):
            row = self.genes_df.loc[gene_symbol]
            self.gene_filter[i] = row.idx
            self.gene_mean_counts[i] = row.mean_counts
            self.gene_count_medians[i] = median_counts[row.idx]
        '''

        selection = self.genes_df.loc[gene_list]
        self.gene_filter = selection.idx.to_numpy()
        self.gene_mean_counts = selection.mean_counts.to_numpy()
        self.gene_count_medians = median_counts[self.gene_filter]

        assert len(gene_list) == len(self.gene_filter)
        assert len(gene_list) == len(self.gene_mean_counts)
        assert len(gene_list) == len(self.gene_count_medians)

    def get_gene_vector(self, spot_barcode):
        full_vector = self.hest_data.adata.var_vector(spot_barcode)
        filtered = full_vector[self.gene_filter]
        return filtered

    """
    @property
    def genes_df(self):
        ''' Columns :
        gene_ids
        feature_types
        genome
        mito
        n_cells_by_counts
        mean_counts
        log1p_mean_counts
        pct_dropout_by_counts
        total_counts
        log1p_total_counts
        '''
        return self.hest_data.adata.var
    """

    @property                   # this is a dictionary of the columns that were in the hest metadata dataframe
    def meta(self): return self.hest_data.meta

    @property   # a dataframe of information about each spot, indexed by DNA barcode
    def spots_df(self):
        ''' Columns returned :
            in_tissue
            array_row
            array_col
            pxl_col_in_fullres
            pxl_row_in_fullres
            n_genes_by_counts
            log1p_n_genes_by_counts
            total_counts
            log1p_total_counts
            pct_counts_in_top_50_genes
            pct_counts_in_top_100_genes
            pct_counts_in_top_200_genes
            pct_counts_in_top_500_genes
            total_counts_mito
            log1p_total_counts_mito
            pct_counts_mito]
                '''
        return self.hest_data.adata.obs

    @property
    def id(self): return self.meta['id']



class IterableSpatialPathologyDataset(IterableDataset):

    def __init__(self, readers, gene_set, transform=None):

        self.transform = transform
        self.gene_list = list(gene_set)
        self.readers = []
        for reader in readers:
            reader: SpatialPathReader
            try:
                reader.select_genes(self.gene_list)
                self.readers.append(reader)
            except Exception as e:
                print(e)

        self.num_tiles = sum([len(reader) for reader in readers])

        self.reader_cycle = itertools.cycle(self.readers)

    @property
    def num_genes(self):
        return len(self.gene_list)

    def __len__(self):
        return self.num_tiles

    def __repr__(self):
        return f'SpatialPathology Dataset with {len(self.readers)} slides and {self.num_tiles} total tiles'

    def __iter__(self):
        while True:
            # get next reader
            reader = next(self.reader_cycle)
            # get next tile from that reader
            tile, gene_expressions = next(reader)

            if self.transform:
                tile = self.transform(tile)

            yield tile, gene_expressions


from torchvision.transforms import transforms


class SpatialPathologyDataset(Dataset):

    def __init__(self, readers, gene_set):

        self.transform = None
        self.gene_list = list(gene_set)
        self.readers = []
        for reader in readers:
            reader: SpatialPathReader
            try:
                reader.select_genes(self.gene_list)
                self.readers.append(reader)
            except Exception as e:
                print(e)

        self.tile_counts = [len(reader) for reader in readers]
        self.num_tiles = sum(self.tile_counts)

        # stores tile ranges to allow mapping between global tile_id and each readers tile_id
        self.reader_start_tiles = []
        self.reader_end_tiles = []

        global_tile_id = 0
        for n_tiles in self.tile_counts:
            self.reader_start_tiles.append(global_tile_id)
            global_tile_id += n_tiles
            self.reader_end_tiles.append(global_tile_id)

        assert len(self.readers) == len(self.reader_start_tiles)
        assert len(self.readers) == self.num_slides
        assert self.num_slides == len(self.reader_end_tiles)

    def make_transform(self, mean, std):
        #mean = mean or (0.485, 0.456, 0.406)
        #std = std or (0.229, 0.224, 0.225)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45.0),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        print(f'Transform is : {self.transform}')

    @property
    def num_genes(self):
        return len(self.gene_list)

    @property
    def num_slides(self):
        return len(self.readers)

    def __len__(self):
        return self.num_tiles

    def __repr__(self):
        return f'SpatialPathology Dataset with {len(self.readers)} slides and {self.num_tiles} total tiles'

    def __getitem__(self, item):
        slide_id = bisect(self.reader_end_tiles, item)
        local_tile_id = item - self.reader_start_tiles[slide_id]
        tile, expressions = self.readers[slide_id][local_tile_id]
        if self.transform:
            tile = self.transform(tile)
        return tile, expressions



def assemble_datasets(selection_df):
    ids_to_query = selection_df['id'].values

    symbol_sets = []
    readers = []
    running_intersection = None
    counter = Counter()
    for ds_id in ids_to_query:

        try:
            hest_data = load_hest(local_dir, id_list=[ds_id])  # location of the data
        except Exception as e:
            print(e)
            continue

        for d in hest_data:
            d: HESTData
            reader = SpatialPathReader(d)

            # to start with, lets assemble datasets wherein each gene has at least one count in each slide
            df = reader.genes_df

            # another idea for later, topk genes by counts
            #df = df.sort_values('total_counts', ascending=False).head(15000)

            #symbols = set(df[df.total_counts > 0].index)

            symbols = set(df.index)
            counter.update(symbols)
            print(f'Number of unique symbols : {len(symbols)}')
            if len(symbols) == 0:
                print(f'No symbols in {d.meta.get("id")}')
                continue

            if running_intersection is None:            # for first iteration
                running_intersection = symbols

            # does this dataset use a different set of symbols? We should skip
            if len(symbols.intersection(running_intersection)) < 100:
                print(f'Problematic gene set in dataset {d.meta.get("id")}')
                print(f'Gene IDs look like : {list(symbols)[:15]}')
                continue

            running_intersection.intersection_update(symbols)
            #if len(running_intersection)==0:
            #    print(symbols)
            #    raise ValueError('Prolematic gene set')

            symbol_sets.append(symbols)
            readers.append(reader)


    # now initialize the full dataset with this list of readers and selected gene list
    shared_symbols = set.intersection(*symbol_sets)
    print(f'Found {len(shared_symbols)} genes in common among datasets')


    print(f'{"="*20} MOST COMMON {"="*20}')
    print(counter.most_common(100))

    #print(f'{"="*20} LEAST COMMON {"="*20}')
    #print(counter.)

    #for tile, exp in iter(dataset):
    #    #print(exp.min(), exp.mean(), exp.max(), exp.shape)
    #    print(exp.mean(), len(exp), exp)

    print('Now initializing dataset with shared geneset - may take some time')
    dataset = SpatialPathologyDataset(readers, shared_symbols)
    print(dataset)
    return dataset


#D = meta_df.head(10)
D = meta_df[meta_df['organ'] == 'Bladder']
#D = meta_df[meta_df['organ'].isin('Bladder Breast Heart Muscle Eye Uterus'.split())]
#D = meta_df[meta_df.st_technology == 'Visium']
#D = D.sample(frac=1).head(10)
dataset = assemble_datasets(D)




def test_single_wsi():
    print('load hest...')

    meta_df = pd.read_csv('/fast/home/cosmo/hest/HEST_v1_0_0.csv')
    # Filter the dataframe by organ, oncotree code...
    # meta_df = meta_df[meta_df['oncotree_code'] == 'IDC']
    # meta_df = meta_df[meta_df['organ'] == 'Breast']
    meta_df = meta_df[meta_df['organ'] == 'Bladder']

    ids_to_query = meta_df['id'].values
    ids_to_query = ids_to_query[:2]

    # list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    # print(f'list_patterns : {list_patterns}')

    hest_data = load_hest(local_dir, id_list=ids_to_query)  # location of the data

    for d in hest_data:
        d: HESTData
        break


    hest_reader = SpatialPathReader(d)
    print(hest_reader.genes_df.sort_values('mean_counts'))

    hest_reader.select_genes(['TP53','IGKC','MYL9','UBC','TAGLN','FLNA'])
    patch, expressions = hest_reader[0]
    print(patch.shape, expressions.shape)
    print(expressions)

    print(hest_reader)

    '''
    i=0
    while True:
        i+=1
        r = next(hest_reader)
        p, e = r
        print(i, p.shape, e)
    '''

    #for p,e in itertools.cycle(iter(hest_reader)):
    #    print(p.shape, e.shape)

#test()


def test_dataloader():

    train_loader = DataLoader(dataset, num_workers=1, batch_size=32)
    for batch in iter(train_loader):
        tiles, expressions = batch
        print(tiles.shape, tiles.dtype, expressions.shape, expressions.dtype)
        print(expressions)


if __name__=='__main__':

    test_dataloader()