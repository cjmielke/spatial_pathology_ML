import collections

import numpy
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import tables

CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])


def get_matrix_from_h5(filename):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        tag_keys = getattr(feature_group, '_all_tag_keys').read()
        for key in tag_keys:
            if type(key)==numpy.bytes_:
                key = key.decode('ascii')
            feature_ref[key] = getattr(feature_group, key).read()

        feature_ref = pd.DataFrame(feature_ref)

        return CountMatrix(feature_ref, barcodes, matrix)


#filtered_h5 = "/opt/sample345/outs/filtered_feature_bc_matrix.h5"
#filtered_matrix_h5 = get_matrix_from_h5(filtered_hs)


# test
if __name__ == '__main__':

    D = get_matrix_from_h5('/home/cosmo/10x_spatial_AI/data/colon_cancer/binned_outputs/square_016um/filtered_feature_bc_matrix.h5')

    print(D.feature_ref)
    print(D.barcodes[:5])
    print(D.matrix.shape)       # (18085, 137051)
    M = D.matrix
    counts_per_gene = M.sum(axis=1)
    #counts_per_gene = np.sum(M, axis=0)
    print(counts_per_gene[:10])
    print(counts_per_gene.shape)

    print(np.argmax(counts_per_gene))
    print(counts_per_gene[2326])

    print(M.shape)  # should return the "expression vector" of 18k genes for spot 42
    vec = M[:,402]
    print(vec.shape)          # should return the "expression vector" of 18k genes for spot 42
    print(vec.max(), vec.mean())