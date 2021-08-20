import anndata as ad
import numpy as np
import scanpy as sc


if __name__ == "__main__":
    adata = ad.read_h5ad("/home/svcapp/tbrain_x/azimuth/pbmc_multimodal_raw.h5ad")
    # raw data does not have label, so we have to load normalized data to borrow label (obs) part
    adata.obs = ad.read_h5ad("/home/svcapp/tbrain_x/azimuth/pbmc_multimodal.h5ad").obs

    cell_type_names, y = np.unique(adata.obs["celltype.l2"].values, return_inverse=True)

    sc.pp.filter_genes(adata, min_counts=40000)
    sc.pp.filter_cells(adata, min_counts=100)
    print(f"Filtered data size: {adata.X.shape}")
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    adata.write_h5ad("pbmc_normalized_gene_min_30000.h5ad")