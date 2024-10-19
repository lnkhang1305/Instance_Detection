import faiss
import numpy as np
from tqdm import tqdm
from typing import Tuple 
import os
import json
import sys


sys.path.append('.')


class FaissIndexStrategy:
    """
    A strategy class for managing different types of FAISS indexes with support of CPU and GPU
    """
    def __init__(
        self,
        index_type: str,
        dimension: int,
        use_gpu: bool = False,
        device: int = 0,
        **kwargs
    ) -> None:
        """
        Initialize the FAISS index strategy.

        Args:
            index_type (str): The type of index to create. Options:
                - 'flat_l2': Exact search using L2 distance
                - 'flat_ip': Exact search using Inner product
                - 'ivfflat': Inverted File Index with Flat quantizer
                - 'ivfpq': Inverted File Index with Product Quantization
                - 'hnsw': Hierarchical Navigable Small World Graph
                - 'ivfsq': IVF with Scalar Quantization
                - 'lsh': Locality-Sensitive Hashing
            dimension (int): The dimension of the feature vectors
            use_gpu (bool): Whether to use GPU resources
            device (int): GPU device ID to use
            **kwargs: Additional parameters specific to the index type:
                - nlist (int): Number of clusters for IVF indexes
                - M (int): Number of connections per layer for HNSW
                - ef_construction (int): Size of dynamic list for HNSW construction
                - ef_search (int): Size of dynamic list for HNSW search
                - nbits (int): Number of bits per sub-vector for PQ/SQ
                - nprobe (int): Number of clusters to visit during search
                - use_precomputed_tables (bool): Use precomputed tables for PQ
                - metric str): Di(stance metric ('l2', 'ip', 'cosine')
                - useFloat16(boolean): 16-bit floating-point
                - storeTransposed(boolean): optimize vector storage and retrieval
                
        """
        self.index_type = index_type
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.device = device
        self.kwargs = kwargs
        self.index = None
        self._needs_training = False
        self._gpu_resources = None
        self._metric = kwargs.get('metric', 'l2')

        # initialize the index
        self._create_index(**kwargs)
        
        self._set_default_parameters()

    def _create_index(self, **kwargs) -> None:
        """
        Initialize the Faiss index with the specified configuration
        """
        print(f"Creating index: index_type={self.index_type}, use_gpu={self.use_gpu}, kwargs={kwargs}")
        if self.use_gpu:
            self._gpu_resources = faiss.StandardGpuResources()
            print(f"GPU resources initialized on device {self.device}")
        
        metric = faiss.METRIC_L2 if self._metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        self.metric = metric
        print(f"Metric set to: {self._metric}")

        if self.index_type == "flat_l2":
            if self.use_gpu:
                config = faiss.GpuIndexFlatL2Config()
                config.device = self.device
                config.useFloat16 = kwargs.get('use_float16', False)
                config.storeTranspose = kwargs.get('store_transposed', False)
                
                # Create CPU index first
                # cpu_index = faiss.IndexFlatL2(self.dimension)
                # Convert to GPU
                self.index = faiss.GpuIndexFlatL2(self._gpu_resources, self.dimension, config)
                print(f"GPU Flat L2 index created on device {self.device}")
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                print("CPU Flat L2 index created")

        elif self.index_type == "flat_ip":
            if self.use_gpu:
                config = faiss.GpuIndexFlatConfig()
                config.device = self.device
                config.useFloat16 = kwargs.get('use_float16', False)
                config.storeTransposed = kwargs.get('store_transposed', False)
                
                # Create CPU index first
                # cpu_index = faiss.IndexFlatIP(self.dimension)
                # Convert to GPU
                self.index = faiss.GpuIndexFlatIP(self._gpu_resources, self.dimension, config)
                print(f"GPU Flat IP index created on device {self.device}")
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                print("CPU Flat IP index created")

        elif self.index_type == "ivfflat":
            nlist = kwargs.get('nlist', 300)
            if self.use_gpu:
                config = faiss.GpuIndexIVFFlatConfig()
                config.device = self.device
                config.indicesOptions = kwargs.get('indices_options', faiss.INDICES_32_BIT)
                config.flatConfig.useFloat16 = kwargs.get('use_float16', False)
                config.flatConfig.storeTransposed = kwargs.get('store_transposed', False)

                quantizer = faiss.IndexFlatL2(self.dimension)
                cpu_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
                
                self.index = faiss.GpuIndexIVFFlat(self._gpu_resources, cpu_index, config)
                print(f"GPU IVF Flat index created with nlist={nlist}")
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
                print(f"CPU IVF Flat index created with nlist={nlist}")
            self._needs_training = True

        elif self.index_type == "ivfpq":
            nlist = kwargs.get('nlist', 300)
            m = kwargs.get('m', 8)  
            nbits = kwargs.get('nbits', 8) 
            if self.use_gpu:
                config = faiss.GpuIndexIVFPQConfig()
                config.device = self.device
                config.indicesOptions = kwargs.get('indices_options', faiss.INDICES_32_BIT)
                config.useFloat16LookupTables = kwargs.get('use_float16_lookup', False)
                config.usePrecomputedTables = kwargs.get('use_precomputed_tables', True)
                config.flatConfig.useFloat16 = kwargs.get('use_float16', False)
                config.flatConfig.storeTransposed = kwargs.get('store_transposed', False)

                quantizer = faiss.IndexFlatL2(self.dimension)
                cpu_index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, metric)
                
                self.index = faiss.GpuIndexIVFPQ(self._gpu_resources, cpu_index, config)
                print(f"GPU IVF PQ index created with nlist={nlist}, m={m}, nbits={nbits}")
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, metric)
                print(f"CPU IVF PQ index created with nlist={nlist}, m={m}, nbits={nbits}")
            self._needs_training = True

        elif self.index_type == 'hnsw':
            M = kwargs.get('M', 32)
            ef_construction = kwargs.get('ef_construction', 128)
            self.index = faiss.IndexHNSWFlat(self.dimension, M, metric)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = kwargs.get('ef_search', 64)
            print(f"HNSW index created with M={M}, ef_construction={ef_construction}, ef_search={kwargs.get('ef_search', 64)}")
            if self.use_gpu:
                print("Warning: HNSW index does not support GPU. Using CPU instead.")

        elif self.index_type == 'ivfsq':
            nlist = kwargs.get('nlist', 300)
            qtype = kwargs.get('qtype', faiss.ScalarQuantizer.QT_8bit)
            
            if self.use_gpu:
                config = faiss.GpuIndexIVFScalarQuantizerConfig()
                config.device = self.device
                config.indicesOptions = kwargs.get('indices_options', faiss.INDICES_32_BIT)
                config.flatConfig.useFloat16 = kwargs.get('use_float16', False)
                config.flatConfig.storeTransposed = kwargs.get('store_transposed', False)
                
                quantizer = faiss.IndexFlatL2(self.dimension)
                cpu_index = faiss.IndexIVFScalarQuantizer(quantizer, self.dimension, 
                                                        nlist, qtype, metric)
                
                self.index = faiss.GpuIndexIVFScalarQuantizer(self._gpu_resources, cpu_index, config)
                print(f"GPU IVF Scalar Quantizer index created with nlist={nlist}, qtype={qtype}")
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFScalarQuantizer(quantizer, self.dimension, 
                                                        nlist, qtype, metric)
                print(f"CPU IVF Scalar Quantizer index created with nlist={nlist}, qtype={qtype}")
            self._needs_training = True

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    

    def _set_default_parameters(self) -> None:
        """Set default search parameters based on index type."""
        if hasattr(self.index, 'nprobe'):
            default_nprobe = min(64, max(1, self.index.nlist // 8))
            self.index.nprobe = self.kwargs.get('nprobe', default_nprobe)
        
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.kwargs.get('ef_search', 64)

    def add(self, vectors: np.ndarray, batch_size: int = 50_000) -> None:
        """Add vectors to the index with batch processing support : D

        Args:
            vectors(or matrix) (np.ndarray): Vectors to add. Shape(N, dimension), where N is the number of vectors
            batch_size (int, optional): Size of batches for processing. Defaults to 50_000
        """
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        if self._metric == 'cosine':
            faiss.normalize_L2(vectors)
        
        if self._needs_training and not self.index.is_trained:
            print("Training Index")
            self.index.train(vectors)
            self._needs_training=False
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Adding vectors"):
            batch = vectors[i:i + batch_size]
            self.index.add(batch)
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
          Search the index for the k nearest neighbors of the query vectors.

          Args:
              query_vectors (np.ndarray): An array of query vectors, shape (N, dimension), where N = 1
              k (int): The number of nearest neighbors to return.

          Returns:
              D (np.ndarray): Distances to the nearest neighbors, shape (N, k).
              I (np.ndarray): Indices of the nearest neighbors, shape (N, k).
        """

        query_vector = np.asarray(query_vector).astype('float32')
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector)
        return D, I

    def save(self, filename:str)->None:
        """Save the index to a file

        Args:
            filename (str): path to save the index
        """
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, filename)
        else:
            faiss.write_index(self.index, filename)
        

        # Save the metadata of the index
        metadata = {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'metric': self.metric,
            'kwargs': self.kwargs
        }

        metadata_filename = f"{os.path.splitext(filename)[0]}_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f)
    
    def load(self, filename: str) -> None:
        """
        Load the index from a file.

        Args:
            filename (str): Path to load the index from
        """
        metadata_file = f"{os.path.splitext(filename)[0]}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.index_type = metadata['index_type']
            self.dimension = metadata['dimension']
            self._metric = metadata['metric']
            self.kwargs.update(metadata['kwargs'])
        
        self.index = faiss.read_index(filename)
        if self.use_gpu:
            self._gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_gpu_to_cpu(self._gpu_resources, self.device, self.index)
        
        self._set_default_parameters()
        