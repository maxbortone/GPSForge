import numpy as np
from mpi4py import MPI
from dataclasses import dataclass


# MPI variables
@dataclass
class MPIVariables:
    comm : MPI.Comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
    rank : int = comm.Get_rank()
    n_nodes : int = comm.Get_size()

MPIVars = MPIVariables()

def compute_chunk_size(multiplier, n_samples, size):
    if multiplier > 0:
        chunk_size = int(2**(np.ceil(np.log2(n_samples*size*multiplier))))
    else:
        raise ValueError("Chunk size multiplier needs to be > 0.0")
    return chunk_size
