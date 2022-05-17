import os
import logging as python_logging
from absl import logging
from mpi4py import MPI
from dataclasses import dataclass


# MPI variables
@dataclass
class MPIVariables:
    comm : MPI.Comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
    rank : int = comm.Get_rank()
    n_nodes : int = comm.Get_size()

MPIVars = MPIVariables()


def add_file_logger(workdir, *, basename='train', level=python_logging.INFO):
  """Adds a file logger to Python logging handlers"""
  filename = f'{workdir}/{basename}.log'
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  mode = 'a' if os.path.isfile(filename) else 'w'
  fh = python_logging.FileHandler(filename, mode)
  fh.setLevel(level)
  fh.setFormatter(logging.PythonFormatter())
  python_logging.getLogger('').addHandler(fh)