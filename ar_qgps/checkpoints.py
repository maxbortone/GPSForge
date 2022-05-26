# Adapted from https://github.com/google/flax/blob/main/flax/training/checkpoints.py
# to temporarily remove dependence on tensorflow.io.gfile.
# Should be taken care of in flax soon (see: https://github.com/google/flax/issues/1924 and https://github.com/google/flax/pull/2073)

from concurrent.futures import thread
import os
import re
import glob
from typing import Any, Iterable, List, Optional, Union

from absl import logging
from flax import errors
from flax import serialization


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(
    r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# Module name folowed by number.
MODULE_NUM_RE = re.compile(r'(.*)_\d+$')
# Alternative schemes handled by `gfile`, e.g. on Google Cloud Storage (GCS).
SCHEME_RE = re.compile('^(?P<scheme>[a-z][a-z0-9.+-]+://)?(?P<path>.*)', re.I)

PyTree = Any


def _checkpoint_path(ckpt_dir: str,
                     step: Union[int, str],
                     prefix: str = 'checkpoint_') -> str:
    return os.path.join(ckpt_dir, f'{prefix}{step}')


def _checkpoint_path_step(path: str) -> Optional[float]:
    """Returns the step number of a checkpoint path."""
    for s in SIGNED_FLOAT_RE.split(path)[::-1]:
        if SIGNED_FLOAT_RE.match(s):
            return float(s)
    return None


def natural_sort(file_list: Iterable[str], signed: bool = True) -> List[str]:
    """Natural sort for filenames with numerical substrings.
    Args:
        file_list: list of paths to sort containing numerical substrings.
        signed: bool: if leading '-' (or '+') signs should be included in
        numerical substrings as a sign or treated as a separator.
    Returns:
        List of filenames sorted 'naturally', not lexicographically: any
        integer substrings are used to subsort numerically. e.g.
        file_1, file_10, file_2  -->  file_1, file_2, file_10
        file_0.1, file_-0.2, file_2.0  -->  file_-0.2, file_0.1, file_2.0
    """
    float_re = SIGNED_FLOAT_RE if signed else UNSIGNED_FLOAT_RE
    def maybe_num(s):
        if float_re.match(s):
            return float(s)
        else:
            return s
    def split_keys(s):
        return [maybe_num(c) for c in float_re.split(s)]
    return sorted(file_list, key=split_keys)


def safe_normpath(path: str) -> str:
    """Normalizes path safely to get around `gfile.glob()` limitations."""
    d = SCHEME_RE.match(path).groupdict()
    return (d['scheme'] or '') + os.path.normpath(d['path'])


def save_checkpoint(ckpt_dir: Union[str, os.PathLike],
                    target: PyTree,
                    step: int,
                    prefix: str = 'checkpoint_',
                    keep: int = 1,
                    overwrite: bool = False,
                    keep_every_n_steps: Optional[int] = None) -> str:
    """Save a checkpoint of the model.
    Attempts to be pre-emption safe by writing to temporary before
    a final rename and cleanup of past files.
    Args:
        ckpt_dir: str or pathlib-like path to store checkpoint files in.
        target: serializable flax object, usually a flax optimizer.
        step: int or float: training step number or other metric number.
        prefix: str: checkpoint file name prefix.
        keep: number of past checkpoint files to keep.
        overwrite: overwrite existing checkpoint files if a checkpoint
        at the current or a later step already exits (default: False).
        keep_every_n_steps: if defined, keep every checkpoints every n steps (in
        addition to keeping the last 'keep' checkpoints).
    Returns:
        Filename of saved checkpoint.
    """
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    # Write temporary checkpoint file.
    logging.info('Saving checkpoint at step: %s', step)
    # normalize path because gfile.glob() can modify path './', '//' ...
    ckpt_dir = safe_normpath(ckpt_dir)
    ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
    ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    base_path = os.path.join(ckpt_dir, prefix)
    checkpoint_files = glob.glob(base_path + '*')

    if ckpt_path in checkpoint_files:
        if not overwrite:
            raise errors.InvalidCheckpointError(ckpt_path, step)
    else:
        checkpoint_files.append(ckpt_path)

    checkpoint_files = natural_sort(checkpoint_files)
    # Handle the case if the job was preempted after the temporary checkpoint was
    # written, but before it was renamed to the final checkpoint name
    if checkpoint_files[-1] == ckpt_tmp_path:
        checkpoint_files.pop(-1)
    if ckpt_path != checkpoint_files[-1]:
        if not overwrite:
            raise errors.InvalidCheckpointError(ckpt_path, step)

    with open(ckpt_tmp_path, 'wb') as fp:
        fp.write(serialization.to_bytes(target))

    # Rename once serialization and writing finished.
    os.rename(ckpt_tmp_path, ckpt_path)
    logging.info('Saved checkpoint at %s', ckpt_path)

    # Remove newer checkpoints
    if overwrite:
        ind = checkpoint_files.index(ckpt_path) + 1
        newer_ckpts = checkpoint_files[ind:]
        checkpoint_files = checkpoint_files[:ind]
        for path in newer_ckpts:
            logging.info('Removing checkpoint at %s', path)
            os.remove(path)

    # Remove old checkpoint files.
    last_kept = -float('inf')
    if len(checkpoint_files) > keep:
        old_ckpts = checkpoint_files[:-keep]
        # Note: old_ckpts is sorted from oldest to newest.
        for path in old_ckpts:
            if keep_every_n_steps:
                step_number = _checkpoint_path_step(path)
                if step_number and (step_number - last_kept) >= keep_every_n_steps:
                    logging.debug('Not deleting %s, because last_kept=%f and keeping '
                                    'every %d steps.',
                                    path, last_kept, keep_every_n_steps)
                    last_kept = step_number
                    continue
            logging.info('Removing checkpoint at %s', path)
            os.remove(path)

    return ckpt_path


def latest_checkpoint(ckpt_dir: Union[str, os.PathLike],
                      prefix: str = 'checkpoint_') -> Optional[str]:
    """Retrieve the path of the latest checkpoint in a directory.
    Args:
        ckpt_dir: str: directory of checkpoints to restore from.
        prefix: str: name prefix of checkpoint files.
    Returns:
        The latest checkpoint path or None if no checkpoints were found.
    """
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    glob_path = os.path.join(ckpt_dir, f'{prefix}*')
    checkpoint_files = natural_sort(glob.glob(glob_path))
    ckpt_tmp_path = _checkpoint_path(ckpt_dir, 'tmp', prefix)
    checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
    if checkpoint_files:
        return checkpoint_files[-1]
    else:
        return None


def restore_checkpoint(ckpt_dir: Union[str, os.PathLike],
                       target: Optional[PyTree],
                       step: Optional[int] = None,
                       prefix: str = 'checkpoint_',
                       parallel: bool = True) -> PyTree:
    """Restore last/best checkpoint from checkpoints in path.
    Sorts the checkpoint files naturally, returning the highest-valued
    file, e.g.:
    *  ``ckpt_1, ckpt_2, ckpt_3 --> ckpt_3``
    *  ``ckpt_0.01, ckpt_0.1, ckpt_0.001 --> ckpt_0.1``
    *  ``ckpt_-1.0, ckpt_1.0, ckpt_1e5 --> ckpt_1e5``
    Args:
        ckpt_dir: str: checkpoint file or directory of checkpoints to restore from.
        target: matching object to rebuild via deserialized state-dict. If None,
        the deserialized state-dict is returned as-is.
        step: int: step number to load or None to load latest. If specified,
        ckpt_dir must be a directory.
        prefix: str: name prefix of checkpoint files.
        parallel: bool: whether to load seekable checkpoints in parallel, for speed.
    Returns:
        Restored `target` updated from checkpoint file, or if no step specified and
        no checkpoint files present, returns the passed-in `target` unchanged.
        If a file path is specified and is not found, the passed-in `target` will be
        returned. This is to match the behavior of the case where a directory path
        is specified but the directory has not yet been created.
    """
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    ckpt_dir = safe_normpath(ckpt_dir)
    if step is not None:
        ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
        if not os.path.exists(ckpt_path):
            raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
    else:
        if not os.path.exists(ckpt_dir):
            logging.info('Found no checkpoint directory at %s', ckpt_dir)
            return target
        if not os.path.isdir(ckpt_dir):
            ckpt_path = ckpt_dir
        else:
            ckpt_path = latest_checkpoint(ckpt_dir, prefix)
        if not ckpt_path:
            logging.info('Found no checkpoint files in %s with prefix %s',
                        ckpt_dir, prefix)
            return target

    logging.info('Restoring checkpoint from %s', ckpt_path)
    with open(ckpt_path, 'rb') as fp:
        if parallel and fp.seekable():
            buf_size = 128 << 20  # 128M buffer.
            fp_size = os.path.getsize(ckpt_path)
            num_bufs = fp_size / buf_size
            logging.debug('num_bufs: %d', num_bufs)
            checkpoint_contents = bytearray(fp_size)

            def read_chunk(i):
                # NOTE: We have to re-open the file to read each chunk, otherwise the
                # parallelism has no effect. But we could reuse the file pointers
                # within each thread.
                with open(ckpt_path, 'rb') as f:
                    f.seek(i * buf_size)
                    buf = f.read(buf_size)
                    if buf:
                        checkpoint_contents[i * buf_size:i * buf_size + len(buf)] = buf
                    return len(buf) / buf_size

            pool_size = 32
            pool = thread.ThreadPoolExecutor(pool_size)
            results = pool.map(read_chunk, range(int(num_bufs) + 1))
            pool.shutdown(wait=False)
            logging.debug(f'results: {list(results)}')
        else:
            checkpoint_contents = fp.read()

        if target is None:
            return serialization.msgpack_restore(checkpoint_contents)
        else:
            return serialization.from_bytes(target, checkpoint_contents)
