# Modified version of: https://github.com/swansonk14/p_tqdm
# Copyright (c) 2022 Kyle Swanson
# MIT License

from collections.abc import Sized
from typing import Any, Callable, Generator, Iterable, List

from pathos.multiprocessing import _ProcessPool as Pool
from tqdm.auto import tqdm


def _parallel(function: Callable, *iterables: Iterable, **kwargs: Any) -> Generator:
    # Determine length of tqdm (equal to length of the shortest iterable or total kwarg)
    total = kwargs.pop("total", None)
    lengths = [len(iterable) for iterable in iterables if isinstance(iterable, Sized)]
    length = total or (min(lengths) if lengths else None)

    # Create parallel generator
    num_cpus = kwargs.pop("num_cpus", 1)
    maxtasksperchild = kwargs.pop("maxtasksperchild", 1)
    chunksize = kwargs.pop("chunksize", 1)
    with Pool(num_cpus, maxtasksperchild=maxtasksperchild) as pool:
        for item in tqdm(
            pool.imap_unordered(function, *iterables, chunksize=chunksize),
            total=length,
            **kwargs
        ):
            yield item


def p_umap(function: Callable, *iterables: Iterable, **kwargs: Any) -> List[Any]:
    return list(_parallel(function, *iterables, **kwargs))
