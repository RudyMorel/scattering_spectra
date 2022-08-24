from typing import *
import itertools
import numpy as np


def transpose(li: Iterable[Iterable]) -> List[List]:
    """Transpose [[a11, ..., a1n], ... [am1, ..., amn]] and return  [[a11, ..., am1], ... [a1n, ..., amn]]."""
    return list(map(list, zip(*li)))


def compose(*functions: Callable) -> Callable:
    """Given functions f1, ..., fn, return f1 o ... o fn."""
    def inner(arg):
        for f in reversed(functions):
            arg = f(arg)
        return arg
    return inner


def dfs_edges(g, source=None, depth_limit=None):
    if source is None:
        # edges for all components
        nodes = g
    else:
        # edges for components with source
        nodes = source
    visited = set()
    if depth_limit is None:
        depth_limit = len(g)
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, depth_limit, iter(g[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(g[child])))
            except StopIteration:
                stack.pop()


def concat_list(a: List[List]) -> List:
    return list(itertools.chain.from_iterable(a))


def reverse_permutation(s: Sequence) -> List:
    """Giben permutation s: i -> j return the inverse permutation j -> i."""
    sinv = [0] * len(s)
    for i in range(len(s)):
        sinv[s[i]] = i
    return sinv


def get_permutation(a: Sequence, b: Sequence):
    """Return the permutation s such that a[s[i]] = b[i]"""
    assert set(a) == set(b)

    d = {val: key for key, val in enumerate(a)}
    s = [d[val] for val in b]

    return s


def split_equal_sum(li: List[int], r: int) -> Tuple[List[List[int]], List[List[int]]]:
    sublists = OrderedDict({s: [] for s in range(r)})
    subindices = OrderedDict({s: [] for s in range(r)})
    sums = np.zeros((r,))

    li = np.array(li)
    order = np.argsort(li)[::-1]

    for i, x in enumerate(li[order]):
        i0 = np.argmin(sums)
        sublists[i0].append(x)
        subindices[i0].append(order[i])
        sums[i0] += x

    return list(subindices.values()), list(sublists.values())
