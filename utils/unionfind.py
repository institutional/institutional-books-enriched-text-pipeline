"""
unionfind.py - Union-Find (Disjoint Set) data structure

See for example https://en.wikipedia.org/wiki/Disjoint-set_data_structure
and "Intro to Algorithms" v3 by CLRS, section 21.3.
"""

from typing import Any, TypeVar

T = TypeVar("T")


class UnionFind:
    """
    Union-Find data structure for grouping items into clusters.

    Uses path compression and union by rank for efficiency.
    See: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """

    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}
        self.rank: dict[Any, int] = {}

    def find(self, x: T) -> T:
        """Find the root of the set containing x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        # Path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, x: T, y: T) -> None:
        """Union the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_clusters(self) -> dict[T, list[T]]:
        """Get the clusters as a dict of root -> list of items."""
        clusters: dict[Any, list[Any]] = {}
        for item in self.parent:
            root = self.find(item)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(item)
        return clusters
