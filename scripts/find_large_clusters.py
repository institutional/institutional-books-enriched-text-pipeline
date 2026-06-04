"""Find and analyze the largest duplicate clusters.

Institutional Books - Enriched Text - 2026
"""

import json
from pathlib import Path


def find_largest_clusters(clusters_file, min_size=100):
    """Find all clusters >= min_size by size, return sorted list."""
    with open(clusters_file, "r") as f:
        data = json.load(f)

    clusters_dict = data.get("clusters", data)

    clusters_data = []
    for cluster_key, para_ids in clusters_dict.items():
        if isinstance(para_ids, list):
            size = len(para_ids)
            if size >= min_size:
                # Extract unique book IDs from paragraph IDs
                # Format is barcode_src.paragraph_index (no colon)
                books_involved = set()
                for p in para_ids:
                    parts = str(p).split(".")
                    if len(parts) >= 1:
                        books_involved.add(parts[0])

                clusters_data.append(
                    {
                        "cluster_id": cluster_key,
                        "size": size,
                        "books_involved": list(books_involved),
                    }
                )

    # Sort by size descending
    clusters_data.sort(key=lambda x: x["size"], reverse=True)
    return clusters_data


def analyze_top_clusters(clusters, min_size):
    """Analyze the top N large clusters."""
    print("=" * 70)
    print(f"LARGEST DUPLICATE CLUSTERS (size >= {min_size})")
    print("=" * 70)

    for i, cluster in enumerate(clusters[:50], 1):
        unique_books = len(cluster["books_involved"])
        book_list_str = ", ".join(cluster["books_involved"][:25])

        print(f"\n{i}. Cluster {cluster['cluster_id']}")
        print(f"   Size: {cluster['size']} duplicate paragraphs")
        if unique_books > 10:
            print(f"   Unique books: {book_list_str}... ({unique_books} total)")
        else:
            print(f"   Books involved: {', '.join(cluster['books_involved'])}")


def main():
    clusters_file = "DATA/Cluster/clusters.json"

    print("Finding largest clusters...")
    clusters = find_largest_clusters(clusters_file, min_size=50)

    print(f"\nFound {len(clusters)} clusters with size >= 50\n")
    analyze_top_clusters(clusters, 50)

    # Also write full list to file for later analysis
    output_path = "scripts/large_clusters_top100.jsonl"
    with open(output_path, "w") as f:
        for cluster in clusters[:100]:
            f.write(json.dumps(cluster) + "\n")

    print(f"\nWrote top 100 clusters to {output_path}")


if __name__ == "__main__":
    main()
