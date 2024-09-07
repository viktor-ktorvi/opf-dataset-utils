echo "case30 (N-1)"
python -m scripts.edit_distance edit_distance.subgraphs=False show=False
echo "Full retention"
python -m scripts.edit_distance bus_size_range.min=29 bus_size_range.max=31 show=False
echo "Low bus retention"
python -m scripts.edit_distance bus_size_range.min=29 bus_size_range.max=31 bus_retention_ratio=0.1 num_hops_range.min=10 num_hops_range.max=40 show=False