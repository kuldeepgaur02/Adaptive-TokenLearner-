# Run B: the same head and the same 20-epoch schedule, with 3 decoder layers
# instead of 1.
#
# Why
# ---
# The 20-epoch run finds the cars but does not place the boxes tightly:
#
#     AP40 moderate @ IoU 0.50   34.89
#     AP40 moderate @ IoU 0.70   12.69
#
# Loosening the overlap threshold nearly triples the score, so the bottleneck
# is localisation precision, not detection. The head-level diagnostic says the
# same thing: predicted centres sit ~2.1 m from their matched car, and the
# rotation term is still 53% of loss_bbox.
#
# A decoder layer is where a query gets to look at the map again and correct
# itself. With one layer every box is produced from a single pass; with three,
# a query can refine what it found. This is the standard fix for exactly this
# symptom in DETR-style detectors, and it is the cheapest one to test.
#
# Memory: cross-attention holds B x heads x S x M weights per layer, which is
# ~1.0 GB at batch 6, 8 heads, S=100, M=53,568. The 1-layer run used 8.3 GB of
# 16 GB, so three layers should land around 12 GB. If it does hit OOM, set
# memory_pool_size=2 below (halves M by average-pooling the memory) before
# reducing the batch size, so the comparison against Run A stays clean.
_base_ = ['./voxelnet_tokenlearner_s100_Kuldeep.py']

work_dir = './work_dirs/voxelnet_tokenlearner_s100_L3'

model = dict(
    bbox_head=dict(
        num_decoder_layers=3,
    ),
)
