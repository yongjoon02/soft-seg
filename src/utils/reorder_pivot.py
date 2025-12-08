"""
Reorder pivot table columns by specified network order
"""
import pandas as pd

# Desired network order
network_order = [
    'berdiff',
    'medsegdiff',
    'segdiff',
    'colddiff',
    'vesselnet',
    'aacaunet',
    'dscnet',
    'transunet',
    'cenet',
    'csnet'
]

# Read pivot tables
betti_0_pivot = pd.read_csv('/home/yongjun/diffusion-seg/results/connectivity_comparison/betti_0_error_pivot.csv')
betti_1_pivot = pd.read_csv('/home/yongjun/diffusion-seg/results/connectivity_comparison/betti_1_error_pivot.csv')

# Reorder columns
available_networks = [col for col in network_order if col in betti_0_pivot.columns]
columns_ordered = ['sample_name'] + available_networks

betti_0_reordered = betti_0_pivot[columns_ordered]
betti_1_reordered = betti_1_pivot[columns_ordered]

# Save
betti_0_reordered.to_csv('/home/yongjun/diffusion-seg/results/connectivity_comparison/betti_0_error_pivot.csv', index=False)
betti_1_reordered.to_csv('/home/yongjun/diffusion-seg/results/connectivity_comparison/betti_1_error_pivot.csv', index=False)

print("✓ Reordered betti_0_error_pivot.csv")
print("✓ Reordered betti_1_error_pivot.csv")
print(f"\nColumn order: {available_networks}")
