import torch

# pc1: (N1, 3)  N1 >= N2
# pc2: (N2, 3)
pc1 = torch.randn(10000, 3)
pc2 = torch.randn(11000, 3)

# Compute pairwise distances: (N2, N1)
dist = torch.cdist(pc2, pc1)

N2, N1 = dist.shape
used_pc1 = torch.zeros(N1, dtype=torch.bool, device=dist.device)

matched_indices = torch.empty(N2, dtype=torch.long, device=dist.device)

for i in range(N2):
    # Get distances from pc2[i] to all pc1
    d = dist[i].clone()

    # Mask out already used pc1 points by setting distance to +inf
    d[used_pc1] = float('inf')

    # Pick the closest unused pc1 point
    j = torch.argmin(d)
    matched_indices[i] = j
    used_pc1[j] = True

# Now get the matched pc1 points; shape: (N2, 3)
pc1_matched = pc1[matched_indices]
print(pc1_matched.shape)