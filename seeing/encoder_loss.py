from torch.nn.functional import cosine_similarity

def cor_distance(x, y, eps=1e-12):
    # Analogous to L1 distance, but in terms of Pearson's correlation
    return (1.0 - cosine_similarity(x, y, eps=eps)).sqrt().mean()

def cor_square_error(x, y, eps=1e-12):
    # Analogous to MSE, but in terms of Pearson's correlation
    return (1.0 - cosine_similarity(x, y, eps=eps)).mean()
