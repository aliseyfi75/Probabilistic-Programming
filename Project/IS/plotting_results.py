from newplot import draw_hists
import torch

thetas = torch.load('thetas.pt')
weights = torch.load('weights.pt')
draw_hists("Importance_Sampling", thetas, 1, weights=weights)