import cvxpy as cp
import torch

class MGDAWrapper:
    def __init__(self, model):
        self.model = model

    def compute_weighted_loss(self, losses, shared_feat_tensor):

        grads = []
        for loss in losses:
            grad = torch.autograd.grad(
                loss,
                shared_feat_tensor,         
                retain_graph=True,
                allow_unused=False,
                create_graph=False       
            )[0]
            grads.append(grad.detach().view(-1))

        G = torch.stack(grads, dim=1)
        G_np = G.cpu().numpy()
        GG = G_np.T @ G_np
        GG = 0.5 * (GG + GG.T)

        T = G.shape[1]
        alpha = cp.Variable(T)
        prob = cp.Problem(cp.Minimize(cp.quad_form(alpha, GG)),
                          [alpha >= 0, cp.sum(alpha) == 1])
        prob.solve()
        return alpha.value 