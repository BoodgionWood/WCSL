import numpy as np
import scipy.optimize
import torch
from torch.autograd import Function
from typing import Tuple
from typing_extensions import Final

# Get the optimal alpha given beta
@torch.jit.script
def optimal_alpha(beta: torch.Tensor, loga: torch.Tensor, M: torch.Tensor, lam: float):
    D = lam * (beta.reshape(1, -1) - M)
    alpha = (loga - torch.logsumexp(D, dim=1)) / lam
    return alpha

# Get the optimal beta given alpha
@torch.jit.script
def optimal_beta(alpha: torch.Tensor, logb: torch.Tensor, M: torch.Tensor, lam: float):
    D = lam * (alpha.reshape(-1, 1) - M)
    beta = (logb - torch.logsumexp(D, dim=0)) / lam
    return beta

# Get the current transport plan from beta
# beta[m], M[n x m]
@torch.jit.script
def sinkhorn_plan(beta: torch.Tensor, loga: torch.Tensor, M: torch.Tensor, lam: float, log: bool = False):
    # Get the optimal alpha
    alpha = optimal_alpha(beta, loga, M, lam)
    # Compute T
    logT = lam * (alpha.reshape(-1, 1) + beta.reshape(1, -1) - M)
    if log:
        return logT, alpha, beta
    T = torch.exp(logT)
    # print("T is:" ,T)
    return T, alpha, beta

# Compute the objective function and gradient
# wa [n], wb [m]
@torch.jit.script
def sinkhorn_dual_obj_grad(beta: torch.Tensor, M: torch.Tensor,
                           a: torch.Tensor, b: torch.Tensor,
                           loga: torch.Tensor, lam: float):
    # Get the transport plan
    T, alpha, _ = sinkhorn_plan(beta, loga, M, lam, log=False)
    Tcolsum = T.sum(dim=0)
    # obj = -alpha.dot(a) - beta.dot(b) + Tcolsum.sum() / lam
    obj = -alpha.dot(a) - beta.dot(b) + 1.0 / lam
    grad = Tcolsum - b
    return float(obj), grad

def sinkhorn_dual_obj_grad_np(beta_np, M, a, b, loga, lam):
    beta = torch.tensor(beta_np, dtype=M.dtype, device=M.device)
    obj, grad = sinkhorn_dual_obj_grad(beta, M, a, b, loga, lam)
    # scipy.optimize.minimize requires "double" type
    grad = grad.detach().cpu().numpy().astype(dtype=np.float64)
    return obj, grad

def sinkhorn_sol_scipy(beta0_np, M, a, b, loga, lam, max_iter=1000, gtol=1e-6, verbose=False):
    # opts = dict(disp=verbose, maxiter=max_iter, gtol=gtol)
    # res = scipy.optimize.minimize(
    #     sinkhorn_dual_obj_grad_np, beta0_np, args=(M, a, b, loga, lam),
    #     method="BFGS", jac=True, options=opts)
    
    opts = dict(iprint=0 if verbose else -1, maxiter=max_iter, ftol=0.0, gtol=gtol)
    res = scipy.optimize.minimize(
        sinkhorn_dual_obj_grad_np, beta0_np, args=(M, a, b, loga, lam),
        method="L-BFGS-B", jac=True, options=opts)
    return res["x"]

# Check whether "weights" is a legal probability distribution
def check_weights(weights):
    return torch.all(weights >= 0.0).item() and abs(weights.sum() - 1.0) < 1e-6

# Computing Sinkhorn loss using BFGS in the forward pass, and analytic
# differentiation in the backward pass
def SinkhornLoss(a=None, b=None, reg=0.1, max_iter=1000, gtol=1e-6,
                 refine=10, verbose=False):
    # Check the weights if not None
    if a is not None:
        assert a.ndim == 1
        if not check_weights(a):
            raise RuntimeError("'a' must be a probability distribution")
    if b is not None:
        assert b.ndim == 1
        if not check_weights(b):
            raise RuntimeError("'b' must be a probability distribution")

    class SinkhornFn(Function):
        @staticmethod
        def forward(ctx, M):
            # Check dimensions
            if M.ndim != 2:
                raise RuntimeError("'M' must be a matrix")
            n = M.shape[0]
            m = M.shape[1]

            # Original data type
            ctx.ori_dt = M.dtype
            # Common arguments for tensor creation
            targs = dict(dtype=M.dtype, device=M.device)
            # Generate default a or b if they are set to None
            wa = torch.ones(n, **targs) / n if a is None else a.to(**targs)
            wb = torch.ones(m, **targs) / m if b is None else b.to(**targs)
            loga = torch.log(wa)
            logb = torch.log(wb)
            
            # Initial value
            beta0_np = np.zeros(m)
            lam = 1.0 / reg
            ctx.lam = lam
            
            # Actual optimization
            with torch.no_grad():
                beta_np = sinkhorn_sol_scipy(beta0_np, M, wa, wb, loga, lam,
                                             max_iter, gtol, verbose)
            # Refine the result using double type
            with torch.no_grad():
                targs = dict(dtype=torch.float64, device=M.device)
                beta = torch.tensor(beta_np, **targs)
                M = M.to(**targs)
                loga = loga.to(**targs)
                logb = logb.to(**targs)
                for i in range(refine):
                    alpha = optimal_alpha(beta, loga, M, lam)
                    beta = optimal_beta(alpha, logb, M, lam)
            
            # Get the log transport plan
            with torch.no_grad():
                logT, alpha, beta = sinkhorn_plan(beta, loga, M, lam, log=True)
            # Matrices and vectors used in the backward pass
            T = torch.exp(logT)
            muc = (M * T).sum(dim=0)
            # Compute loss
            loss = muc.sum()

            # Save for backward stage
            ctx.save_for_backward(M, logT, T, muc)
            ctx.set_materialize_grads(False)

            return loss.to(dtype=ctx.ori_dt)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grad_output):
            # The length of grad_output is determined by the number
            # of outputs in forward(). For example, if only loss
            # is returned, then grad_output has only one element.
            # If both loss and gamma are returned, then grad_output
            # contains two elements
            #
            # However, we never compute gradient for gamma, so we
            # simply ignore the second element of grad_output,
            # even if it exists
            #
            # The number of outputs of backward() is determined by
            # the number of inputs of forward()

            # Early exit if M does not require gradient
            if not ctx.needs_input_grad[0]:
                return None
            # Early exit if gradient for loss is None
            if grad_output[0] is None:
                return None

            M, logT, T, muc = ctx.saved_tensors
            
            wb = T.sum(dim=0)
            Ta = torch.softmax(logT, dim=1) # = diag(1/a) * T
            mura = (M * Ta).sum(dim=1)      # = mur / wa
            Tta = Ta[:, :-1]
            Tt = T[:, :-1]
            D = torch.diag(wb[:-1] + 1e-10) - torch.mm(Tta.t(), Tt)
            sv_rhs = muc[:-1] - torch.mv(Tt.t(), mura)
            sv = torch.linalg.solve(D, sv_rhs)
            su = mura - torch.mv(Tta, sv)
            suv = su.reshape(-1, 1) + sv
            suv = torch.hstack((suv, su.reshape(-1, 1)))
            res = T + ctx.lam * (suv - M) * T
            return grad_output[0] * res.to(dtype=ctx.ori_dt)

    return SinkhornFn.apply
