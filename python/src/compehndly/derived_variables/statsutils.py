import numpy as np

from scipy.stats import lognorm
from scipy.optimize import minimize


def fit_censored_lognorm(values_np, censored_np):
    vals = values_np
    cens = censored_np

    # Require at least some uncensored observations
    if (~cens).sum() == 0:
        raise RuntimeError(
            "Cannot fit lognormal: all observations are censored."
        )

    def nll(params):
        sigma, mu = params
        if sigma <= 0:
            return np.inf

        dist = lognorm(s=sigma, scale=np.exp(mu))
        ll_unc = dist.logpdf(vals[~cens]).sum()
        ll_cens = np.log(dist.cdf(vals[cens])).sum()
        ll = ll_unc + ll_cens

        # Regularization to avoid pathological sigma
        penalty = 0
        if sigma < 0.05:
            penalty += 1e3 * (0.05 - sigma) ** 2
        if sigma > 5.0:
            penalty += 1e3 * (sigma - 5.0) ** 2

        return -(ll - penalty)

    unc = vals[~cens]
    mu0 = np.log(np.median(unc))
    sigma0 = np.std(np.log(unc))
    sigma0 = sigma0 if sigma0 > 0.1 else 0.5  # stability

    res = minimize(
        nll,
        x0=[sigma0, mu0],
        method="L-BFGS-B",
        bounds=[(1e-6, None), (None, None)],
    )

    if not res.success:
        raise RuntimeError(f"Censored MLE did not converge: {res.message}")

    sigma_hat, mu_hat = res.x
    return lognorm(s=sigma_hat, scale=np.exp(mu_hat))
