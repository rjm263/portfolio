import yfinance as yf
import numpy as np
import pandas as pd
import portfolio_analysis as pa
from scipy.stats import multivariate_normal, multivariate_t, quantile
from scipy.optimize import minimize

def VaR(self, horizon: int, alpha: float, period: str, dist: str='t', n_scen: int=100_000) -> float:
    """
    Returns the value at risk (VaR) of the portfolio within
    a given time horizon, with confidence level 1-alpha. VaR
    is computed via Monte Carlo, using dist as log-return
    distribution.

    Arguments:
        horizon: int
            time horizon, in days
        alpha: float
            determines confidence level 1-alpha, e.g. alpha=.05
        dist: str
            choose between 'normal' and 't' for (multivariate)
            normal and student-t distribution
        period: str
            choose time period of historical data for MLE
        n_scen: int
            number of scenarios for MC simulation
    """
    if dist != 'normal' and dist != 't': raise ValueError("dist has to be 'normal' or 't'!")

    ##########################################################################
    # Fetch log-returns for all assets (except savings, bonds, options)
    ##########################################################################

    rts = {}
    for key, a in self.instruments.items():
        if not a: continue
        l = []
        if isinstance(a[0], pa.Stock):
            for s in a:
                df = yf.download(s.symbol, period=period, auto_adjust=True, progress=False)['Close']
                ldf = np.log(df).diff()
                if len(ldf) < horizon: raise ValueError(f'Not enough historical data for {s.name}!')
                cldf = ldf.rolling(window=horizon, min_periods=horizon).sum().dropna()
                l.append(cldf)
        if isinstance(a[0], (pa.Future, pa.Option)):
            pass
        if isinstance(a[0], pa.SavingsPlan):
            pass
        if isinstance(a[0], pa.Bond): continue
        rts[key] = l

    rts_df = pd.concat({k: pd.concat(v, axis=1) for k, v in rts.items()}, axis=1)
    rts_df.dropna(inplace=True)
    rts_np = rts_df.to_numpy()

    # get weight (in %) of assets in Stock, Futures classes
    # relative to total portfolio value
    wgts = self.get_weight_vector()
    wgts_v = np.array([a for b in [v for k, v in wgts.items() if k in ['Stocks', 'Futures']] for a in b])

    # get current total value of all assets
    p0 = self.get_asset_values().sum()

    ##########################################################################
    # MC using normal distribution + ML estimate
    ##########################################################################
    if dist == 'normal':
        # helper function rearranging 1d array of arguments into paramters
        def _unpack_norm(args, d):
            mean = args[:d]

            k = d * (d + 1) // 2
            chol_fl = args[d:d + k]
            chol = np.zeros((d, d))
            for i in range(d):
                for j in range(i + 1):
                    chol[i][j] = chol_fl[i + j]
            C = chol @ chol.T

            return mean, C

        # helper function rearranging parameters into 1d array
        def _pack_norm(mean, C):
            d = len(mean)

            chol = np.linalg.cholesky(C)
            chol_fl = chol[np.tril_indices(d)]

            return np.concatenate([mean, chol_fl])

        # multivariate normal PDF
        def _loglik_norm(args, x):
            n, p = x.shape
            mean, C = _unpack_norm(args, p)

            constant = -1 / 2 * p * np.log(2 * np.pi)
            sign, slogdet = np.linalg.slogdet(C)
            constant -= slogdet

            dif = x - mean
            Q = np.sum(dif @ np.linalg.inv(C) * dif, axis=1)

            out = np.sum(constant - 1 / 2 * Q)
            return -out

        # ML estimator function
        def norm_est(X, mean0, C0):
            args0 = _pack_norm(mean0, C0)
            res = minimize(_loglik_norm, args0, args=(X,), method='L-BFGS-B')
            return res

        # sample mean and covariance as first guess
        mean0 = rts_df.mean().to_numpy()
        C0 = rts_df.cov(ddof=0).to_numpy()

        # MLE mean and covariance
        est = norm_est(rts_np, mean0, C0)
        mean, C = _unpack_norm(est.x, len(mean0))

        # MC simulation with 1e5 observations
        sample_paths_norm = multivariate_normal.rvs(mean=mean, cov=C, size=(n_scen, horizon)) @ wgts_v
        sample_paths_norm = np.array(sample_paths_norm).reshape(n_scen, horizon)  # forcing 2d array when horizon=1
        comp_rts_norm = sample_paths_norm.cumsum(axis=1)
        price_paths_norm = p0 * np.exp(comp_rts_norm)

        VaR_norm = p0 - quantile(price_paths_norm[:, -1], alpha)
        return VaR_norm

    ##########################################################################
    # MC using student-t distribution + ML estimate
    ##########################################################################
    else:
        # helper function rearranging 1d array of arguments into paramters
        def unpack_t(args, d):
            mu = args[:d]

            k = d * (d + 1) // 2
            chol_fl = args[d:d + k]
            chol = np.zeros((d, d))
            for i in range(d):
                for j in range(i + 1):
                    chol[i][j] = chol_fl[i + j]
            Sigma = chol @ chol.T

            nu = np.exp(args[d + k]) + 2

            return mu, Sigma, nu

        # helper function rearranging parameters into 1d array
        def pack_t(mu, Sig, nu):
            d = len(mu)

            chol = np.linalg.cholesky(Sig)
            chol_fl = chol[np.tril_indices(d)]

            lognu = np.log(nu + 2)

            return np.concatenate([mu, chol_fl, [lognu]])

        # multivariate normal PDF
        from scipy.special import gammaln
        def loglik_t(args, x):
            n, p = x.shape
            mu, Sig, nu = unpack_t(args, p)
            Sig_inv = np.linalg.inv(Sig)

            constant = gammaln((nu + p) / 2) - gammaln(nu / 2) - p / 2 * np.log(np.pi * nu)
            sign, slogdet = np.linalg.slogdet(Sig)
            constant -= 1 / 2 * slogdet
            dist = x - mu
            Q = np.sum(dist @ Sig_inv * dist, axis=1)

            out = np.sum(constant - (nu + p) / 2 * np.log1p(Q / nu))
            return -out

        # ML estimator function
        def student_t_est(X, mu0, Sig0, nu0):
            args0 = pack_t(mu0, Sig0, nu0)
            res = minimize(loglik_t, args0, args=(X,), method='L-BFGS-B')
            return res

        # first guess for mu, Sig, nu
        mu0 = rts_df.mean().to_numpy()
        Sig0 = rts_df.cov(ddof=0).to_numpy()
        nu0 = 2.01

        # MLE mean and covariance
        est = student_t_est(rts_np, mu0, Sig0, nu0)
        mu, Sig, nu = unpack_t(est.x, len(mu0))

        # MC simulation with 1e5 observations
        sample_paths_t = multivariate_t.rvs(loc=mu, shape=Sig, df=nu, size=(n_scen, horizon)) @ wgts_v
        sample_paths_t = np.array(sample_paths_t).reshape(n_scen, horizon)  # force 2d array when horizon=1
        comp_rts_t = sample_paths_t.cumsum(axis=1)
        price_paths_t = p0 * np.exp(comp_rts_t)

        VaR_t = p0 - quantile(price_paths_t[:, -1], alpha)
        return VaR_t


