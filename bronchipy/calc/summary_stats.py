import logging
import warnings
import pandas as pd
import numpy as np
from .visualise import save_pi10_figure
from sklearn.linear_model import LinearRegression


def fractal_dimension(c, plot_type=None):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.asarray(c > 0, dtype=int)

    dim = np.ndim(c)

    if dim > 3:
        raise ValueError("Maximum dimension is 3.")

    if dim == 1 and c.shape[0] != 1:
        c = c.reshape(1, -1)

    width = np.max(c.shape)
    p = np.log2(width)

    if p != int(p) or any(s != width for s in c.shape):
        p = int(np.ceil(p))
        width = 2**p
        new_shape = (width, ) * dim
        mz = np.zeros(new_shape, dtype=int)
        slices = tuple(slice(0, s) for s in c.shape)
        mz[slices] = c
        c = mz

    n = np.zeros(p + 1, dtype=int)
    n[p] = np.sum(c)

    for g in range(p - 1, -1, -1):
        siz = 2**(p - g)
        siz2 = siz // 2

        if dim == 1:
            for i in range(0, width - siz + 1, siz):
                c[i] = c[i] or c[i + siz2]
            n[g] = np.sum(c[0:width - siz + 1:siz])
        elif dim == 2:
            for i in range(0, width - siz + 1, siz):
                for j in range(0, width - siz + 1, siz):
                    c[i, j] = (c[i, j] or c[i + siz2, j] or c[i, j + siz2]
                               or c[i + siz2, j + siz2])
            n[g] = np.sum(c[0:width - siz + 1:siz, 0:width - siz + 1:siz])
        elif dim == 3:
            for i in range(0, width - siz + 1, siz):
                for j in range(0, width - siz + 1, siz):
                    for k in range(0, width - siz + 1, siz):
                        c[i, j, k] = (c[i, j, k] or c[i + siz2, j, k]
                                      or c[i, j + siz2, k]
                                      or c[i + siz2, j + siz2, k]
                                      or c[i, j, k + siz2]
                                      or c[i + siz2, j, k + siz2]
                                      or c[i, j + siz2, k + siz2]
                                      or c[i + siz2, j + siz2, k + siz2])
            n[g] = np.sum(c[0:width - siz + 1:siz, 0:width - siz + 1:siz,
                            0:width - siz + 1:siz])

    n = n[::-1]
    r = 2**np.arange(0, p + 1)

    if plot_type == "slope":
        s = -np.gradient(np.log(n)) / np.gradient(np.log(r))
        return r, s
    else:
        return n, r


def calc_pi10(wa: list,
              rad: list,
              plot: bool = False,
              name: str = "anon",
              save_dir: str = "./") -> float:

    # Calculate regression line
    x = np.array(rad).reshape((-1, 1))
    x = (2 * np.pi) * x
    logging.debug(f"Radii {rad}\nPerimeters {x}")
    y = np.array(wa)
    y = np.sqrt(y)
    logging.debug(f"Square Root Wall Areas {y}")

    # Calculate best fit for regression line
    pi10_model = LinearRegression(n_jobs=-1).fit(x, y)
    logging.info(f"Pi10 R2 value is: {pi10_model.score(x, y)}")
    logging.info(
        f"Slope {pi10_model.coef_} and intercept {pi10_model.intercept_}")

    # Get sqrt WA for hypothetical airway of 10mm internal perimeter
    pi10 = pi10_model.predict([[10]])

    if plot:
        save_pi10_figure(x, y, pi10_model, pi10, name=name, savedir=save_dir)

    return pi10[0]


def param_by_gen(air_tree: pd.DataFrame, gen: int, param: str) -> float:
    """

    Parameters
    ----------
    param    : parameter to summarise
    gen      : generation to summarise
    air_tree : pandas dataframe
    """
    return air_tree.groupby("generation")[param].describe().at[gen, "mean"]
    # return air_tree[[param, "generation"]].groupby("generation").describe().at[gen, "mean"]


def agg_param(tree: pd.DataFrame, gens: list, param: str) -> float:
    """

    Parameters
    ----------
    tree
    gens
    param
    """

    return tree[(tree.generation >= gens[0])
                & (tree.generation <= gens[1])][param].mean()


def total_count(tree: pd.DataFrame) -> int:
    return tree.index.max()
