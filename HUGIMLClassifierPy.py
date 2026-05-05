"""
* This file is copyright (c) 2026 Srikumar Krishnamoorthy
*
* This program is free software: you can redistribute it and/or modify it under the
* terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
*
* This program is distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
* A PARTICULAR PURPOSE. See the GNU General Public License for more details.
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.


HUG-IML Classifier — Pure Python Implementation
=================================================

A pure-Python, scikit-learn compatible implementation of the High Utility Gain
Interpretable Machine Learning (HUG-IML) classifier described in:

    Krishnamoorthy, S. (2024). "Interpretable Classifier Models for Decision
    Support Using High Utility Gain Patterns." IEEE Access, 12, 126088–126107.

The model extracts High Utility Gain (HUG) patterns from labelled tabular data,
transforms the input into a binary pattern-presence matrix, and fits an
interpretable downstream classifier (logistic regression by default) on that
matrix.  The resulting patterns are human-readable and serve as the primary
source of model explanations.

Quick start
-----------
Two usage paths are supported:

**Path A — prepareXy** (recommended when the full dataset is available upfront):

    from HUGIMLClassifierPy import HUGIMLClassifierPy
    from sklearn.model_selection import train_test_split

    clf = HUGIMLClassifierPy(B=7, L=1, G=5e-3)
    X, y = clf.prepareXy(X_df, y_series)          # detects column types
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    print(clf.get_hug_features())                  # e.g. ['age=[35,50]', 'gender=F']
    print(clf.get_pattern_info())                  # utility / IG / support table

**Path B — constructor allCols/origColumns** (use inside cross-validation loops
or any context where the full dataset is not available at construction time):

    clf = HUGIMLClassifierPy(
        allCols=[int_col_names, float_col_names, cat_col_names],
        origColumns=X_df.columns.tolist(),
        B=7, L=1, G=5e-3,
    )
    clf.fit(X_train, y_train)
    clf.predict(X_test)

Column type handling
--------------------
The model treats integer, float, and categorical columns differently:

- **Integer columns**: binned on raw integer values (no MinMax scaling).
  Pattern labels show integer bin edges, e.g. ``age=[35,50]``.
  External utility = |Pearson correlation of discretised column with y|.

- **Float columns**: MinMax-scaled to [0, 1] before binning.
  Pattern labels show original-scale edges, e.g. ``income=[40000.000,65000.000]``.
  External utility = |Pearson correlation of discretised column with y|.

- **Categorical columns**: one item per unique value (one-hot treatment).
  Pattern labels show original category values, e.g. ``gender=F``.
  External utility = NMI(binarised column, y) computed per category value.
  Internal utility = 1.0 if point-biserial correlation with y is positive,
  0.05 otherwise.

For all column types, utilities are normalised per class and a minimum
information-gain threshold G filters out uninformative patterns.
"""

import copy
import heapq
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import check_X_y as _check_X_y
    from sklearn.utils.validation import check_array as _check_array
except ImportError:                                         # sklearn >= 1.6
    from sklearn.utils.validation import validate_data as _vd   # type: ignore
    def _check_X_y(X, y, **kw):   return _vd(None, X, y, **kw)
    def _check_array(X, **kw):    return _vd(None, X, **kw)

warnings.filterwarnings("ignore")


# =============================================================================
# Discretisation helpers
# =============================================================================

def _entropy(y: np.ndarray, n_classes: int) -> float:
    """
    Normalised Shannon entropy of integer class labels, in [0, 1].

    Normalised by log(n_classes) so values from datasets with different numbers
    of classes are comparable.  Returns 0.0 for empty arrays or single-class
    inputs (zero uncertainty).
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    total  = counts.sum()
    if total == 0:
        return 0.0
    p    = counts / total
    base = math.log(max(n_classes, 2))
    h    = -sum(pi * math.log(pi) / base for pi in p if pi > 0)
    return max(h, 0.0)


def _ig_col(x_disc: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """
    Information gain of a discretised column with respect to class labels.

    IG = H(y) - sum_v[ P(x=v) * H(y | x=v) ]
    """
    base  = _entropy(y, n_classes)
    total = len(y)
    ce    = 0.0
    for v in np.unique(x_disc):
        m   = x_disc == v
        ce += m.sum() / total * _entropy(y[m], n_classes)
    return round(base - ce, 6)


def _nmi_binary(x_bin: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """
    Normalised Mutual Information between a binarised (0/1) column and y.

    Computed on a per-category indicator column (one-hot binarised), not on
    the full multi-valued categorical column.  Returns a value in [0, 1];
    returns 0.0 for constant columns.
    """
    vals, x_counts = np.unique(x_bin, return_counts=True)
    if len(vals) <= 1:
        return 0.0
    n = len(y)
    px = x_counts / n
    hx = -sum(p * math.log(p) for p in px if p > 0)
    y_counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    py = y_counts / n
    hy = -sum(p * math.log(p) for p in py if p > 0)
    if hx == 0 or hy == 0:
        return 0.0
    mi = 0.0
    for v, nx_v in zip(vals, x_counts):
        mask = x_bin == v
        y_v  = np.bincount(y[mask].astype(int), minlength=n_classes).astype(float)
        for c in range(n_classes):
            nxy = y_v[c]
            if nxy > 0:
                mi += (nxy / n) * math.log((nxy / n) / ((nx_v / n) * (y_counts[c] / n)))
    return round(max(0.0, min(1.0, mi / math.sqrt(hx * hy))), 6)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation coefficient between x and y.

    Returns 0.0 when x has zero variance or the result is not finite.
    """
    if x.std() == 0:
        return 0.0
    r = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    return 0.0 if not math.isfinite(r) else round(r, 6)


def _kbins(col: np.ndarray, nb: int):
    """
    Quantile-based discretisation of a 1-D array into nb bins.

    Returns (binned_col, edges) where binned_col contains 0-based bin indices
    and edges is the sorted array of unique cut-points including both extremes.
    Duplicate quantile edges are collapsed so the actual bin count may be less
    than nb.  Constant columns receive a two-edge fallback to prevent
    out-of-bounds indexing.
    """
    qs    = np.linspace(0, 100, nb + 1)
    edges = np.unique(np.percentile(col, qs))
    if len(edges) < 2:
        lo    = col.min()
        edges = np.array([lo, lo + 1e-9])
    binned = np.searchsorted(edges[1:-1], col, side='right')
    return binned, edges


def _choose_nb(col: np.ndarray, y: np.ndarray, n_classes: int,
               B: int, distinct_count: int) -> int:
    """
    Select the number of bins for one column.

    The bin count is capped at (distinct_count - 1) so that integer columns
    with few unique values are never over-binned into fractional ranges.
    The result is always at least 2.

    When B == -1 the bin count that maximises information gain is chosen by
    searching over [2, 20].  Otherwise B is used directly, subject to the
    same cap and minimum.
    """
    if B == -1:
        best_ig, best_nb = 0.0, 2
        for nb in range(2, 21):
            nb_capped = max(min(distinct_count - 1, nb), 2)
            disc, _   = _kbins(col, nb_capped)
            ig        = _ig_col(disc, y, n_classes)
            if ig > best_ig:
                best_ig, best_nb = ig, nb_capped
        return best_nb
    else:
        return max(min(distinct_count - 1, B), 2)


# =============================================================================
# Transaction generation
# =============================================================================

class _TransactionData:
    """
    Container for all artefacts produced during training by _prepare_transactions.

    Passed to _apply_bins for test-data transformation and to _THUIsl for mining.
    All attributes are set once at construction and treated as read-only.

    Attributes
    ----------
    transactions : list of list of (item_id, utility)
        One inner list per training row; sentinel [(-1, 0.0)] for zero-utility rows.
    item_twu  : list of float
        Transaction-Weighted Utility per item (1-indexed, length = n_items).
    item_map  : dict {item_id: str}
        Human-readable label per item, e.g. ``'age=[35,50]'`` or ``'gender=F'``.
    RIU       : list of float
        Raw Item Utility — sum of item utility across all transactions it appears in.
    disc      : np.ndarray (n, p) int32
        Discretised/encoded representation of X used to build transactions.
        Integer and float columns hold 0-based bin indices; categorical columns
        hold 0-based category codes (-1 for missing/NaN).
    bn2id     : dict {bin_key: item_id}
        Maps the integer bin-key (bi * 10000 + col_idx) to the 1-based item id.
    colnew    : list of int
        Bin-keys of all items with non-zero utility; used for membership testing.
    nb_col    : list of int
        Number of bins (integer/float) or unique categories (categorical) per column.
    bkey      : callable(bi, j) -> int
        Encodes a 1-based bin/category index bi and column index j as a single int.
    ber       : list of np.ndarray
        Normalised right bin edges per numerical column (IU values for positive corr).
        Placeholder arrays [1.0] for categorical columns.
    cv        : list of float
        Pearson correlation of discretised column with y for integer/float columns.
        0.0 placeholder for categorical columns (NMI is per-bin, stored in cat_corr).
    all_edges : list of np.ndarray
        Raw bin edges per numerical column (scaled for float, raw for integer).
        Placeholder [0.0, 1.0] for categorical columns.
    col_min   : np.ndarray (p,)
        Per-column minimum used for MinMax scaling; 0.0 for integer and cat columns.
    col_range : np.ndarray (p,)
        Per-column range used for MinMax scaling; 1.0 for integer and cat columns.
    is_cat    : np.ndarray (p,) bool
        True for categorical columns.
    is_int    : np.ndarray (p,) bool
        True for integer columns (no MinMax scaling applied).
    cat_categories : list (p,)
        Sorted array of original label values per categorical column; None for others.
    cat_corr  : list (p,)
        Dict {original_label: point-biserial correlation} per categorical column;
        None for numerical columns.
    """
    __slots__ = ('transactions', 'item_twu', 'item_map', 'RIU',
                 'disc', 'bn2id', 'colnew', 'nb_col',
                 'bkey', 'ber', 'cv', 'all_edges', 'col_min', 'col_range',
                 'is_cat', 'is_int', 'cat_categories', 'cat_corr')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def riu_thresh(self, k: int) -> float:
        """Return the k-th largest RIU value; 0.0 when k exceeds the list length."""
        vals = self.RIU
        if k <= 0 or not vals:
            return 0.0
        return sorted(vals, reverse=True)[min(k - 1, len(vals) - 1)]


def _prepare_transactions(X: np.ndarray, y: np.ndarray,
                           B: int = -1,
                           col_names: list = None,
                           is_cat: np.ndarray = None,
                           is_int: np.ndarray = None,
                           X_cat_raw: list = None) -> _TransactionData:
    """
    Build utility-annotated transactions from labelled training data.

    The function runs in three passes:
      1. Discretise / encode each column (integer, float, or categorical path).
      2. Compute external × internal utility per bin/category and register items.
      3. Assemble the transaction list used by the top-K HUI miner.

    Integer columns
    ~~~~~~~~~~~~~~~
    Binned directly on raw integer values — no MinMax scaling is applied.
    The number of bins is capped at (distinct_count - 1) so that columns with
    few unique values are never split into fractional ranges.  Bin-range labels
    show integer values, e.g. ``age=[35,50]``.
    External utility = |Pearson correlation of the discretised column with y|.
    Internal utility = normalised right bin-edge; edges are reversed for
    negatively correlated columns.

    Float columns
    ~~~~~~~~~~~~~
    MinMax-scaled to [0, 1] before binning (same distinct-count cap).
    Bin-range labels are inverse-transformed back to original scale, e.g.
    ``income=[40000.000,65000.000]``.
    External utility = |Pearson correlation of the discretised (scaled) column
    with y|.  Internal utility = normalised right bin-edge in [0, 1].

    Categorical columns
    ~~~~~~~~~~~~~~~~~~~
    One item per unique category value (one-hot treatment); no binning.
    External utility = NMI(indicator_column, y) computed per category value,
    where the indicator column is 1 for that value and 0 elsewhere.
    Internal utility = 1.0 if the point-biserial correlation of that indicator
    with y is positive, else 0.05.
    Pattern labels show original string values, e.g. ``gender=F``.

    All utilities are normalised by the per-class maximum before building
    transactions.

    Parameters
    ----------
    X         : float64 ndarray (n, p).
                Integer and float columns carry cleaned numeric values.
                Categorical column slots hold 0.0 placeholders; actual values
                are supplied through X_cat_raw.
    y         : int64 ndarray (n,).  Class labels (0-indexed, minority class
                assigned the highest integer label).
    B         : int.  Number of bins per numerical column.  Use -1 for
                supervised auto-selection (maximises per-column IG over [2,20]).
    col_names : list of str or None.  Column names used in pattern labels.
                Falls back to 'col0', 'col1', … when None or wrong length.
    is_cat    : bool ndarray (p,).  True for categorical columns.
    is_int    : bool ndarray (p,).  True for integer columns.
    X_cat_raw : list (p,) of numpy object arrays.  X_cat_raw[j] contains the
                original string/object values for categorical column j; None
                for numerical columns.

    Returns
    -------
    _TransactionData
    """
    n, p  = X.shape
    n_cls = len(np.unique(y))
    if is_cat is None:
        is_cat = np.zeros(p, dtype=bool)
    if is_int is None:
        is_int = np.zeros(p, dtype=bool)
    if X_cat_raw is None:
        X_cat_raw = [None] * p

    _names = col_names if (col_names is not None and len(col_names) == p) \
             else [f'col{j}' for j in range(p)]

    # MinMax params — computed only for float columns; int/cat cols get 0/1
    col_min   = np.zeros(p)
    col_range = np.ones(p)
    for j in range(p):
        if not is_cat[j] and not is_int[j]:
            cmin = X[:, j].min()
            cmax = X[:, j].max()
            col_min[j]   = cmin
            col_range[j] = (cmax - cmin) if cmax > cmin else 1.0

    disc           = np.zeros((n, p), dtype=np.int32)
    nb_col         = []
    ber            = []      # normalised right bin edges (num cols); placeholder for cat
    bro            = []      # original-space bin ranges [(lo,hi), ...]
    all_edges      = []      # raw edges (scaled for float, raw for int)
    cv             = []      # |corr| for num, NMI-like for cat (filled per-bin below)
    cat_categories = [None] * p
    cat_corr       = [None] * p

    # ── Pass 1: discretise / encode each column ──────────────────────────────
    for j in range(p):
        if is_cat[j]:
            # ── Categorical ──────────────────────────────────────────────────
            raw        = X_cat_raw[j]
            valid_mask = np.array([v is not None and
                                   not (isinstance(v, float) and math.isnan(v))
                                   for v in raw])
            raw_valid  = raw[valid_mask]
            y_valid    = y[valid_mask]
            uniq       = sorted(set(raw_valid.tolist()),
                                key=lambda x: (str(type(x)), x))
            cat_categories[j] = np.array(uniq, dtype=object)
            label2int  = {v: i for i, v in enumerate(uniq)}

            codes_all  = np.full(n, -1, dtype=np.int32)
            for r in range(n):
                v = raw[r]
                if v in label2int:
                    codes_all[r] = label2int[v]
            disc[:, j] = codes_all

            # Point-biserial sign per unique label (for IU)
            pb_sign = {}
            for v in uniq:
                x_bin = (codes_all == label2int[v]).astype(float)
                pb_sign[v] = _pearson(x_bin, y.astype(float))
            cat_corr[j] = pb_sign

            nb_col.append(len(uniq))
            all_edges.append(np.array([0.0, 1.0]))   # placeholder
            ber.append(np.array([1.0]))               # placeholder
            bro.append([(0.0, 1.0)])                  # placeholder
            cv.append(0.0)                            # filled per-bin; not used as scalar

        elif is_int[j]:
            # ── Integer: bin on raw values, NO MinMax scaling ────────────────
            col_raw    = X[:, j]                      # raw integer values
            distinct   = int(np.unique(col_raw).size)
            nb         = _choose_nb(col_raw, y, n_cls, B, distinct)
            d, edges   = _kbins(col_raw, nb)
            nb_act     = len(edges) - 1
            disc[:, j] = d
            nb_col.append(nb_act)
            all_edges.append(edges)

            # Normalise right edges by the maximum edge (raw integer space)
            mx = edges.max() if edges.max() > 0 else 1.0
            ber.append(edges[1:] / mx)

            # bin-range labels in raw integer space — format as integers
            bro.append([(edges[bi], edges[bi + 1]) for bi in range(nb_act)])

            # Correlation on discretised column
            cv.append(_pearson(d.astype(float), y.astype(float)))

        else:
            # ── Float: MinMax scale then bin ──────────────────────────────────
            col_scaled = (X[:, j] - col_min[j]) / col_range[j]
            distinct   = int(np.unique(col_scaled).size)
            nb         = _choose_nb(col_scaled, y, n_cls, B, distinct)
            d, edges   = _kbins(col_scaled, nb)
            nb_act     = len(edges) - 1
            disc[:, j] = d
            nb_col.append(nb_act)
            all_edges.append(edges)          # edges are in scaled [0,1] space

            mx = edges.max() if edges.max() > 0 else 1.0
            ber.append(edges[1:] / mx)

            # Inverse-transform edges back to original float space for labels
            lo_o = edges[:-1] * col_range[j] + col_min[j]
            hi_o = edges[1:]  * col_range[j] + col_min[j]
            bro.append(list(zip(lo_o, hi_o)))

            cv.append(_pearson(d.astype(float), y.astype(float)))

    # ── Pass 2: build item registry (EU × IU per bin/category) ───────────────
    def bkey(bi, j):
        return bi * 10_000 + j

    item_map, bn2id, colnew = {}, {}, []
    tu, tu_y = {}, defaultdict(float)
    ic = 0

    for j in range(p):
        nb = nb_col[j]
        if is_cat[j]:
            # NMI computed per bin (one-hot binarised), matching Java exactly:
            # corrValsCat / nmiScore are per-bin, not per-column.
            codes_all = disc[:, j]
            for idx, v in enumerate(cat_categories[j]):
                bi    = idx + 1
                x_bin = (codes_all == idx).astype(np.int32)
                nmi   = _nmi_binary(x_bin, y, n_cls)
                pb    = cat_corr[j].get(v, 0.0)
                iu_t  = 1.0 if pb > 0 else 0.05
                eiu   = nmi * iu_t
                bname = bkey(bi, j)
                if eiu > 0:
                    ic += 1
                    item_map[ic] = f"{_names[j]}={v}"
                    colnew.append(bname)
                    bn2id[bname] = ic
                for yi in range(n_cls):
                    tu[(bname, yi)] = eiu
                    tu_y[yi] = max(tu_y[yi], eiu)
        else:
            # Integer or float: EU = |corr on disc col|, IU = normalised edge
            eu = abs(cv[j])
            for bi in range(1, nb + 1):
                iu_t  = ber[j][bi - 1] if cv[j] >= 0 else ber[j][nb - bi]
                eiu   = eu * iu_t
                bname = bkey(bi, j)
                if eiu > 0:
                    ic += 1
                    lo, hi = bro[j][bi - 1]
                    if is_int[j]:
                        label = f"{_names[j]}=[{int(lo)},{int(hi)}]"
                    else:
                        label = f"{_names[j]}=[{lo:.3f},{hi:.3f}]"
                    item_map[ic] = label
                    colnew.append(bname)
                    bn2id[bname] = ic
                for yi in range(n_cls):
                    tu[(bname, yi)] = eiu
                    tu_y[yi] = max(tu_y[yi], eiu)

    # Normalise utility per class
    for k in tu:
        _, yi = k
        tu[k] = tu[k] / tu_y[yi] if tu_y[yi] > 0 else 0.0

    # ── Pass 3: build transaction list ───────────────────────────────────────
    transactions = []
    item_twu     = [0.0] * ic
    RIU          = [0.0] * ic

    for r in range(n):
        yi = int(y[r])
        tutils, trans = 0.0, []
        for j in range(p):
            if is_cat[j]:
                code = int(disc[r, j])
                if code < 0:
                    continue
                bi = code + 1
            else:
                bi = int(disc[r, j]) + 1

            bname = bkey(bi, j)
            txk   = (bname, yi)
            if bname not in colnew or txk not in tu:
                continue
            iu  = round(tu[txk], 6)
            iid = bn2id[bname]
            trans.append((iid, iu))
            tutils += iu

        if tutils > 0:
            transactions.append(trans)
            for iid, iu in trans:
                item_twu[iid - 1] += tutils
                RIU[iid - 1]      += iu
        else:
            transactions.append([(-1, 0.0)])

    return _TransactionData(
        transactions=transactions, item_twu=item_twu,
        item_map=item_map, RIU=RIU, disc=disc,
        bn2id=bn2id, colnew=colnew, nb_col=nb_col,
        bkey=bkey, ber=ber, cv=cv,
        all_edges=all_edges, col_min=col_min, col_range=col_range,
        is_cat=is_cat, is_int=is_int,
        cat_categories=cat_categories, cat_corr=cat_corr,
    )



def _apply_bins(X_raw: np.ndarray, td: _TransactionData,
                X_cat_raw: list = None) -> list:
    """
    Apply training-fit discretisation to test or validation data.

    Produces a transaction list in the same item-id space as the training
    transactions, so the result can be passed directly to _build_matrix.

    Integer columns  : binned on raw values using training edges (no scaling).
    Float columns    : MinMax-scaled with training parameters, then binned.
    Categorical cols : original label looked up in the training category list.

    Out-of-range numerical values are clamped to the nearest boundary bin.
    Unseen categorical values are silently skipped (no item emitted for that
    column in that row).  Missing categorical values (None or NaN) are also
    skipped.

    Parameters
    ----------
    X_raw     : float64 ndarray (n, p).  Numerical column values; categorical
                slots hold 0.0 placeholders.
    td        : _TransactionData produced by _prepare_transactions on the
                training data.
    X_cat_raw : list (p,) of numpy object arrays with original string/object
                values for categorical columns; None entries for numerical cols.

    Returns
    -------
    list of list of (item_id, 1.0), one inner list per sample.
    Rows with no matching items contain the sentinel [(-1, 0.0)].
    """
    n, p  = X_raw.shape
    if X_cat_raw is None:
        X_cat_raw = [None] * p

    # Pre-build label→code lookup for categorical columns
    label2code = [None] * p
    for j in range(p):
        if td.is_cat[j]:
            label2code[j] = {v: i for i, v in enumerate(td.cat_categories[j])}

    test_trans = []
    for r in range(n):
        row = []
        for j in range(p):
            if td.is_cat[j]:
                v    = X_cat_raw[j][r] if X_cat_raw[j] is not None else None
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                code = label2code[j].get(v)
                if code is None:
                    continue
                bi = code + 1
            else:
                edges = td.all_edges[j]
                nb    = td.nb_col[j]
                if td.is_int[j]:
                    val = X_raw[r, j]
                else:
                    val = (X_raw[r, j] - td.col_min[j]) / td.col_range[j]
                bi = int(np.searchsorted(edges[1:-1], val, side='right')) + 1
                bi = max(1, min(bi, nb))

            bname = td.bkey(bi, j)
            if bname in td.bn2id:
                row.append((td.bn2id[bname], 1.0))
        test_trans.append(row if row else [(-1, 0.0)])
    return test_trans


# =============================================================================
# Utility-list data structures
# =============================================================================

class _El:
    """One element of a utility list, representing a single transaction entry."""
    __slots__ = ('tid', 'iu', 'ru')

    def __init__(self, t, i, r):
        self.tid = t    # transaction id (row index)
        self.iu  = i    # internal utility of this item in the transaction
        self.ru  = r    # remaining utility of all items after this one


class _UL:
    """
    Utility List for one candidate itemset, used by the top-K HUI miner.

    Maintains the sorted list of transaction elements (_El), the sum of
    internal utilities (sI), and the sum of remaining utilities (sR).
    sI + sR forms an upper bound on the itemset's potential utility, used
    for LIU pruning.  The information gain (ig) is set by compute_ig after
    the utility list is fully constructed.
    """

    def __init__(self, item):
        self.item = item
        self.sI   = 0.0
        self.sR   = 0.0
        self.els  = []
        self.ig   = 0.0

    def add(self, el: _El):
        """Append one transaction element and update the running utility sums."""
        self.sI += el.iu
        self.sR += el.ru
        self.els.append(el)

    def compute_ig(self, parent, ytrain: list, n_cls: int):
        """
        Compute the information gain of this itemset relative to its parent.

        IG measures how much discriminative power this pattern adds over its
        parent pattern (or over the full dataset for depth-1 patterns).
        The result is stored in self.ig.

        Parameters
        ----------
        parent  : _UL or None.  The parent itemset's utility list, or None for
                  the root level.
        ytrain  : list of int.  Class labels for all training transactions.
        n_cls   : int.  Number of distinct classes.
        """
        tids = [e.tid for e in self.els]
        if not tids:
            self.ig = 0.0
            return

        y        = np.array(ytrain)
        y_in     = y[tids]
        base     = (_entropy(y, n_cls) if parent is None
                    else _entropy(y[[e.tid for e in parent.els]], n_cls))
        n_parent = len(y) if parent is None else len(parent.els)
        tid_set  = set(tids)

        if parent is None:
            y_out = y[np.array([i for i in range(len(y)) if i not in tid_set])]
        else:
            ptids    = [e.tid for e in parent.els]
            out_tids = [t for t in ptids if t not in tid_set]
            y_out    = (y[np.array(out_tids, dtype=int)]
                        if out_tids else np.array([], dtype=int))

        if n_parent == 0:
            self.ig = 0.0
            return

        ce = (len(y_in)  / n_parent * _entropy(y_in,  n_cls) +
              len(y_out) / n_parent * _entropy(y_out, n_cls))
        self.ig = round(max(base - ce, 0.0), 6)


# =============================================================================
# Top-K HUI miner
# =============================================================================

class _THUIsl:
    """
    Top-K High Utility Itemset (HUI) miner with information-gain filtering.

    Mines the K itemsets with the highest total utility from utility-annotated
    transactions, subject to a minimum information-gain threshold G and a
    maximum pattern length L.  Uses the following pruning strategies:

    - LIU pruning: prune any node where sI + sR < minU (upper-bound on utility).
    - EUCS pruning: for L > 1, prune pairs whose estimated co-occurrence utility
      is below minU before constructing the child utility list.
    - IG threshold: patterns with IG < G are not saved to the heap.
    - The utility threshold minU is raised to the K-th utility once the heap
      reaches capacity, making subsequent pruning more aggressive.

    Parameters
    ----------
    K : int   — maximum number of patterns to retain.
    L : int   — maximum pattern length (1 = singletons only; -1 = unlimited).
    G : float — minimum information-gain threshold.
    """

    def __init__(self, K: int = 200, L: int = 2, G: float = 1e-4):
        self.K    = K
        self.L    = L
        self.G    = G
        self.heap : list  = []
        self.minU : float = 0.0

    def _save(self, items, ul):
        """Push a qualifying pattern onto the bounded min-heap of size K."""
        u = ul.sI
        if len(self.heap) < self.K:
            heapq.heappush(self.heap, (u, list(items), ul))
            if len(self.heap) == self.K:
                self.minU = self.heap[0][0]
        elif u > self.minU:
            heapq.heapreplace(self.heap, (u, list(items), ul))
            self.minU = self.heap[0][0]

    def _child(self, p: _UL, x: _UL) -> _UL:
        """
        Build the child utility list for the itemset (prefix ∪ {x.item}).

        Uses a sorted-merge intersection of the parent utility list p and the
        extension item's utility list x, restricted to transactions where both
        appear.
        """
        c = _UL(x.item)
        i = j = 0
        pe, xe = p.els, x.els
        while i < len(pe) and j < len(xe):
            if pe[i].tid == xe[j].tid:
                c.add(_El(pe[i].tid, pe[i].iu + xe[j].iu, xe[j].ru))
                i += 1; j += 1
            elif pe[i].tid < xe[j].tid:
                i += 1
            else:
                j += 1
        return c

    def mine(self, transactions: list, item_twu: list,
             ytrain: list, n_cls: int) -> list:
        """
        Run the top-K HUI mining algorithm.

        Parameters
        ----------
        transactions : list of list of (item_id, utility)
            One inner list per training row.
        item_twu     : list of float
            Transaction-Weighted Utility per item, 0-indexed (length = n_items).
        ytrain       : list of int
            Class label per training transaction.
        n_cls        : int
            Number of distinct classes.

        Returns
        -------
        self.heap : list of (utility, items, ul) in min-heap order.
        """
        self.minU, self.heap = 0.0, []
        fmap     = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        n_items  = len(item_twu)
        use_eucs = (self.L != 1)

        ul_map = {iid: _UL(iid) for iid in range(1, n_items + 1)
                  if item_twu[iid - 1] >= self.minU}
        sorted_items = sorted(ul_map, key=lambda x: item_twu[x - 1])

        # Single pass: populate utility lists and EUCS co-occurrence map
        for tid, trans in enumerate(transactions):
            if len(trans) == 1 and trans[0][0] == -1:
                continue
            active  = [(it, u) for it, u in trans if it in ul_map]
            if not active:
                continue
            new_twu = sum(u for _, u in active)
            active.sort(key=lambda x: item_twu[x[0] - 1])
            rem = 0.0
            for i in range(len(active) - 1, -1, -1):
                it, u = active[i]
                ul_map[it].add(_El(tid, u, rem))
                if use_eucs:
                    fm = fmap[it]
                    for j2 in range(i + 1, len(active)):
                        oj, ou = active[j2]
                        if oj != it:
                            fm[oj][0] += new_twu
                            fm[oj][1] += u + ou
                rem += u

        # Compute IG for all 1-itemsets before search tree traversal
        for ul in ul_map.values():
            ul.compute_ig(None, ytrain, n_cls)

        uls = [ul_map[i] for i in sorted_items if i in ul_map]
        self._explore([], uls, ytrain, n_cls, 0, fmap)
        return self.heap

    def _explore(self, prefix, uls, ytrain, n_cls, depth, fmap):
        """
        Depth-first search of the itemset lattice.

        Applies LIU pruning (sI + sR < minU), EUCS pruning for extensions when
        L > 1, and the IG threshold G.  Recursion depth is capped at L (or 99
        when L is -1, meaning unlimited).
        """
        maxd = self.L if self.L not in (-1, 0) else 99
        for i, ux in enumerate(uls):
            if ux.sI + ux.sR < self.minU:
                continue
            if ux.sI >= self.minU and ux.ig >= self.G:
                self._save(prefix + [ux.item], ux)
            if depth + 1 >= maxd:
                continue
            ext = []
            for j in range(i + 1, len(uls)):
                uy = uls[j]
                if self.L != 1 and fmap[ux.item][uy.item][0] < self.minU:
                    continue
                ch = self._child(ux, uy)
                if ch.sI + ch.sR >= self.minU:
                    ch.compute_ig(ux, ytrain, n_cls)
                    ext.append(ch)
            if ext:
                self._explore(prefix + [ux.item], ext, ytrain, n_cls, depth + 1, fmap)


# =============================================================================
# Feature-matrix builder
# =============================================================================

def _build_matrix(transactions: list, patterns: list, n: int) -> csr_matrix:
    """
    Build a sparse binary pattern-presence matrix of shape (n, len(patterns)).

    Entry (i, j) is 1 if all items of pattern j are present in transaction i,
    and 0 otherwise.  The matrix is stored in CSR format with float32 values.

    Parameters
    ----------
    transactions : list of list of (item_id, utility)
    patterns     : list of (utility, items, ul) from _THUIsl.heap
    n            : number of transactions (rows)

    Returns
    -------
    csr_matrix, shape (n, len(patterns)), dtype float32
    """
    rows, cols = [], []
    for pi, (_, items, _ul) in enumerate(patterns):
        iset = set(items)
        for tid, trans in enumerate(transactions):
            if iset.issubset(t[0] for t in trans):
                rows.append(tid)
                cols.append(pi)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)),
                      shape=(n, len(patterns)), dtype=np.float32)


# =============================================================================
# HUGIMLClassifierPy - Pure python version
# =============================================================================

class HUGIMLClassifierPy(ClassifierMixin, BaseEstimator):
    """
    HUG-IML interpretable classifier — pure Python, scikit-learn compatible.

    Implements the seven-stage HUG-IML workflow from Krishnamoorthy (2024):

      1. Prepare data (missing value imputation, label encoding of target).
      2. Construct utility-annotated transactions from the training features.
      3. Mine top-K High Utility Gain (HUG) patterns from the transactions.
      4. Build a binary pattern-presence matrix from the mined patterns.
      5. Construct transactions for test data using training-fitted parameters.
      6. Evaluate the model on test data.
      7. Interpret the model via HUG pattern profiles.

    The resulting patterns are human-readable (e.g. ``age=[35,50]``,
    ``income=[40000.000,65000.000]``, ``gender=F``) and serve as the primary
    source of model explanations.

    Column type information must be provided so integer, float, and categorical
    columns are processed correctly (see Parameters below).  Two ways are
    supported:

    **prepareXy** — call on the full dataset before any train/test split.
    The method detects column types from DataFrame dtypes, sets up the
    necessary masks, and encodes the target variable:

        clf = HUGIMLClassifierPy(B=7, L=1, G=5e-3)
        X, y = clf.prepareXy(X_df, y_series)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
        clf.fit(X_tr, y_tr)

    **allCols + origColumns** — supply column type lists in the constructor.
    Use this path when the full dataset is not available at construction time,
    for example inside a cross-validation loop:

        clf = HUGIMLClassifierPy(
            allCols=[int_col_names, float_col_names, cat_col_names],
            origColumns=X_df.columns.tolist(),
            B=7, L=1, G=5e-3,
        )
        clf.fit(X_train, y_train)

    When neither path is used and a plain ndarray is passed to fit, all columns
    are treated as float.  When a DataFrame is passed directly to fit without
    prior setup, column types are inferred from dtypes.

    Parameters
    ----------
    allCols : list of 3 lists, optional
        ``[int_col_names, float_col_names, cat_col_names]`` — lists of column
        names grouped by type.  Must be paired with ``origColumns``.
    origColumns : list of str, optional
        Ordered list of all column names matching the columns of X as passed
        to fit/predict.  Must be paired with ``allCols``.
    B : int, default 5
        Number of quantile bins per numerical feature.  Use -1 for supervised
        auto-selection, which picks the bin count that maximises information
        gain over the range [2, 20].
    L : int, default 1
        Maximum HUG pattern length.  L=1 mines singleton patterns only;
        L=2 mines singletons and pairs; and so on.
    G : float, default 1e-4
        Minimum information-gain threshold.  Patterns whose IG falls below G
        are discarded during mining.
    topK : int, default -1
        Maximum number of top-utility patterns to retain.  -1 means the budget
        is computed automatically as C(100, L).
    base_estimator : sklearn estimator, optional
        Downstream classifier trained on the binary HUG pattern matrix.
        Defaults to LogisticRegression (solver='liblinear' for binary problems,
        'lbfgs' for multi-class).
    verbose : bool, default False
        Print progress messages during fit.

    Attributes (available after fit)
    ---------------------------------
    classes_          : np.ndarray — unique class labels seen during fit.
    n_features_in_    : int        — number of input features.
    feature_names_in_ : list or None — resolved column names.
    cat_cols_mask_    : np.ndarray[bool] — True for categorical columns.
    is_int_mask_      : np.ndarray[bool] — True for integer columns.
    td_               : _TransactionData — all training-time discretisation
                        artefacts needed to transform new data.
    patterns_         : list of (utility, items, ul) — mined HUG patterns.
    x_train_hup_      : csr_matrix — binary pattern matrix for training data.
    model_            : fitted sklearn Pipeline wrapping the downstream estimator.
    """

    def __init__(self,
                 allCols=None, origColumns=None,
                 B=5, L=1, G=1e-4,
                 topK=-1,
                 base_estimator=None,
                 verbose=False):
        self.allCols        = allCols
        self.origColumns    = origColumns
        self.B              = B
        self.L              = L
        self.G              = G
        self.topK           = topK
        self.base_estimator = base_estimator
        self.verbose        = verbose

    def get_params(self, deep=True):
        """Return init parameters as a dict (required for clone and GridSearchCV)."""
        return dict(
            allCols=self.allCols,
            origColumns=self.origColumns,
            B=self.B, L=self.L, G=self.G,
            topK=self.topK,
            base_estimator=(copy.deepcopy(self.base_estimator)
                            if deep else self.base_estimator),
            verbose=self.verbose,
        )

    def set_params(self, **params):
        """Set parameters in-place; returns self for chaining."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def prepareXy(self, X, y):
        """
        Detect column types and encode the target variable.

        Call this method on the full dataset before any train/test split.
        It records which columns are integer, float, or categorical so that
        fit() and predict() can apply the correct utility computation for each
        type.  The DataFrame is returned with the same column order and with
        categorical columns left at their original string values — no encoding
        of features is performed here.

        The target variable y is encoded so that class labels become
        consecutive integers in descending order of class frequency.  The
        majority class receives label 0; the minority class receives the
        highest integer.  The original-to-encoded mapping is stored in
        self.yNewToOriginal.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.  Integer, float, and string/object/category dtype
            columns are all supported.
        y : pd.Series or array-like
            Target labels.

        Returns
        -------
        X : pd.DataFrame
        y : np.ndarray of int64
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame')

        X = X.copy()
        X.columns = [str(c) for c in X.columns]

        catCols = [c for c in X.columns
                   if pd.api.types.is_object_dtype(X[c])
                   or pd.api.types.is_string_dtype(X[c])
                   or pd.api.types.is_categorical_dtype(X[c])]
        intCols = [c for c in X.columns if pd.api.types.is_integer_dtype(X[c])]

        X = X.reset_index(drop=True)
        self.feature_names_in_ = X.columns.tolist()

        self.cat_cols_mask_ = np.array([c in set(catCols) for c in X.columns], dtype=bool)
        self.is_int_mask_   = np.array([c in set(intCols) for c in X.columns], dtype=bool)

        y = np.asarray(y)
        
        return X, y

    @staticmethod
    def _to_float_array(arr, cat_mask: np.ndarray = None):
        """
        Split input data into a float64 numeric array and a list of raw
        categorical value arrays.

        Returns
        -------
        X_num : float64 ndarray (n, p)
            Numerical column values, cleaned: NaN and Inf are replaced by the
            column median.  Categorical column slots hold 0.0 and are never
            used in arithmetic.
        X_cat_raw : list of length p
            X_cat_raw[j] is a numpy object array of the original string/object
            values for categorical column j, preserving them so pattern labels
            can show e.g. ``gender=F``.  Entries for numerical columns are None.

        Parameters
        ----------
        arr      : pd.DataFrame, ndarray, or array-like of shape (n, p).
        cat_mask : bool ndarray (p,).  True marks a categorical column.
                   When None every column is treated as numerical.
        """
        is_df  = isinstance(arr, pd.DataFrame)
        n      = len(arr)
        p      = len(arr.columns) if is_df else np.asarray(arr).shape[1]
        arr_np = None if is_df else np.asarray(arr)   # convert once, not per column

        if cat_mask is None:
            cat_mask = np.zeros(p, dtype=bool)

        X_num     = np.zeros((n, p), dtype=np.float64)
        X_cat_raw = [None] * p

        for j in range(p):
            raw = arr.iloc[:, j] if is_df else arr_np[:, j]

            if cat_mask[j]:
                col_obj = np.asarray(raw, dtype=object)
                for i, v in enumerate(col_obj):
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        col_obj[i] = np.nan
                X_cat_raw[j] = col_obj
                X_num[:, j]  = 0.0
            else:
                col = np.asarray(raw, dtype=np.float64)
                bad = ~np.isfinite(col)
                if bad.any():
                    good     = col[~bad]
                    col[bad] = float(np.median(good)) if good.size > 0 else 0.0
                X_num[:, j] = col

        return X_num, X_cat_raw

    def _effective_topK(self) -> int:
        """
        Return the top-K budget for pattern mining.

        When topK is set explicitly that value is used directly.  Otherwise
        the budget is C(100, L) — the number of length-L combinations over
        100 items — which serves as a conservative upper bound for most
        real datasets.
        """
        if self.topK != -1:
            return self.topK
        nitems = 100
        lsize  = {i: math.comb(nitems, i) for i in range(1, 7)}
        if self.L in (-1, 1):
            return lsize[1]
        elif isinstance(self.L, int) and 2 <= self.L <= 6:
            return lsize[self.L]
        else:
            return lsize[2]

    def _make_estimator(self, n_cls: int):
        """
        Return a fresh copy of the downstream classifier.

        When base_estimator is not set, defaults to LogisticRegression with
        solver='liblinear' for binary problems and 'lbfgs' for multi-class
        (liblinear does not support more than two classes).
        """
        if self.base_estimator is not None:
            return copy.deepcopy(self.base_estimator)
        solver = 'liblinear' if n_cls == 2 else 'lbfgs'
        return LogisticRegression(solver=solver, random_state=0, max_iter=500)

    def _validate_params(self):
        """Raise TypeError or ValueError for parameters with incorrect types or values."""
        if not isinstance(self.B, int):
            raise TypeError(f'B must be an int, got {type(self.B)}')
        if not isinstance(self.L, int):
            raise TypeError(f'L must be an int, got {type(self.L)}')
        if not isinstance(self.G, float):
            raise TypeError(f'G must be a float, got {type(self.G)}')
        if self.allCols is not None or self.origColumns is not None:
            if self.allCols is None or self.origColumns is None:
                raise ValueError(
                    'allCols and origColumns must both be supplied together.')
            if not (isinstance(self.allCols, list) and len(self.allCols) == 3):
                raise ValueError(
                    'allCols must be a list of 3 lists: '
                    '[int_cols, float_cols, cat_cols].')

    def _resolve_col_meta(self, X_train):
        """
        Determine column names, categorical mask, and integer mask for X_train.

        Checks four sources in priority order and sets self.feature_names_in_,
        self.cat_cols_mask_, and self.is_int_mask_ accordingly:

        1. prepareXy was already called — masks are already set on self.
        2. allCols and origColumns were supplied in the constructor — build
           masks from those lists.
        3. X_train is a DataFrame — infer masks from column dtypes.
        4. X_train is a plain ndarray — treat all columns as float (no mask).

        Returns the cat_cols_mask_ array.
        """
        # Path 1: prepareXy already set everything
        if hasattr(self, 'cat_cols_mask_'):
            return self.cat_cols_mask_

        # Path 2: allCols + origColumns supplied in constructor
        if self.allCols is not None and self.origColumns is not None:
            cat_set  = set(self.allCols[2])
            int_set  = set(self.allCols[0])
            col_list = list(self.origColumns)
            self.cat_cols_mask_ = np.array([c in cat_set for c in col_list], dtype=bool)
            self.is_int_mask_   = np.array([c in int_set for c in col_list], dtype=bool)
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        # Path 3: X_train is a DataFrame — infer from dtypes
        if isinstance(X_train, pd.DataFrame):
            col_list = X_train.columns.astype(str).tolist()
            self.cat_cols_mask_ = np.array([
                pd.api.types.is_object_dtype(X_train[c])
                or pd.api.types.is_string_dtype(X_train[c])
                or pd.api.types.is_categorical_dtype(X_train[c])
                for c in X_train.columns
            ], dtype=bool)
            self.is_int_mask_ = np.array([
                pd.api.types.is_integer_dtype(X_train[c])
                for c in X_train.columns
            ], dtype=bool)
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        # Path 4: plain ndarray — all treated as float
        p = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        self.cat_cols_mask_ = np.zeros(p, dtype=bool)
        self.is_int_mask_   = np.zeros(p, dtype=bool)
        if not hasattr(self, 'feature_names_in_'):
            self.feature_names_in_ = None
        return self.cat_cols_mask_

    def fit(self, X_train, y_train):
        """
        Fit the HUG-IML model on training data.

        Executes the first four stages of the HUG-IML workflow:
          1. Resolve column metadata (types and names).
          2. Build utility-annotated transactions from X_train.
          3. Mine top-K HUG patterns.
          4. Build the binary pattern matrix and fit the downstream classifier.

        Parameters
        ----------
        X_train : pd.DataFrame or ndarray, shape (n_samples, n_features)
            Training features.  When passed as a DataFrame after prepareXy,
            categorical columns carry original string values.  When passed as
            an ndarray, column types must have been declared via allCols and
            origColumns in the constructor.
        y_train : array-like of int, shape (n_samples,)
            Class labels.  Should be the encoded labels returned by prepareXy
            when that path is used.

        Returns
        -------
        self
        """
        self._validate_params()

        # Resolve column names, cat mask, and int mask from whichever path was used.
        cat_mask = self._resolve_col_meta(X_train)
        int_mask = getattr(self, 'is_int_mask_', None)

        X_num, X_cat_raw = self._to_float_array(X_train, cat_mask)
        y_train = np.asarray(y_train, dtype=np.int64)
        X_num, y_train = _check_X_y(X_num, y_train, dtype=None)

        self.n_features_in_ = X_num.shape[1]
        self.classes_       = np.unique(y_train)
        n_cls               = len(self.classes_)

        if self.verbose:
            print('HUGIMLClassifier — preparing transactions …')

        # Stage 2: build utility-annotated transactions
        self.td_ = _prepare_transactions(
            X_num, y_train, B=self.B,
            col_names=getattr(self, 'feature_names_in_', None),
            is_cat=cat_mask, is_int=int_mask, X_cat_raw=X_cat_raw)

        K = self._effective_topK()
        if self.verbose:
            print(f'  items={len(self.td_.item_twu)}, K={K}')
            print('  Mining HUG patterns …')

        # Stage 3: mine HUG patterns
        miner = _THUIsl(K=K, L=self.L, G=self.G)
        miner.mine(self.td_.transactions, self.td_.item_twu,
                   y_train.tolist(), n_cls)
        self.patterns_ = miner.heap

        if len(self.patterns_) == 0:
            raise RuntimeError(
                'No HUG patterns found. '
                'Try reducing G, increasing topK, or adjusting B / L.'
            )

        if self.verbose:
            print(f'  {len(self.patterns_)} patterns found')

        # Stage 4: build binary pattern matrix and fit downstream model
        self.x_train_hup_ = _build_matrix(
            self.td_.transactions, self.patterns_, len(y_train))

        self.model_ = Pipeline([('clf', self._make_estimator(n_cls))])
        self.model_.fit(self.x_train_hup_, y_train)

        return self

    def predict_proba(self, X_test) -> np.ndarray:
        """
        Predict class probabilities for X_test.

        Transforms X_test using the mined HUG patterns and applies the
        fitted downstream classifier.

        Parameters
        ----------
        X_test : array-like or DataFrame, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        cat_mask = getattr(self, 'cat_cols_mask_', None)
        X_num, X_cat_raw = self._to_float_array(X_test, cat_mask)
        X_num = _check_array(X_num, dtype=None)
        test_trans = _apply_bins(X_num, self.td_, X_cat_raw)
        x_test_hup = _build_matrix(test_trans, self.patterns_, len(test_trans))
        return self.model_.predict_proba(x_test_hup)

    def predict(self, X_test) -> np.ndarray:
        """
        Predict class labels for X_test.

        Parameters
        ----------
        X_test : array-like or DataFrame, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        check_is_fitted(self)
        cat_mask = getattr(self, 'cat_cols_mask_', None)
        X_num, X_cat_raw = self._to_float_array(X_test, cat_mask)
        X_num = _check_array(X_num, dtype=None)
        test_trans = _apply_bins(X_num, self.td_, X_cat_raw)
        x_test_hup = _build_matrix(test_trans, self.patterns_, len(test_trans))
        return self.model_.predict(x_test_hup)

    def transform(self, X) -> csr_matrix:
        """
        Return the binary HUG pattern matrix for X without making predictions.

        Each column of the returned matrix corresponds to one mined HUG pattern.
        Entry (i, j) is 1 if all items of pattern j are present in row i, else 0.
        Useful when combining the pattern matrix with other features before
        fitting a custom downstream model.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)

        Returns
        -------
        csr_matrix, shape (n_samples, n_patterns)
        """
        check_is_fitted(self)
        cat_mask = getattr(self, 'cat_cols_mask_', None)
        X_num, X_cat_raw = self._to_float_array(X, cat_mask)
        X_num = _check_array(X_num, dtype=None)
        trans = _apply_bins(X_num, self.td_, X_cat_raw)
        return _build_matrix(trans, self.patterns_, len(trans))
        
    def get_hug_features(self) -> list:
        """
        Return a human-readable label for each mined HUG pattern.

        Singleton patterns use the format ``feature=[lo,hi]`` for integer and
        float columns (e.g. ``age=[35,50]``, ``income=[40000.000,65000.000]``)
        and ``feature=value`` for categorical columns (e.g. ``gender=F``).
        Compound patterns (L > 1) are comma-separated combinations of the
        above, e.g. ``age=[35,50], gender=F``.

        Returns
        -------
        list of str, one entry per mined pattern.
        """
        check_is_fitted(self)
        names = []
        for _, items, _ in self.patterns_:
            parts = [self.td_.item_map.get(it, str(it)) for it in items]
            names.append(', '.join(parts))
        return names

    def get_transformed_shape(self) -> tuple:
        """
        Return the shape (n_samples, n_patterns) of the training pattern matrix.
        """
        check_is_fitted(self)
        return self.x_train_hup_.shape

    def get_pattern_info(self) -> pd.DataFrame:
        """
        Return a summary DataFrame with one row per mined HUG pattern.

        Columns
        -------
        pattern          : str   — human-readable pattern label.
        utility          : float — total utility of the pattern across all
                           training transactions in which it appears.
        information_gain : float — discriminative IG of the pattern relative
                           to its parent in the search tree.
        support          : float — fraction of training samples that match
                           the pattern (pattern prevalence).
        """
        check_is_fitted(self)
        n_train  = self.x_train_hup_.shape[0]
        features = self.get_hug_features()
        records  = []
        for i, (u, _, ul) in enumerate(self.patterns_):
            support = float(self.x_train_hup_[:, i].sum()) / n_train
            records.append({
                'pattern':          features[i],
                'utility':          round(u, 6),
                'information_gain': round(ul.ig, 6),
                'support':          round(support, 4),
            })
        return pd.DataFrame(records)
