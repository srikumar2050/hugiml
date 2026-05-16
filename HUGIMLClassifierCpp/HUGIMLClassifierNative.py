"""
* This file is copyright (c) 2026 Srikumar Krishnamoorthy
*
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.

HUGIMLClassifierNative
======================

A scikit-learn compatible implementation of the HUG-IML classifier backed by a
compiled C++ extension (_hugiml_core).  Computationally intensive stages run at
native speed; the Python layer handles DataFrame ingestion, column-type
detection, and all explanation methods.

Stages executed in C++
----------------------
- Discretisation     (quantile binning, bin-count selection, entropy, NMI, Pearson)
- Transaction construction
- Top-K HUI pattern mining with information-gain filtering
- Binary feature-matrix assembly (train and test)

Stages kept in Python
---------------------
- Column-type detection and DataFrame handling   (prepareXy)
- NaN imputation and categorical splitting
- Downstream sklearn estimator                   (LogisticRegression by default)
- Explanation methods                            (get_hug_features, get_pattern_info)

Installation
------------
Build and install the C++ extension with a single command from the package root:

    pip install -e .

Quick start
-----------
    from HUGIMLClassifierNative import HUGIMLClassifierNative
    from sklearn.model_selection import train_test_split

    clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    X, y = clf.prepareXy(X_df, y_series)          # detects column types
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    print(clf.get_hug_features())                  # e.g. ['age=[35,50]', 'gender=F']
    print(clf.get_pattern_info())                  # utility / IG / support table
"""

import copy
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

try:
    import _hugiml_core as _core
except ImportError as exc:
    raise ImportError(
        "HUGIMLClassifierNative requires the compiled C++ extension '_hugiml_core'.\n"
        "Build it with:  pip install -e .  or  python setup.py build_ext --inplace"
    ) from exc

warnings.filterwarnings("ignore")


# =============================================================================
# HUGIMLClassifierNative
# =============================================================================

class HUGIMLClassifierNative(ClassifierMixin, BaseEstimator):
    """
    HUG-IML interpretable classifier — C++ accelerated, scikit-learn compatible.

    Extracts High Utility Gain (HUG) patterns from labelled tabular data,
    transforms the input into a binary pattern-presence matrix, and fits an
    interpretable downstream classifier (logistic regression by default).
    The mined patterns are human-readable and serve as the primary source of
    model explanations.

    All computationally intensive stages (discretisation, transaction
    construction, HUI mining, and matrix assembly) execute inside a compiled
    C++ extension (_hugiml_core) for native-speed performance.

    Two usage paths are supported:

    **Path A — prepareXy** (recommended when the full dataset is available upfront):

        clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
        X, y = clf.prepareXy(X_df, y_series)          # detects column types
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        print(clf.get_hug_features())                  # e.g. ['age=[35,50]', 'gender=F']
        print(clf.get_pattern_info())                  # utility / IG / support table

    **Path B — allCols + origColumns** (use inside cross-validation loops):

        clf = HUGIMLClassifierNative(
            allCols=[int_col_names, float_col_names, cat_col_names],
            origColumns=X_df.columns.tolist(),
            B=7, L=1, G=5e-3,
        )
        clf.fit(X_train, y_train)

    Column type handling
    --------------------
    Integer columns  : binned on raw integer values (no MinMax scaling).
                       Pattern labels show integer bin edges, e.g. ``age=[35,50]``.
    Float columns    : MinMax-scaled to [0, 1] before binning.
                       Pattern labels show original-scale edges, e.g. ``income=[40000.000,65000.000]``.
    Categorical cols : one item per unique value (one-hot treatment).
                       Pattern labels show the category value, e.g. ``gender=F``.

    Parameters
    ----------
    allCols : list of 3 lists, optional
        ``[int_col_names, float_col_names, cat_col_names]``.
        Must be paired with ``origColumns``.
    origColumns : list of str, optional
        Ordered column names matching the columns of X passed to fit/predict.
        Must be paired with ``allCols``.
    B : int, default 5
        Number of quantile bins per numerical feature.  Use -1 for supervised
        auto-selection (maximises per-column information gain over [2, 20]).
    L : int, default 1
        Maximum HUG pattern length.  1 = singletons only; 2 = pairs and
        singletons; -1 = unlimited.
    G : float, default 1e-4
        Minimum information-gain threshold.  Patterns below this are discarded.
    topK : int, default -1
        Maximum number of patterns to retain.  -1 computes automatically as C(100, L).
    base_estimator : sklearn estimator, optional
        Downstream classifier trained on the binary HUG pattern matrix.
        Defaults to LogisticRegression (liblinear for binary, lbfgs for multi-class).
    verbose : bool, default False
        Print progress messages during fit.

    Attributes (available after fit)
    ---------------------------------
    classes_          : np.ndarray      — unique class labels seen during fit
    n_features_in_    : int             — number of input features
    feature_names_in_ : list or None    — resolved column names
    cat_cols_mask_    : np.ndarray[bool]— True for categorical columns
    is_int_mask_      : np.ndarray[bool]— True for integer columns
    td_               : TransactionDataCpp — training-time discretisation artefacts
    patterns_         : list of PatternEntry — mined HUG patterns
    x_train_hup_      : csr_matrix      — binary pattern matrix for training data
    model_            : Pipeline        — fitted downstream estimator
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

    # ── sklearn parameter protocol ────────────────────────────────────────────

    def get_params(self, deep=True):
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
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ── Data preparation (identical to HUGIMLClassifierPy) ───────────────────

    def prepareXy(self, X, y):
        """
        Detect column types and encode the target variable.

        Call on the full dataset before any train/test split.
        Stores integer / categorical masks on self so fit() uses them.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series or array-like

        Returns
        -------
        X : pd.DataFrame  (unchanged, column order preserved)
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
    def _to_float_array(arr, cat_mask=None):
        """
        Split input into a float64 numeric array and raw categorical value arrays.

        Returns
        -------
        X_num     : float64 ndarray (n, p)  — NaN/Inf replaced by column median.
        X_cat_raw : list of length p        — object array per cat col, None for num.
        """
        is_df  = isinstance(arr, pd.DataFrame)
        n      = len(arr)
        p      = len(arr.columns) if is_df else np.asarray(arr).shape[1]
        arr_np = None if is_df else np.asarray(arr)

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

    def _effective_topK(self):
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

    def _make_estimator(self, n_cls):
        if self.base_estimator is not None:
            return copy.deepcopy(self.base_estimator)
        solver = 'liblinear' if n_cls == 2 else 'lbfgs'
        return LogisticRegression(solver=solver, random_state=0, max_iter=500)

    def _validate_params(self):
        if not isinstance(self.B, int):
            raise TypeError(f'B must be an int, got {type(self.B)}')
        if not isinstance(self.L, int):
            raise TypeError(f'L must be an int, got {type(self.L)}')
        if not isinstance(self.G, float):
            raise TypeError(f'G must be a float, got {type(self.G)}')
        if self.allCols is not None or self.origColumns is not None:
            if self.allCols is None or self.origColumns is None:
                raise ValueError('allCols and origColumns must both be supplied together.')
            if not (isinstance(self.allCols, list) and len(self.allCols) == 3):
                raise ValueError('allCols must be a list of 3 lists: [int_cols, float_cols, cat_cols].')

    def _resolve_col_meta(self, X_train):
        """Determine column names and type masks from whichever path was used."""
        # Path 1: prepareXy already set everything
        if hasattr(self, 'cat_cols_mask_'):
            return self.cat_cols_mask_

        # Path 2: allCols + origColumns in constructor
        if self.allCols is not None and self.origColumns is not None:
            cat_set  = set(self.allCols[2])
            int_set  = set(self.allCols[0])
            col_list = list(self.origColumns)
            self.cat_cols_mask_    = np.array([c in cat_set for c in col_list], dtype=bool)
            self.is_int_mask_      = np.array([c in int_set for c in col_list], dtype=bool)
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

        # Path 4: plain ndarray — all float
        p = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        self.cat_cols_mask_ = np.zeros(p, dtype=bool)
        self.is_int_mask_   = np.zeros(p, dtype=bool)
        if not hasattr(self, 'feature_names_in_'):
            self.feature_names_in_ = None
        return self.cat_cols_mask_

    # ── Core fit / predict methods (delegate to C++) ──────────────────────────

    def fit(self, X_train, y_train):
        """
        Fit the HUG-IML model on training data.

        Stages 1–4 of the HUG-IML workflow, with Stages 2–4 executed in C++:
          1. Resolve column metadata.
          2. Build utility-annotated transactions  (C++)
          3. Mine top-K HUG patterns              (C++)
          4. Build binary pattern matrix           (C++)
          5. Fit downstream sklearn classifier.

        Parameters
        ----------
        X_train : pd.DataFrame or ndarray, shape (n_samples, n_features)
        y_train : array-like of int, shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_params()

        cat_mask = self._resolve_col_meta(X_train)
        int_mask = getattr(self, 'is_int_mask_', None)

        X_num, X_cat_raw = self._to_float_array(X_train, cat_mask)
        y_train = np.asarray(y_train, dtype=np.int64)
        X_num, y_train = _check_X_y(X_num, y_train, dtype=None)

        self.n_features_in_ = X_num.shape[1]
        self.classes_       = np.unique(y_train)
        n_cls               = len(self.classes_)

        if self.verbose:
            print('HUGIMLClassifierNative — preparing transactions (C++) …')

        # ── Stage 2 : prepare transactions (C++) ─────────────────────────────
        col_names = getattr(self, 'feature_names_in_', None)
        is_cat_np = cat_mask.astype(np.uint8)
        is_int_np = (int_mask if int_mask is not None
                     else np.zeros(X_num.shape[1], dtype=bool)).astype(np.uint8)

        # X_cat_raw is a Python list; pass as-is — C++ extracts strings
        self.td_ = _core.prepare_transactions(
            X_num,
            y_train,
            self.B,
            col_names,
            is_cat_np,
            is_int_np,
            X_cat_raw if any(v is not None for v in X_cat_raw) else None,
        )

        K = self._effective_topK()
        if self.verbose:
            print(f'  items={len(self.td_.item_twu)}, K={K}')
            print('  Mining HUG patterns (C++) …')

        # ── Stage 3 : mine patterns (C++) ────────────────────────────────────
        raw_patterns = _core.mine_patterns(
            self.td_, y_train, n_cls, K, self.L, self.G
        )
        # Sort by (utility asc, items asc) — deterministic column ordering
        # regardless of heap-internal ordering, which is implementation-defined.
        # This matches Python heapq tuple-comparison semantics and ensures
        # bit-identical predictions between HUGIMLClassifierPy and Native.
        self.patterns_ = sorted(raw_patterns, key=lambda pe: (pe.utility, pe.items))

        if len(self.patterns_) == 0:
            raise RuntimeError(
                'No HUG patterns found. '
                'Try reducing G, increasing topK, or adjusting B / L.'
            )

        if self.verbose:
            print(f'  {len(self.patterns_)} patterns found')

        # ── Stage 4 : binary pattern matrix (C++) ────────────────────────────
        n_train  = len(y_train)
        n_pats   = len(self.patterns_)
        rows, cols = _core.build_train_matrix(self.td_, self.patterns_)
        data = np.ones(len(rows), dtype=np.float32)
        self.x_train_hup_ = csr_matrix(
            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
        )

        # ── Stage 5 : fit downstream classifier ──────────────────────────────
        self.model_ = Pipeline([('clf', self._make_estimator(n_cls))])
        self.model_.fit(self.x_train_hup_, y_train)

        return self

    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.

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

        n      = X_num.shape[0]
        n_pats = len(self.patterns_)
        rows, cols = _core.build_test_matrix(
            X_num,
            self.td_,
            X_cat_raw if any(v is not None for v in X_cat_raw) else None,
            self.patterns_,
        )
        data       = np.ones(len(rows), dtype=np.float32)
        x_test_hup = csr_matrix(
            (data, (rows, cols)), shape=(n, n_pats), dtype=np.float32
        )
        return self.model_.predict_proba(x_test_hup)

    def predict(self, X_test):
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

        n      = X_num.shape[0]
        n_pats = len(self.patterns_)
        rows, cols = _core.build_test_matrix(
            X_num,
            self.td_,
            X_cat_raw if any(v is not None for v in X_cat_raw) else None,
            self.patterns_,
        )
        data       = np.ones(len(rows), dtype=np.float32)
        x_test_hup = csr_matrix(
            (data, (rows, cols)), shape=(n, n_pats), dtype=np.float32
        )
        return self.model_.predict(x_test_hup)

    def transform(self, X):
        """
        Return the binary HUG pattern matrix for X without making predictions.

        Each column corresponds to one mined HUG pattern; entry (i, j) is 1 if
        all items of pattern j are present in sample i.

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

        n      = X_num.shape[0]
        n_pats = len(self.patterns_)
        rows, cols = _core.build_test_matrix(
            X_num,
            self.td_,
            X_cat_raw if any(v is not None for v in X_cat_raw) else None,
            self.patterns_,
        )
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix(
            (data, (rows, cols)), shape=(n, n_pats), dtype=np.float32
        )

    # ── Explanation / inspection methods ─────────────────────────────────────

    def get_hug_features(self):
        """
        Return a human-readable label for each mined HUG pattern.

        Pattern labels use the same format as HUGIMLClassifierPy:
        ``feature=[lo,hi]`` for numerical columns, ``feature=value`` for
        categorical columns; compound patterns are comma-separated.

        Returns
        -------
        list of str, one entry per mined pattern.
        """
        check_is_fitted(self)
        item_map = self.td_.item_map   # py::dict from C++
        names    = []
        for pe in self.patterns_:
            parts = [item_map.get(it, str(it)) for it in pe.items]
            names.append(', '.join(parts))
        return names

    def get_transformed_shape(self):
        """Return (n_samples, n_patterns) of the training pattern matrix."""
        check_is_fitted(self)
        return self.x_train_hup_.shape

    def get_pattern_info(self):
        """
        Return a summary DataFrame with one row per mined HUG pattern.

        Columns
        -------
        pattern          : str   — human-readable label.
        utility          : float — total utility in training transactions.
        information_gain : float — discriminative IG relative to parent.
        support          : float — fraction of training samples matching the pattern.

        Returns
        -------
        pd.DataFrame
        """
        check_is_fitted(self)
        n_train  = self.x_train_hup_.shape[0]
        features = self.get_hug_features()
        records  = []
        for i, pe in enumerate(self.patterns_):
            support = float(self.x_train_hup_[:, i].sum()) / n_train
            records.append({
                'pattern':          features[i],
                'utility':          round(pe.utility, 6),
                'information_gain': round(pe.ig,      6),
                'support':          round(support,    4),
            })
        return pd.DataFrame(records)
