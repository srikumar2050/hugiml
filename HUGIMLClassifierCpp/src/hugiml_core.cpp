/*
 * hugiml_core.cpp  —  pybind11 C++ extension for HUGIMLClassifierNative
 *
 * Copyright (c) 2026 Srikumar Krishnamoorthy — GPL v3
 *
 * Implements the core HUG-IML computation pipeline in native C++:
 *
 *   Section 1 : Math helpers        normalised entropy, information gain, NMI, Pearson
 *   Section 2 : Discretisation      quantile binning, supervised bin-count selection
 *   Section 3 : TransactionData     C++ container holding all training artefacts
 *   Section 4 : prepare_transactions
 *                 Pass 1 — discretise / encode each column
 *                 Pass 2 — build item registry with utility weights
 *                 Pass 3 — construct utility-annotated transaction list
 *   Section 5 : Utility lists and top-K HUI miner
 *                 UL (utility list), THUIsl (top-K HUI miner with IG filter)
 *   Section 6 : Feature-matrix builders (training and test/validation)
 *   Section 7 : pybind11 module bindings
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// ── Convenience aliases ───────────────────────────────────────────────────────
using TItem     = std::pair<int, double>;   // (item_id, utility)
using Trans     = std::vector<TItem>;
using TransList = std::vector<Trans>;

// =============================================================================
// Section 1 : Math helpers
// =============================================================================

/// Normalised Shannon entropy of integer class labels (maps to _entropy).
static double entropy_vec(const std::vector<int>& y, int n_cls)
{
    if (y.empty()) return 0.0;
    // Array sized to max(y)+1, not n_cls, so non-zero-based labels work
    // (matches Python np.bincount which auto-sizes to max(y)+1)
    int max_y = 0;
    for (int v : y) if (v > max_y) max_y = v;
    std::vector<double> cnts(max_y + 1, 0.0);
    for (int v : y) if (v >= 0) cnts[v] += 1.0;
    double total = static_cast<double>(y.size());
    double base  = std::log(std::max(n_cls, 2));
    double h = 0.0;
    for (double p : cnts) {
        if (p > 0.0) { double pn = p / total; h -= pn * std::log(pn) / base; }
    }
    return std::max(h, 0.0);
}

/// Information gain of a discretised column w.r.t. class labels (_ig_col).
static double ig_col_cpp(const std::vector<int>& x_disc,
                          const std::vector<int>& y, int n_cls)
{
    double base  = entropy_vec(y, n_cls);
    double total = static_cast<double>(y.size());
    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(n_cls * 2);
    for (size_t i = 0; i < x_disc.size(); i++) groups[x_disc[i]].push_back(y[i]);
    double ce = 0.0;
    for (auto& kv : groups)
        ce += static_cast<double>(kv.second.size()) / total * entropy_vec(kv.second, n_cls);
    return base - ce;
}

/// Pearson correlation coefficient (_pearson).
static double pearson_cpp(const std::vector<double>& x, const std::vector<double>& y)
{
    size_t n = x.size();
    if (n == 0) return 0.0;
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sx = 0.0, sy = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        sx += dx * dx; sy += dy * dy; sxy += dx * dy;
    }
    if (sx == 0.0 || sy == 0.0) return 0.0;
    double r = sxy / std::sqrt(sx * sy);
    return std::isfinite(r) ? r : 0.0;
}

/// Normalised Mutual Information between a binarised (0/1) column and y (_nmi_binary).
static double nmi_binary_cpp(const std::vector<int>& x_bin,
                              const std::vector<int>& y, int n_cls)
{
    size_t n = x_bin.size();
    std::unordered_map<int,int> x_counts;
    for (int v : x_bin) x_counts[v]++;
    if (x_counts.size() <= 1) return 0.0;

    double hx = 0.0;
    for (auto& kv : x_counts) {
        double p = static_cast<double>(kv.second) / n;
        if (p > 0.0) hx -= p * std::log(p);
    }
    // Use max(y)+1 for count arrays (handles non-zero-based labels)
    int max_y = 0;
    for (int v : y) if (v > max_y) max_y = v;
    int arr_sz = max_y + 1;
    std::vector<double> yc(arr_sz, 0.0);
    for (int v : y) if (v >= 0) yc[v] += 1.0;
    double hy = 0.0;
    for (double c : yc) { double p = c / n; if (p > 0.0) hy -= p * std::log(p); }
    if (hx == 0.0 || hy == 0.0) return 0.0;

    double mi = 0.0;
    for (auto& kv : x_counts) {
        int xv = kv.first; int nx_v = kv.second;
        std::vector<double> xyc(arr_sz, 0.0);
        for (size_t i = 0; i < n; i++) if (x_bin[i] == xv && y[i] >= 0) xyc[y[i]] += 1.0;
        for (int c = 0; c < arr_sz; c++) {
            double nxy = xyc[c];
            if (nxy > 0.0) {
                double pxy = nxy / n;
                double px  = static_cast<double>(nx_v) / n;
                double py_c = yc[c] / n;
                mi += pxy * std::log(pxy / (px * py_c));
            }
        }
    }
    return std::max(0.0, std::min(1.0, mi / std::sqrt(hx * hy)));
}


// =============================================================================
// Section 2 : Discretisation helpers
// =============================================================================

/// Quantile-based discretisation into nb bins (_kbins).
/// Returns (binned 0-based indices, sorted unique edges).
static std::pair<std::vector<int>, std::vector<double>>
kbins_cpp(const std::vector<double>& col, int nb)
{
    size_t n = col.size();
    std::vector<double> sc = col;
    std::sort(sc.begin(), sc.end());

    // Compute nb+1 percentile points (numpy-compatible linear interpolation)
    // Must replicate NumPy's FP rounding path exactly:
    //   np.linspace(0,100,nb+1) computes step = 100.0/nb, then q = step*i
    //   np.percentile converts q to index: fidx = q / 100.0 * (n-1)
    // Using (double)i / nb * (n-1) gives a different FP result due to
    // different intermediate rounding, causing 1-ULP edge differences
    // that mis-classify boundary data points.
    double step = 100.0 / nb;
    std::vector<double> edges;
    edges.reserve(nb + 1);
    for (int i = 0; i <= nb; i++) {
        double q    = step * static_cast<double>(i);
        double fidx = q / 100.0 * (n - 1);
        size_t lo   = static_cast<size_t>(fidx);
        size_t hi   = std::min(lo + 1, n - 1);
        double frac = fidx - lo;
        edges.push_back(sc[lo] + frac * (sc[hi] - sc[lo]));
    }

    // Collapse duplicate edges (mirrors np.unique)
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) {
        double lo = sc.front();
        edges = {lo, lo + 1e-9};
    }

    // Bin using upper_bound on edges[1:-1]  (matches np.searchsorted side='right')
    std::vector<double> inner(edges.begin() + 1, edges.end() - 1);
    std::vector<int> binned(n);
    for (size_t i = 0; i < n; i++) {
        auto it   = std::upper_bound(inner.begin(), inner.end(), col[i]);
        binned[i] = static_cast<int>(it - inner.begin());
    }
    return {binned, edges};
}

/// Select the number of bins for one column (_choose_nb).
static int choose_nb_cpp(const std::vector<double>& col,
                          const std::vector<int>&    y,
                          int n_cls, int B, int distinct)
{
    if (B == -1) {
        double best_ig = 0.0; int best_nb = 2;
        for (int nb = 2; nb <= 20; nb++) {
            int nb_cap = std::max(std::min(distinct - 1, nb), 2);
            auto [dv, _e] = kbins_cpp(col, nb_cap);
            double ig = ig_col_cpp(dv, y, n_cls);
            if (ig > best_ig) { best_ig = ig; best_nb = nb_cap; }
        }
        return best_nb;
    }
    return std::max(std::min(distinct - 1, B), 2);
}


// =============================================================================
// Section 3 : TransactionData C++ class  (wraps _TransactionData)
// =============================================================================

struct TransactionDataCpp {
    TransList                                      transactions;
    std::vector<double>                            item_twu;   // 0-indexed (item_id - 1)
    std::unordered_map<int,std::string>            item_map;   // 1-based item_id → label
    std::vector<double>                            RIU;        // 0-indexed

    // disc_mat[r][j] = 0-based bin index (or -1 for missing categorical)
    std::vector<std::vector<int32_t>>              disc_mat;
    int disc_n = 0, disc_p = 0;

    std::unordered_map<int,int>                    bn2id;       // bkey → 1-based item_id
    std::unordered_set<int>                        colnew_set;

    std::vector<int>                               nb_col;
    std::vector<std::vector<double>>               ber;        // normalised right edges per col
    std::vector<double>                            cv;         // corr per col
    std::vector<std::vector<double>>               all_edges;  // raw edges per col
    std::vector<double>                            col_min, col_range;
    std::vector<bool>                              is_cat_v, is_int_v;

    // Categorical-specific
    std::vector<std::vector<std::string>>          cat_categories; // one per col
    std::vector<std::unordered_map<std::string,double>> cat_corr;

    // Python-facing accessor: returns item_map as py::dict
    py::dict get_item_map_py() const {
        py::dict d;
        for (auto& kv : item_map)
            d[py::int_(kv.first)] = py::str(kv.second);
        return d;
    }

    double riu_thresh(int k) const {
        if (k <= 0 || RIU.empty()) return 0.0;
        std::vector<double> s = RIU;
        std::sort(s.rbegin(), s.rend());
        return s[std::min(static_cast<size_t>(k - 1), s.size() - 1)];
    }

    static int bkey_fn(int bi, int j) { return bi * 10000 + j; }
};


// =============================================================================
// Section 4 : prepare_transactions_cpp  (maps _prepare_transactions)
// =============================================================================

/// Build utility-annotated transactions from labelled training data.
/// Mirrors _prepare_transactions faithfully including the three-pass structure.
TransactionDataCpp prepare_transactions_cpp(
    py::array_t<double,  py::array::c_style | py::array::forcecast> X_num_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_arr,
    int B,
    py::object col_names_py,    // None  or  list of str
    py::array_t<uint8_t, py::array::forcecast> is_cat_arr,
    py::array_t<uint8_t, py::array::forcecast> is_int_arr,
    py::object X_cat_raw_py)    // None  or  list of (None | np.ndarray[object])
{
    auto Xb  = X_num_arr.unchecked<2>();
    auto yb  = y_arr.unchecked<1>();
    auto icb = is_cat_arr.unchecked<1>();
    auto iib = is_int_arr.unchecked<1>();

    int n = static_cast<int>(Xb.shape(0));
    int p = static_cast<int>(Xb.shape(1));

    // y as std::vector<int>
    std::vector<int> y_vec(n);
    int max_label = 0;
    std::set<int> cls_set;
    for (int i = 0; i < n; i++) {
        y_vec[i] = static_cast<int>(yb(i));
        if (y_vec[i] > max_label) max_label = y_vec[i];
        cls_set.insert(y_vec[i]);
    }
    // n_cls = number of unique classes (matches Python: len(np.unique(y)))
    // key_stride = max(y)+1 (safe stride for tu key encoding: bname*stride+yi)
    int n_cls     = static_cast<int>(cls_set.size());
    int key_stride = max_label + 1;

    // Column names
    std::vector<std::string> names(p);
    if (!col_names_py.is_none()) {
        py::list lst = col_names_py.cast<py::list>();
        if (static_cast<int>(lst.size()) == p) {
            for (int j = 0; j < p; j++) names[j] = lst[j].cast<std::string>();
        } else {
            for (int j = 0; j < p; j++) names[j] = "col" + std::to_string(j);
        }
    } else {
        for (int j = 0; j < p; j++) names[j] = "col" + std::to_string(j);
    }

    // is_cat / is_int masks
    std::vector<bool> is_cat(p), is_int(p);
    for (int j = 0; j < p; j++) {
        is_cat[j] = static_cast<bool>(icb(j));
        is_int[j] = static_cast<bool>(iib(j));
    }

    // ── Extract categorical raw strings from X_cat_raw_py ────────────────────
    // cat_raw_str[j][r] and cat_valid[j][r]
    std::vector<std::vector<std::string>> cat_raw_str(p);
    std::vector<std::vector<bool>>        cat_valid(p);

    auto extract_cat_col = [&](int j, py::object col_obj) {
        cat_raw_str[j].resize(n);
        cat_valid[j].resize(n, false);
        if (col_obj.is_none()) return;
        py::array arr = col_obj.cast<py::array>();
        py::list lst  = arr.attr("tolist")().cast<py::list>();
        for (int r = 0; r < n; r++) {
            py::object val = lst[r];
            if (val.is_none()) continue;
            try {
                double dv = val.cast<double>();
                if (std::isnan(dv)) continue;
            } catch (...) {}
            cat_valid[j][r]   = true;
            cat_raw_str[j][r] = py::str(val).cast<std::string>();
        }
    };

    if (!X_cat_raw_py.is_none()) {
        py::list raw_list = X_cat_raw_py.cast<py::list>();
        for (int j = 0; j < p; j++) {
            if (is_cat[j]) extract_cat_col(j, raw_list[j].cast<py::object>());
        }
    }

    // ── MinMax params for float columns ──────────────────────────────────────
    std::vector<double> col_min(p, 0.0), col_range(p, 1.0);
    for (int j = 0; j < p; j++) {
        if (is_cat[j] || is_int[j]) continue;
        double cmin = std::numeric_limits<double>::max();
        double cmax = std::numeric_limits<double>::lowest();
        for (int r = 0; r < n; r++) {
            double v = Xb(r, j);
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
        }
        col_min[j]   = cmin;
        col_range[j] = (cmax > cmin) ? (cmax - cmin) : 1.0;
    }

    // ── Pass 1 : Discretise / encode each column ──────────────────────────────
    std::vector<std::vector<int32_t>>               disc_mat(n, std::vector<int32_t>(p, 0));
    std::vector<int>                                nb_col;
    std::vector<std::vector<double>>                ber, all_edges;
    std::vector<double>                             cv;
    std::vector<std::vector<std::string>>           cat_categories(p);
    std::vector<std::unordered_map<std::string,double>> cat_corr(p);
    std::vector<std::vector<std::pair<double,double>>> bro(p); // bin-range labels

    for (int j = 0; j < p; j++) {
        if (is_cat[j]) {
            // ── Categorical ──────────────────────────────────────────────────
            // Collect unique valid string values, sort alphabetically
            std::set<std::string> uniq_set;
            for (int r = 0; r < n; r++)
                if (cat_valid[j][r]) uniq_set.insert(cat_raw_str[j][r]);
            std::vector<std::string> uniq(uniq_set.begin(), uniq_set.end());
            cat_categories[j] = uniq;

            std::unordered_map<std::string,int> label2int;
            for (int i = 0; i < static_cast<int>(uniq.size()); i++)
                label2int[uniq[i]] = i;

            for (int r = 0; r < n; r++) {
                if (!cat_valid[j][r]) { disc_mat[r][j] = -1; continue; }
                auto it = label2int.find(cat_raw_str[j][r]);
                disc_mat[r][j] = (it != label2int.end()) ? it->second : -1;
            }

            // Point-biserial sign per label for IU
            std::unordered_map<std::string,double> pb_sign;
            for (auto& v : uniq) {
                int code = label2int.at(v);
                std::vector<double> x_bin_d(n, 0.0), y_d(n);
                for (int r = 0; r < n; r++) {
                    x_bin_d[r] = (disc_mat[r][j] == code) ? 1.0 : 0.0;
                    y_d[r]     = static_cast<double>(y_vec[r]);
                }
                pb_sign[v] = pearson_cpp(x_bin_d, y_d);
            }
            cat_corr[j] = std::move(pb_sign);

            nb_col.push_back(static_cast<int>(uniq.size()));
            all_edges.push_back({0.0, 1.0});
            ber.push_back({1.0});
            bro[j] = {{0.0, 1.0}};
            cv.push_back(0.0);

        } else if (is_int[j]) {
            // ── Integer : bin on raw values, no MinMax scaling ───────────────
            std::vector<double> col_raw(n);
            for (int r = 0; r < n; r++) col_raw[r] = Xb(r, j);

            std::set<double> dset(col_raw.begin(), col_raw.end());
            int distinct = static_cast<int>(dset.size());
            int nb       = choose_nb_cpp(col_raw, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_raw, nb);
            int nb_act = static_cast<int>(edges.size()) - 1;

            for (int r = 0; r < n; r++) disc_mat[r][j] = static_cast<int32_t>(binned[r]);
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            ber.push_back(ber_j);

            std::vector<std::pair<double,double>> bro_j;
            for (int bi = 0; bi < nb_act; bi++)
                bro_j.push_back({edges[bi], edges[bi + 1]});
            bro[j] = bro_j;

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            cv.push_back(pearson_cpp(df, yf));

        } else {
            // ── Float : MinMax-scale then bin ────────────────────────────────
            std::vector<double> col_sc(n);
            for (int r = 0; r < n; r++)
                col_sc[r] = (Xb(r, j) - col_min[j]) / col_range[j];

            std::set<double> dset(col_sc.begin(), col_sc.end());
            int distinct = static_cast<int>(dset.size());
            int nb       = choose_nb_cpp(col_sc, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_sc, nb);
            int nb_act = static_cast<int>(edges.size()) - 1;

            for (int r = 0; r < n; r++) disc_mat[r][j] = static_cast<int32_t>(binned[r]);
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);   // edges in scaled [0,1] space

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            ber.push_back(ber_j);

            // Inverse-transform edges to original scale for labels
            std::vector<std::pair<double,double>> bro_j;
            for (int bi = 0; bi < nb_act; bi++) {
                double lo_o = edges[bi]     * col_range[j] + col_min[j];
                double hi_o = edges[bi + 1] * col_range[j] + col_min[j];
                bro_j.push_back({lo_o, hi_o});
            }
            bro[j] = bro_j;

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            cv.push_back(pearson_cpp(df, yf));
        }
    }

    // ── Pass 2 : Build item registry (EU × IU per bin/category) ──────────────
    std::unordered_map<int,std::string> item_map;
    std::unordered_map<int,int>         bn2id;
    std::unordered_set<int>             colnew_set;

    // tu[bname * key_stride + yi] = eiu;  tu_y[yi] = per-class max
    std::unordered_map<int64_t,double>  tu;
    std::vector<double>                 tu_y(key_stride, 0.0);
    int ic = 0;  // 1-based item counter

    for (int j = 0; j < p; j++) {
        int nb = nb_col[j];
        if (is_cat[j]) {
            for (int idx = 0; idx < static_cast<int>(cat_categories[j].size()); idx++) {
                int bi = idx + 1;
                // x_bin: 1 if disc[r][j] == idx
                std::vector<int> x_bin(n, 0);
                for (int r = 0; r < n; r++)
                    if (disc_mat[r][j] == idx) x_bin[r] = 1;

                double nmi  = nmi_binary_cpp(x_bin, y_vec, n_cls);
                const std::string& v = cat_categories[j][idx];
                double pb   = 0.0;
                auto it = cat_corr[j].find(v);
                if (it != cat_corr[j].end()) pb = it->second;
                double iu_t = (pb > 0.0) ? 1.0 : 0.05;
                double eiu  = nmi * iu_t;

                int bname = TransactionDataCpp::bkey_fn(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    item_map[ic] = names[j] + "=" + v;
                    colnew_set.insert(bname);
                    bn2id[bname] = ic;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }
        } else {
            double eu = std::abs(cv[j]);
            for (int bi = 1; bi <= nb; bi++) {
                // iu_t: normalised right edge; reversed if corr < 0
                double iu_t = (cv[j] >= 0.0) ? ber[j][bi - 1] : ber[j][nb - bi];
                double eiu  = eu * iu_t;

                int bname = TransactionDataCpp::bkey_fn(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    auto& [lo, hi] = bro[j][bi - 1];
                    std::ostringstream oss;
                    if (is_int[j]) {
                        oss << names[j] << "=[" << static_cast<int>(lo)
                            << "," << static_cast<int>(hi) << "]";
                    } else {
                        oss << names[j] << "=["
                            << std::fixed << std::setprecision(3) << lo
                            << ","
                            << std::fixed << std::setprecision(3) << hi << "]";
                    }
                    item_map[ic] = oss.str();
                    colnew_set.insert(bname);
                    bn2id[bname] = ic;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }
        }
    }

    // Normalise utility per class
    for (auto& kv : tu) {
        int yi = static_cast<int>(kv.first % key_stride);
        kv.second = (tu_y[yi] > 0.0) ? kv.second / tu_y[yi] : 0.0;
    }

    // ── Pass 3 : Build transaction list ──────────────────────────────────────
    TransList              transactions;
    std::vector<double>    item_twu(ic, 0.0);
    std::vector<double>    RIU(ic, 0.0);
    transactions.reserve(n);

    for (int r = 0; r < n; r++) {
        int yi = y_vec[r];
        double tutils = 0.0;
        Trans  trans;

        for (int j = 0; j < p; j++) {
            int bi;
            if (is_cat[j]) {
                int code = static_cast<int>(disc_mat[r][j]);
                if (code < 0) continue;
                bi = code + 1;
            } else {
                bi = static_cast<int>(disc_mat[r][j]) + 1;
            }

            int bname = TransactionDataCpp::bkey_fn(bi, j);
            if (colnew_set.find(bname) == colnew_set.end()) continue;

            int64_t txk = static_cast<int64_t>(bname) * key_stride + yi;
            auto tit = tu.find(txk);
            if (tit == tu.end()) continue;

            // Round to 6 d.p. (matches Python round(tu[txk], 6))
            double iu = std::round(tit->second * 1e6) / 1e6;
            int iid   = bn2id.at(bname);
            trans.push_back({iid, iu});
            tutils += iu;
        }

        if (tutils > 0.0) {
            for (auto& [iid, iu] : trans) {
                item_twu[iid - 1] += tutils;
                RIU[iid - 1]      += iu;
            }
            transactions.push_back(std::move(trans));
        } else {
            transactions.push_back({{-1, 0.0}});
        }
    }

    // ── Assemble result ───────────────────────────────────────────────────────
    TransactionDataCpp td;
    td.transactions    = std::move(transactions);
    td.item_twu        = std::move(item_twu);
    td.item_map        = std::move(item_map);
    td.RIU             = std::move(RIU);
    td.disc_mat        = std::move(disc_mat);
    td.disc_n          = n;  td.disc_p = p;
    td.bn2id           = std::move(bn2id);
    td.colnew_set      = std::move(colnew_set);
    td.nb_col          = std::move(nb_col);
    td.ber             = std::move(ber);
    td.cv              = std::move(cv);
    td.all_edges       = std::move(all_edges);
    td.col_min         = std::move(col_min);
    td.col_range       = std::move(col_range);
    td.is_cat_v        = std::move(is_cat);
    td.is_int_v        = std::move(is_int);
    td.cat_categories  = std::move(cat_categories);
    td.cat_corr        = std::move(cat_corr);
    return td;
}


// =============================================================================
// Section 5 : Utility lists and miner  (_El, _UL, _THUIsl)
// =============================================================================

struct El { int tid; double iu, ru; };

struct UL {
    int    item = 0;
    double sI = 0.0, sR = 0.0, ig = 0.0;
    std::vector<El>      els;
    std::vector<int32_t> tid_arr;
    std::vector<double>  iu_arr, ru_arr;
    bool sealed = false;

    explicit UL(int it) : item(it) {}

    void add(int t, double i, double r) {
        sI += i; sR += r;
        els.push_back({t, i, r});
    }

    void seal() {
        if (sealed) return;
        size_t m = els.size();
        tid_arr.resize(m); iu_arr.resize(m); ru_arr.resize(m);
        for (size_t k = 0; k < m; k++) {
            tid_arr[k] = els[k].tid;
            iu_arr[k]  = els[k].iu;
            ru_arr[k]  = els[k].ru;
        }
        sealed = true;
    }

    void seal_from_arrays(std::vector<int32_t>&& ta,
                          std::vector<double>&&  ia,
                          std::vector<double>&&  ra)
    {
        tid_arr = std::move(ta); iu_arr = std::move(ia); ru_arr = std::move(ra);
        sI = 0.0; sR = 0.0;
        for (double v : iu_arr) sI += v;
        for (double v : ru_arr) sR += v;
        // Rebuild els for compute_ig
        els.clear();
        els.reserve(tid_arr.size());
        for (size_t k = 0; k < tid_arr.size(); k++)
            els.push_back({static_cast<int>(tid_arr[k]), iu_arr[k], ru_arr[k]});
        sealed = true;
    }

    /// Compute IG relative to parent (or full set when parent==nullptr).
    void compute_ig(const UL* parent, const std::vector<int>& y_arr, int n_cls) {
        if (els.empty()) { ig = 0.0; return; }

        std::vector<int> y_in;
        y_in.reserve(els.size());
        for (auto& e : els) y_in.push_back(y_arr[e.tid]);

        double base;
        int    n_parent;
        if (parent == nullptr) {
            base     = entropy_vec(y_arr, n_cls);
            n_parent = static_cast<int>(y_arr.size());
        } else {
            std::vector<int> py_v;
            py_v.reserve(parent->els.size());
            for (auto& e : parent->els) py_v.push_back(y_arr[e.tid]);
            base     = entropy_vec(py_v, n_cls);
            n_parent = static_cast<int>(parent->els.size());
        }
        if (n_parent == 0) { ig = 0.0; return; }

        // y_out: elements in parent (or all) NOT in this UL
        std::unordered_set<int> tid_set;
        tid_set.reserve(els.size());
        for (auto& e : els) tid_set.insert(e.tid);

        std::vector<int> y_out;
        if (parent == nullptr) {
            for (int i = 0; i < static_cast<int>(y_arr.size()); i++)
                if (tid_set.find(i) == tid_set.end()) y_out.push_back(y_arr[i]);
        } else {
            for (auto& e : parent->els)
                if (tid_set.find(e.tid) == tid_set.end()) y_out.push_back(y_arr[e.tid]);
        }

        double ce = (static_cast<double>(y_in.size())  / n_parent * entropy_vec(y_in,  n_cls) +
                     static_cast<double>(y_out.size()) / n_parent * entropy_vec(y_out, n_cls));
        ig = std::max(base - ce, 0.0);
    }
};

/// One entry on the top-K heap.
struct PatternEntry {
    double           utility;
    std::vector<int> items;
    double           ig;
};

/// Min-heap comparator (smallest utility on top, matching Python heapq).
struct MinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility;
    }
};

/// Top-K HUI miner with IG filtering (_THUIsl).
class THUIsl {
public:
    int    K, L;
    double G, minU = 0.0;
    std::vector<PatternEntry> heap;

    THUIsl(int K, int L, double G) : K(K), L(L), G(G) {}

    void save(const std::vector<int>& items, const UL& ul) {
        double u = ul.sI;
        PatternEntry pe{u, items, ul.ig};
        if (static_cast<int>(heap.size()) < K) {
            heap.push_back(pe);
            std::push_heap(heap.begin(), heap.end(), MinHeapCmp{});
            if (static_cast<int>(heap.size()) == K) minU = heap.front().utility;
        } else if (u > minU) {
            std::pop_heap(heap.begin(), heap.end(), MinHeapCmp{});
            heap.back() = pe;
            std::push_heap(heap.begin(), heap.end(), MinHeapCmp{});
            minU = heap.front().utility;
        }
    }

    /// Build child utility list (_child) — vectorised sorted-merge intersection.
    UL child_ul(const UL& p_ul, const UL& x_ul) {
        UL c(x_ul.item);
        if (p_ul.els.empty() || x_ul.els.empty()) return c;

        const auto& pt = p_ul.tid_arr;
        const auto& xt = x_ul.tid_arr;
        std::vector<int32_t> c_tid; std::vector<double> c_iu, c_ru;
        c_tid.reserve(std::min(pt.size(), xt.size()));
        c_iu.reserve(c_tid.capacity()); c_ru.reserve(c_tid.capacity());

        for (size_t xi = 0; xi < xt.size(); xi++) {
            auto it = std::lower_bound(pt.begin(), pt.end(), xt[xi]);
            if (it == pt.end() || *it != xt[xi]) continue;
            size_t pi = static_cast<size_t>(it - pt.begin());
            c_tid.push_back(xt[xi]);
            c_iu.push_back(p_ul.iu_arr[pi] + x_ul.iu_arr[xi]);
            c_ru.push_back(x_ul.ru_arr[xi]);
        }
        if (c_tid.empty()) return c;
        c.seal_from_arrays(std::move(c_tid), std::move(c_iu), std::move(c_ru));
        return c;
    }

    void mine(const TransList& transactions,
              const std::vector<double>& item_twu,
              const std::vector<int>& ytrain,
              int n_cls)
    {
        minU = 0.0; heap.clear();
        int  n_items  = static_cast<int>(item_twu.size());
        bool use_eucs = (L != 1);

        // EUCS: fmap[item_a][item_b] = {twu_sum, util_sum}
        using FMap = std::unordered_map<int, std::unordered_map<int, std::pair<double,double>>>;
        FMap fmap;

        // Build utility-list map for items that pass the initial threshold
        std::unordered_map<int, UL> ul_map;
        ul_map.reserve(n_items);
        for (int iid = 1; iid <= n_items; iid++) {
            if (item_twu[iid - 1] >= minU)
                ul_map.emplace(iid, UL(iid));
        }

        // Sorted by TWU ascending (same as Python sorted_items)
        std::vector<int> sorted_items;
        sorted_items.reserve(ul_map.size());
        for (auto& kv : ul_map) sorted_items.push_back(kv.first);
        std::sort(sorted_items.begin(), sorted_items.end(),
                  [&](int a, int b){ return item_twu[a-1] < item_twu[b-1]; });

        // Single pass: populate ULs and EUCS map
        for (int tid = 0; tid < static_cast<int>(transactions.size()); tid++) {
            const Trans& trans = transactions[tid];
            if (trans.size() == 1 && trans[0].first == -1) continue;

            std::vector<std::pair<int,double>> active;
            active.reserve(trans.size());
            for (auto& [it, u] : trans)
                if (ul_map.count(it)) active.push_back({it, u});
            if (active.empty()) continue;

            double new_twu = 0.0;
            for (auto& [it, u] : active) new_twu += u;

            std::sort(active.begin(), active.end(),
                      [&](auto& a, auto& b){ return item_twu[a.first-1] < item_twu[b.first-1]; });

            double rem = 0.0;
            for (int i = static_cast<int>(active.size()) - 1; i >= 0; i--) {
                int it = active[i].first; double u = active[i].second;
                ul_map.at(it).add(tid, u, rem);
                if (use_eucs) {
                    auto& fm = fmap[it];
                    for (int j2 = i + 1; j2 < static_cast<int>(active.size()); j2++) {
                        int oj = active[j2].first; double ou = active[j2].second;
                        if (oj != it) { fm[oj].first += new_twu; fm[oj].second += u + ou; }
                    }
                }
                rem += u;
            }
        }

        // Seal all 1-item ULs, then compute IG
        for (auto& kv : ul_map) kv.second.seal();
        for (auto& kv : ul_map) kv.second.compute_ig(nullptr, ytrain, n_cls);

        std::vector<UL*> uls;
        uls.reserve(sorted_items.size());
        for (int iid : sorted_items) uls.push_back(&ul_map.at(iid));

        explore({}, uls, ytrain, n_cls, 0, fmap);
    }

    void explore(std::vector<int>  prefix,
                 std::vector<UL*>& uls,
                 const std::vector<int>& y_arr,
                 int n_cls, int depth,
                 std::unordered_map<int, std::unordered_map<int,std::pair<double,double>>>& fmap)
    {
        int maxd = (L == -1 || L == 0) ? 99 : L;
        for (size_t i = 0; i < uls.size(); i++) {
            UL* ux = uls[i];
            if (ux->sI + ux->sR < minU) continue;

            if (ux->sI >= minU && ux->ig >= G) {
                std::vector<int> pat = prefix;
                pat.push_back(ux->item);
                save(pat, *ux);
            }
            if (depth + 1 >= maxd) continue;

            std::vector<std::unique_ptr<UL>> ext_owned;
            std::vector<UL*>                 ext;
            for (size_t j = i + 1; j < uls.size(); j++) {
                UL* uy = uls[j];
                // EUCS pruning: if pair never reached minU, skip
                if (L != 1) {
                    double eucs_val = 0.0;
                    auto fi = fmap.find(ux->item);
                    if (fi != fmap.end()) {
                        auto fj = fi->second.find(uy->item);
                        if (fj != fi->second.end()) eucs_val = fj->second.first;
                    }
                    if (eucs_val < minU) continue;
                }
                auto ch = std::make_unique<UL>(child_ul(*ux, *uy));
                if (ch->sI + ch->sR >= minU) {
                    ch->compute_ig(ux, y_arr, n_cls);
                    ext.push_back(ch.get());
                    ext_owned.push_back(std::move(ch));
                }
            }
            if (!ext.empty()) {
                std::vector<int> np2 = prefix;
                np2.push_back(ux->item);
                explore(np2, ext, y_arr, n_cls, depth + 1, fmap);
            }
        }
    }
};

/// Entry point: mine patterns from a fitted TransactionDataCpp.
std::vector<PatternEntry> mine_patterns_cpp(
    const TransactionDataCpp& td,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ytrain_arr,
    int n_cls, int K, int L, double G)
{
    auto yb = ytrain_arr.unchecked<1>();
    std::vector<int> ytrain(yb.shape(0));
    for (py::ssize_t i = 0; i < yb.shape(0); i++) ytrain[i] = static_cast<int>(yb(i));

    THUIsl miner(K, L, G);
    miner.mine(td.transactions, td.item_twu, ytrain, n_cls);
    return miner.heap;
}


// =============================================================================
// Section 6 : build_matrix and apply_bins  (_build_matrix, _apply_bins)
// =============================================================================

/// Build COO arrays for a sparse binary pattern-presence matrix.
/// Caller constructs scipy.sparse.csr_matrix from the returned (rows, cols).
py::tuple build_matrix_cpp(const TransList& transactions,
                            const std::vector<PatternEntry>& patterns,
                            int n)
{
    int n_pats = static_cast<int>(patterns.size());

    // Precompute transaction item-sets once (mirrors trans_sets in Python)
    std::vector<std::unordered_set<int>> trans_sets(n);
    for (int i = 0; i < n && i < static_cast<int>(transactions.size()); i++)
        for (auto& [it, _u] : transactions[i])
            if (it != -1) trans_sets[i].insert(it);

    std::vector<int32_t> rows_v, cols_v;
    for (int pi = 0; pi < n_pats; pi++) {
        const auto& items = patterns[pi].items;
        for (int tid = 0; tid < n; tid++) {
            bool match = true;
            for (int it : items) {
                if (trans_sets[tid].find(it) == trans_sets[tid].end()) { match = false; break; }
            }
            if (match) { rows_v.push_back(tid); cols_v.push_back(pi); }
        }
    }

    auto rows_arr = py::array_t<int32_t>(rows_v.size());
    auto cols_arr = py::array_t<int32_t>(cols_v.size());
    auto r = rows_arr.mutable_unchecked<1>();
    auto c = cols_arr.mutable_unchecked<1>();
    for (size_t k = 0; k < rows_v.size(); k++) { r(k) = rows_v[k]; c(k) = cols_v[k]; }
    return py::make_tuple(rows_arr, cols_arr);
}

/// Apply training-fitted bins to test data, then build pattern matrix.
/// Combines _apply_bins + _build_matrix into one C++ call for predict paths.
py::tuple build_test_matrix_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_raw_arr,
    const TransactionDataCpp& td,
    py::object X_cat_raw_py,
    const std::vector<PatternEntry>& patterns)
{
    auto Xb = X_raw_arr.unchecked<2>();
    int n   = static_cast<int>(Xb.shape(0));
    int p   = static_cast<int>(Xb.shape(1));

    // Build label→code maps for categorical cols
    std::vector<std::unordered_map<std::string,int>> label2code(p);
    for (int j = 0; j < p; j++) {
        if (!td.is_cat_v[j]) continue;
        for (int i = 0; i < static_cast<int>(td.cat_categories[j].size()); i++)
            label2code[j][td.cat_categories[j][i]] = i;
    }

    // Extract categorical strings for test rows
    std::vector<std::vector<std::string>> cat_str(p);
    std::vector<std::vector<bool>>        cat_ok(p);

    auto extract_col = [&](int j, py::object col_obj) {
        cat_str[j].resize(n);
        cat_ok[j].resize(n, false);
        if (col_obj.is_none()) return;
        py::array arr  = col_obj.cast<py::array>();
        py::list  lst  = arr.attr("tolist")().cast<py::list>();
        for (int r = 0; r < n; r++) {
            py::object val = lst[r];
            if (val.is_none()) continue;
            try { double dv = val.cast<double>(); if (std::isnan(dv)) continue; } catch (...) {}
            cat_ok[j][r]  = true;
            cat_str[j][r] = py::str(val).cast<std::string>();
        }
    };

    if (!X_cat_raw_py.is_none()) {
        py::list raw_list = X_cat_raw_py.cast<py::list>();
        for (int j = 0; j < p; j++)
            if (td.is_cat_v[j]) extract_col(j, raw_list[j].cast<py::object>());
    }

    // Build test transactions (_apply_bins logic)
    TransList test_trans(n);
    for (int r = 0; r < n; r++) {
        Trans row;
        for (int j = 0; j < p; j++) {
            int bi;
            if (td.is_cat_v[j]) {
                if (!cat_ok[j][r]) continue;
                auto it = label2code[j].find(cat_str[j][r]);
                if (it == label2code[j].end()) continue;
                bi = it->second + 1;
            } else {
                const auto& edges = td.all_edges[j];
                int          nb   = td.nb_col[j];
                double val = td.is_int_v[j]
                             ? Xb(r, j)
                             : (Xb(r, j) - td.col_min[j]) / td.col_range[j];
                // upper_bound on edges[1:-1]
                auto it = std::upper_bound(edges.begin() + 1, edges.end() - 1, val);
                bi = static_cast<int>(it - (edges.begin() + 1)) + 1;
                bi = std::max(1, std::min(bi, nb));
            }
            int bname = TransactionDataCpp::bkey_fn(bi, j);
            auto it   = td.bn2id.find(bname);
            if (it != td.bn2id.end()) row.push_back({it->second, 1.0});
        }
        test_trans[r] = row.empty() ? Trans{{-1, 0.0}} : row;
    }

    return build_matrix_cpp(test_trans, patterns, n);
}

/// Build pattern matrix from training transactions already stored in td.
py::tuple build_train_matrix_cpp(const TransactionDataCpp& td,
                                  const std::vector<PatternEntry>& patterns)
{
    int n = static_cast<int>(td.transactions.size());
    return build_matrix_cpp(td.transactions, patterns, n);
}


// =============================================================================
// Section 7 : pybind11 module
// =============================================================================

PYBIND11_MODULE(_hugiml_core, m)
{
    m.doc() = "HUGIMLClassifierNative — C++ core extension (pybind11)";

    // ── TransactionDataCpp ────────────────────────────────────────────────────
    py::class_<TransactionDataCpp>(m, "TransactionDataCpp",
        "C++ counterpart of _TransactionData; holds all training-time artefacts.")
        .def(py::init<>())
        .def_property_readonly("item_map",  &TransactionDataCpp::get_item_map_py,
             "dict {item_id (int) -> label (str)}")
        .def_readonly("item_twu",  &TransactionDataCpp::item_twu,
             "Transaction-Weighted Utility per item (0-indexed).")
        .def_readonly("nb_col",    &TransactionDataCpp::nb_col,
             "Number of bins / categories per column.")
        .def("riu_thresh",         &TransactionDataCpp::riu_thresh,
             py::arg("k"),
             "Return the k-th largest RIU value (0.0 when k exceeds list length).");

    // ── PatternEntry ──────────────────────────────────────────────────────────
    py::class_<PatternEntry>(m, "PatternEntry",
        "One mined HUG pattern: utility, item IDs, and information gain.")
        .def_readonly("utility", &PatternEntry::utility, "Total utility of this pattern.")
        .def_readonly("items",   &PatternEntry::items,   "List of 1-based item IDs.")
        .def_readonly("ig",      &PatternEntry::ig,       "Information gain (IG).");

    // ── Core functions ────────────────────────────────────────────────────────
    m.def("prepare_transactions", &prepare_transactions_cpp,
        py::arg("X_num"), py::arg("y"), py::arg("B"),
        py::arg("col_names"), py::arg("is_cat"), py::arg("is_int"),
        py::arg("X_cat_raw"),
        "Build utility-annotated TransactionDataCpp from training data.");

    m.def("mine_patterns", &mine_patterns_cpp,
        py::arg("td"), py::arg("ytrain"), py::arg("n_cls"),
        py::arg("K"), py::arg("L"), py::arg("G"),
        "Run top-K HUI mining; returns list of PatternEntry.");

    m.def("build_train_matrix", &build_train_matrix_cpp,
        py::arg("td"), py::arg("patterns"),
        "Build COO (rows, cols) arrays for the training binary pattern matrix.");

    m.def("build_test_matrix", &build_test_matrix_cpp,
        py::arg("X_raw"), py::arg("td"), py::arg("X_cat_raw"), py::arg("patterns"),
        "Apply training bins to test data and build pattern matrix in one pass.");
}
