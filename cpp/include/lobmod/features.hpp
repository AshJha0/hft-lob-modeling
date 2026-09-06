#ifndef LOBMOD_FEATURES_HPP
#define LOBMOD_FEATURES_HPP

/// \file features.hpp
/// \brief Order-flow features and direction labels from replayed book states.
///
/// OFI (Cont-Kukanov-Stoikov style, pinned): per-event contribution for
/// consecutive states n-1 -> n with best bid (Pb, qb) and best ask (Pa, qa)
///
///     e_n =  1{Pb_n >= Pb_{n-1}} qb_n  - 1{Pb_n <= Pb_{n-1}} qb_{n-1}
///          - 1{Pa_n <= Pa_{n-1}} qa_n  + 1{Pa_n >= Pa_{n-1}} qa_{n-1}
///
/// (bid improvements/additions push up, ask improvements/additions push
/// down); e_0 = NaN, and e_n is NaN when any of the four best prices is NaN.
/// OFI over a trailing window of w events ending at event i sums those
/// contributions.
///
/// Feature vector (pinned column order) at event i:
/// [OFI_i, queue imbalance (qb-qa)/(qb+qa), spread, depth_ratio bid/ask
/// over the top k levels, ret = mid_i - mid_{i-ret_window}].
///
/// Label: mid-price direction over the next H events with an epsilon dead
/// zone; classes pinned 0 = down, 1 = flat, 2 = up.  Windows, lookback and
/// horizon are counted in events (the schema has no timestamps); prices
/// and returns are in ticks.

#include <cstddef>
#include <string>
#include <vector>

#include "lobmod/nn.hpp"
#include "lobmod/replay.hpp"

namespace lobmod {

/// Pinned label classes.
constexpr int LABEL_DOWN = 0;
constexpr int LABEL_FLAT = 1;
constexpr int LABEL_UP = 2;

/// Pinned feature column names (order of the X columns).
extern const std::vector<std::string> FEATURE_NAMES;

/// Pinned feature / label / NN hyperparameters (data/features_config.json).
struct FeatureConfig {
    int ofi_window;          ///< trailing OFI window length in events
    int ret_window;          ///< lookback for the short-horizon return
    int horizon;             ///< label horizon H in events
    double epsilon;          ///< dead-zone half-width for the flat class
    int depth_k;             ///< levels per side in the depth-ratio feature
    std::vector<int> hidden; ///< MLP hidden layer sizes
    double learning_rate;    ///< full-batch gradient-descent step
    int epochs;              ///< training epochs (full-batch steps)
    double train_frac;       ///< chronological train fraction

    /// Validate all fields (positive integers / positive lr / epsilon >= 0 /
    /// train_frac in (0,1)).
    /// \throws std::invalid_argument on any violation.
    void validate() const;
};

/// Load a FeatureConfig from its pinned flat JSON file and validate it.
/// \throws std::invalid_argument on an unreadable/malformed file or bad
///         values.
FeatureConfig load_feature_config(const std::string& path);

/// Feature matrix with aligned labels and trading metadata.
struct FeatureSet {
    std::vector<std::vector<double>> X; ///< (n, 5) rows in FEATURE_NAMES order
    std::vector<int> y;                 ///< labels in {0, 1, 2}
    std::vector<std::size_t> event_index; ///< 0-based index into BookStates
    std::vector<double> fwd_ret;        ///< realized mid_{i+H} - mid_i (PnL)
    std::vector<double> spread;         ///< spread at decision time (costs)

    std::size_t size() const { return y.size(); }

    /// The selected rows of X as a dense Matrix (e.g. a chronological_split
    /// index vector).  \throws std::invalid_argument on an out-of-range index.
    Matrix x_rows(const std::vector<std::size_t>& idx) const;
    /// The selected labels.
    std::vector<int> y_rows(const std::vector<std::size_t>& idx) const;
    /// The selected forward returns.
    std::vector<double> fwd_ret_rows(const std::vector<std::size_t>& idx) const;
    /// The selected decision-time spreads.
    std::vector<double> spread_rows(const std::vector<std::size_t>& idx) const;
};

/// Per-event order-flow contributions e_n (file docs formula); e_0 = NaN and
/// events touching an empty side yield NaN.
std::vector<double> ofi_series(const BookStates& states);

/// OFI over the `window` events ending at 0-based `event_index`.  Pinned
/// rule: error iff `event_index < window` (e_0 is undefined); event_index
/// == window is the first valid index and sums e_1 .. e_window.
/// \throws std::invalid_argument if the window does not fit or arguments are
///         invalid.
double ofi_at(const BookStates& states, std::size_t event_index, int window);

/// Build the feature matrix and labels from replayed book states; rows with
/// any undefined input are skipped (see file docs for the pinned emission
/// conditions).
/// \throws std::invalid_argument if no valid rows can be built.
FeatureSet compute_features(const BookStates& states, const FeatureConfig& cfg);

/// Pinned chronological split: the first floor(train_frac * n) row indices
/// train, the rest test.
/// \throws std::invalid_argument if either side would be empty or n <= 1.
std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
chronological_split(std::size_t n, double train_frac);

} // namespace lobmod

#endif // LOBMOD_FEATURES_HPP
