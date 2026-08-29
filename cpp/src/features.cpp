#include "lobmod/features.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lobmod {

const std::vector<std::string> FEATURE_NAMES = {
    "ofi", "queue_imbalance", "spread", "depth_ratio", "ret"};

void FeatureConfig::validate() const {
    const std::pair<const char*, int> ints[] = {
        {"ofi_window", ofi_window}, {"ret_window", ret_window},
        {"horizon", horizon},       {"depth_k", depth_k},
        {"epochs", epochs}};
    for (const auto& [name, v] : ints) {
        if (v <= 0) {
            throw std::invalid_argument(std::string(name) +
                                        " must be a positive integer, got " +
                                        std::to_string(v));
        }
    }
    if (!(std::isfinite(epsilon) && epsilon >= 0.0)) {
        throw std::invalid_argument("epsilon must be >= 0, got " +
                                    std::to_string(epsilon));
    }
    if (!(std::isfinite(learning_rate) && learning_rate > 0.0)) {
        throw std::invalid_argument("learning_rate must be positive, got " +
                                    std::to_string(learning_rate));
    }
    if (!(train_frac > 0.0 && train_frac < 1.0)) {
        throw std::invalid_argument("train_frac must be in (0, 1), got " +
                                    std::to_string(train_frac));
    }
    for (int h : hidden) {
        if (h <= 0) {
            throw std::invalid_argument("hidden sizes must be positive integers");
        }
    }
}

namespace {

/// Minimal scanner for the flat config JSON: `"key": number` and
/// `"key": [numbers]` pairs only — sufficient for features_config.json.
class ConfigScanner {
public:
    explicit ConfigScanner(std::string text) : s_(std::move(text)) {}

    double number(const std::string& key) const {
        std::size_t pos = find_value(key);
        return parse_number(pos);
    }

    std::vector<int> int_array(const std::string& key) const {
        std::size_t pos = find_value(key);
        if (s_[pos] != '[') {
            throw std::invalid_argument("config key '" + key + "' is not an array");
        }
        ++pos;
        std::vector<int> out;
        while (true) {
            pos = skip_ws(pos);
            if (pos < s_.size() && s_[pos] == ']') break;
            std::size_t end = pos;
            out.push_back(static_cast<int>(parse_number(end)));
            pos = skip_ws(advance_number(pos));
            if (pos < s_.size() && s_[pos] == ',') ++pos;
        }
        return out;
    }

private:
    std::size_t skip_ws(std::size_t pos) const {
        while (pos < s_.size() &&
               std::isspace(static_cast<unsigned char>(s_[pos]))) {
            ++pos;
        }
        return pos;
    }

    std::size_t advance_number(std::size_t pos) const {
        while (pos < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[pos])) ||
                s_[pos] == '-' || s_[pos] == '+' || s_[pos] == '.' ||
                s_[pos] == 'e' || s_[pos] == 'E')) {
            ++pos;
        }
        return pos;
    }

    std::size_t find_value(const std::string& key) const {
        const std::string quoted = "\"" + key + "\"";
        const std::size_t at = s_.find(quoted);
        if (at == std::string::npos) {
            throw std::invalid_argument("config key '" + key + "' not found");
        }
        std::size_t pos = skip_ws(at + quoted.size());
        if (pos >= s_.size() || s_[pos] != ':') {
            throw std::invalid_argument("malformed config near key '" + key + "'");
        }
        return skip_ws(pos + 1);
    }

    double parse_number(std::size_t pos) const {
        try {
            return std::stod(s_.substr(pos, 64));
        } catch (const std::exception&) {
            throw std::invalid_argument("malformed number in config");
        }
    }

    std::string s_;
};

} // namespace

FeatureConfig load_feature_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("cannot open feature config: " + path);
    }
    std::stringstream buf;
    buf << in.rdbuf();
    const ConfigScanner scan(buf.str());
    FeatureConfig cfg;
    cfg.ofi_window = static_cast<int>(scan.number("ofi_window"));
    cfg.ret_window = static_cast<int>(scan.number("ret_window"));
    cfg.horizon = static_cast<int>(scan.number("horizon"));
    cfg.epsilon = scan.number("epsilon");
    cfg.depth_k = static_cast<int>(scan.number("depth_k"));
    cfg.hidden = scan.int_array("hidden");
    cfg.learning_rate = scan.number("learning_rate");
    cfg.epochs = static_cast<int>(scan.number("epochs"));
    cfg.train_frac = scan.number("train_frac");
    cfg.validate();
    return cfg;
}

std::vector<double> ofi_series(const BookStates& states) {
    const std::size_t n = states.size();
    std::vector<double> e(n, std::nan(""));
    for (std::size_t i = 1; i < n; ++i) {
        const double pb1 = states.best_bid[i], pb0 = states.best_bid[i - 1];
        const double pa1 = states.best_ask[i], pa0 = states.best_ask[i - 1];
        if (!(std::isfinite(pb1) && std::isfinite(pb0) && std::isfinite(pa1) &&
              std::isfinite(pa0))) {
            continue; // e[i] stays NaN
        }
        double contrib = 0.0;
        if (pb1 >= pb0) contrib += states.bid_size[i];
        if (pb1 <= pb0) contrib -= states.bid_size[i - 1];
        if (pa1 <= pa0) contrib -= states.ask_size[i];
        if (pa1 >= pa0) contrib += states.ask_size[i - 1];
        e[i] = contrib;
    }
    return e;
}

double ofi_at(const BookStates& states, std::size_t event_index, int window) {
    if (window <= 0) {
        throw std::invalid_argument("window must be a positive integer, got " +
                                    std::to_string(window));
    }
    const std::size_t n = states.size();
    if (event_index >= n) {
        throw std::invalid_argument("event_index " + std::to_string(event_index) +
                                    " out of range [0, " + std::to_string(n) + ")");
    }
    if (event_index + 1 < static_cast<std::size_t>(window)) {
        throw std::invalid_argument("window " + std::to_string(window) +
                                    " does not fit before event " +
                                    std::to_string(event_index));
    }
    const std::vector<double> e = ofi_series(states);
    double s = 0.0;
    for (std::size_t i = event_index + 1 - static_cast<std::size_t>(window);
         i <= event_index; ++i) {
        s += e[i];
    }
    return s;
}

FeatureSet compute_features(const BookStates& states, const FeatureConfig& cfg) {
    cfg.validate();
    const std::size_t n = states.size();
    const std::vector<double> e = ofi_series(states);
    const std::size_t start =
        static_cast<std::size_t>(std::max(cfg.ofi_window, cfg.ret_window));
    const std::size_t horizon = static_cast<std::size_t>(cfg.horizon);
    FeatureSet fs;
    for (std::size_t i = start; i + horizon < n; ++i) {
        bool window_ok = true;
        for (std::size_t j = i + 1 - static_cast<std::size_t>(cfg.ofi_window);
             j <= i; ++j) {
            if (!std::isfinite(e[j])) {
                window_ok = false;
                break;
            }
        }
        if (!window_ok) continue;
        const double qb = states.bid_size[i], qa = states.ask_size[i];
        const double mid_now = states.mid[i];
        const double mid_past = states.mid[i - static_cast<std::size_t>(cfg.ret_window)];
        const double mid_fwd = states.mid[i + horizon];
        const double spread = states.spread[i];
        const double bd = states.bid_depth_k[i], ad = states.ask_depth_k[i];
        if (!(std::isfinite(mid_now) && std::isfinite(mid_past) &&
              std::isfinite(mid_fwd) && std::isfinite(spread))) {
            continue;
        }
        if (qb + qa <= 0.0 || ad <= 0.0) continue;
        double ofi = 0.0;
        for (std::size_t j = i + 1 - static_cast<std::size_t>(cfg.ofi_window);
             j <= i; ++j) {
            ofi += e[j];
        }
        const double qi = (qb - qa) / (qb + qa);
        const double depth_ratio = bd / ad;
        const double ret = mid_now - mid_past;
        const double d = mid_fwd - mid_now;
        int label;
        if (d > cfg.epsilon) {
            label = LABEL_UP;
        } else if (d < -cfg.epsilon) {
            label = LABEL_DOWN;
        } else {
            label = LABEL_FLAT;
        }
        fs.X.push_back({ofi, qi, spread, depth_ratio, ret});
        fs.y.push_back(label);
        fs.event_index.push_back(i);
        fs.fwd_ret.push_back(d);
        fs.spread.push_back(spread);
    }
    if (fs.y.empty()) {
        throw std::invalid_argument(
            "no valid feature rows could be built from the states");
    }
    return fs;
}

std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
chronological_split(std::size_t n, double train_frac) {
    if (n <= 1) {
        throw std::invalid_argument("n must be an integer > 1, got " +
                                    std::to_string(n));
    }
    if (!(train_frac > 0.0 && train_frac < 1.0)) {
        throw std::invalid_argument("train_frac must be in (0, 1), got " +
                                    std::to_string(train_frac));
    }
    const std::size_t n_train = static_cast<std::size_t>(
        std::floor(train_frac * static_cast<double>(n)));
    if (n_train == 0 || n_train == n) {
        throw std::invalid_argument("split leaves an empty side: n=" +
                                    std::to_string(n));
    }
    std::vector<std::size_t> train(n_train), test(n - n_train);
    for (std::size_t i = 0; i < n_train; ++i) train[i] = i;
    for (std::size_t i = n_train; i < n; ++i) test[i - n_train] = i;
    return {std::move(train), std::move(test)};
}

} // namespace lobmod
