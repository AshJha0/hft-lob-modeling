#ifndef LOBMOD_TESTS_TEST_DATA_HPP
#define LOBMOD_TESTS_TEST_DATA_HPP

/// \file test_data.hpp
/// \brief Shared lazy-loaded fixtures for the test suite: bundled datasets,
///        golden cases, and the expensive fit / training pipeline computed
///        once per test run.

#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lobmod/features.hpp"
#include "lobmod/hawkes.hpp"
#include "lobmod/nn.hpp"
#include "lobmod/replay.hpp"
#include "mini_json.hpp"

#ifndef LOBMOD_DATA_DIR
#define LOBMOD_DATA_DIR "../data"
#endif

namespace testdata {

inline std::string data_dir() { return LOBMOD_DATA_DIR; }

/// One golden case: flat input/expect maps plus its tolerance.
struct GoldenCase {
    std::map<std::string, double> inputs;
    std::map<std::string, double> expect;
    double tol = 0.0;
};

/// All golden cases keyed by name (loaded once).
inline const std::map<std::string, GoldenCase>& golden() {
    static const auto cases = [] {
        std::map<std::string, GoldenCase> out;
        const auto root = mini_json::parse_file(data_dir() + "/golden/golden.json");
        for (const auto& c : root->at("cases").array) {
            GoldenCase gc;
            for (const auto& [k, v] : c->at("inputs").object) {
                gc.inputs[k] = v->as_number();
            }
            for (const auto& [k, v] : c->at("expect").object) {
                gc.expect[k] = v->as_number();
            }
            gc.tol = c->at("tol").as_number();
            out[c->at("name").as_string()] = gc;
        }
        return out;
    }();
    return cases;
}

/// The bundled 5000-event LOB stream.
inline const std::vector<lobmod::Event>& events() {
    static const auto ev =
        lobmod::read_events_csv(data_dir() + "/lob_events.csv");
    return ev;
}

/// The pinned feature configuration.
inline const lobmod::FeatureConfig& config() {
    static const auto cfg =
        lobmod::load_feature_config(data_dir() + "/features_config.json");
    return cfg;
}

/// Replayed book states for the bundled stream (depth_k from the config).
inline const lobmod::BookStates& states() {
    static const auto st = lobmod::replay(events(), config().depth_k);
    return st;
}

/// Bundled univariate Hawkes event times (header "t").
inline const std::vector<double>& uni_times() {
    static const auto t = [] {
        std::ifstream in(data_dir() + "/hawkes_uni.csv");
        if (!in) throw std::runtime_error("cannot open hawkes_uni.csv");
        std::string line;
        std::getline(in, line);
        std::vector<double> out;
        while (std::getline(in, line)) {
            if (!line.empty()) out.push_back(std::stod(line));
        }
        return out;
    }();
    return t;
}

/// Bundled bivariate Hawkes stream (header "t,type").
inline const std::pair<std::vector<double>, std::vector<int>>& biv_data() {
    static const auto data = [] {
        std::ifstream in(data_dir() + "/hawkes_biv.csv");
        if (!in) throw std::runtime_error("cannot open hawkes_biv.csv");
        std::string line;
        std::getline(in, line);
        std::pair<std::vector<double>, std::vector<int>> out;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const std::size_t comma = line.find(',');
            out.first.push_back(std::stod(line.substr(0, comma)));
            out.second.push_back(std::stoi(line.substr(comma + 1)));
        }
        return out;
    }();
    return data;
}

constexpr double kUniT = 1000.0; // dgp_params.json: hawkes_uni.T
constexpr double kBivT = 600.0;  // dgp_params.json: hawkes_biv.T

/// Univariate MLE on the bundled sample (computed once).
inline const lobmod::HawkesUniFit& uni_fit() {
    static const auto fit = lobmod::fit_uni(uni_times(), kUniT);
    return fit;
}

/// Bivariate MLE on the bundled sample (computed once).
inline const lobmod::HawkesBivFit& biv_fit() {
    static const auto fit =
        lobmod::fit_biv(biv_data().first, biv_data().second, kBivT);
    return fit;
}

/// The full deterministic learning pipeline (features -> split ->
/// standardize -> train MLP + logistic), computed once.
struct Pipeline {
    lobmod::FeatureSet fs;
    std::vector<std::size_t> train_idx;
    std::vector<std::size_t> test_idx;
    lobmod::Matrix x_train;
    lobmod::Matrix x_test;
    std::vector<int> y_train;
    std::vector<int> y_test;
    std::unique_ptr<lobmod::MLP> net;
    std::unique_ptr<lobmod::MLP> logistic;
    std::vector<double> history;
};

inline const Pipeline& pipeline() {
    static const auto p = [] {
        auto pl = std::make_unique<Pipeline>();
        pl->fs = lobmod::compute_features(states(), config());
        auto [tr, te] = lobmod::chronological_split(pl->fs.size(),
                                                    config().train_frac);
        pl->train_idx = tr;
        pl->test_idx = te;
        lobmod::Matrix x_tr(tr.size(), 5), x_te(te.size(), 5);
        pl->y_train.resize(tr.size());
        pl->y_test.resize(te.size());
        for (std::size_t i = 0; i < tr.size(); ++i) {
            for (std::size_t j = 0; j < 5; ++j) x_tr(i, j) = pl->fs.X[tr[i]][j];
            pl->y_train[i] = pl->fs.y[tr[i]];
        }
        for (std::size_t i = 0; i < te.size(); ++i) {
            for (std::size_t j = 0; j < 5; ++j) x_te(i, j) = pl->fs.X[te[i]][j];
            pl->y_test[i] = pl->fs.y[te[i]];
        }
        lobmod::Standardizer sc;
        sc.fit(x_tr);
        pl->x_train = sc.transform(x_tr);
        pl->x_test = sc.transform(x_te);
        pl->net = std::make_unique<lobmod::MLP>(5, config().hidden, 3);
        pl->history = pl->net->fit(pl->x_train, pl->y_train,
                                   config().learning_rate, config().epochs);
        pl->logistic = std::make_unique<lobmod::MLP>(5, std::vector<int>{}, 3);
        pl->logistic->fit(pl->x_train, pl->y_train, config().learning_rate,
                          config().epochs);
        return pl;
    }();
    return *p;
}

} // namespace testdata

#endif // LOBMOD_TESTS_TEST_DATA_HPP
