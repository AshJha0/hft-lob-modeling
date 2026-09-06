/// Feature pipeline: pinned OFI formula, emission conditions, labels,
/// chronological split, and config validation.

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include "lobmod/features.hpp"
#include "test_data.hpp"

namespace {

lobmod::BookStates tiny_states() {
    // hand-built two-event state pair for the OFI formula
    lobmod::BookStates st;
    st.best_bid = {100.0, 101.0};
    st.best_ask = {102.0, 102.0};
    st.bid_size = {5.0, 3.0};
    st.ask_size = {4.0, 6.0};
    st.bid_depth_k = {5.0, 3.0};
    st.ask_depth_k = {4.0, 6.0};
    st.mid = {101.0, 101.5};
    st.microprice = {101.0, 101.5};
    st.spread = {2.0, 1.0};
    return st;
}

TEST(Ofi, PinnedFormulaHandChecked) {
    const auto st = tiny_states();
    const auto e = lobmod::ofi_series(st);
    ASSERT_EQ(e.size(), 2u);
    EXPECT_TRUE(std::isnan(e[0])); // no predecessor
    // bid up (101 > 100): +qb_n = +3; not <=, no -qb_{n-1}
    // ask equal: -qa_n + qa_{n-1} = -6 + 4 = -2  ==> e_1 = 1
    EXPECT_DOUBLE_EQ(e[1], 1.0);
    EXPECT_DOUBLE_EQ(lobmod::ofi_at(st, 1, 1), 1.0);
}

TEST(Ofi, NanWhenSideEmpty) {
    auto st = tiny_states();
    st.best_ask[0] = std::nan("");
    const auto e = lobmod::ofi_series(st);
    EXPECT_TRUE(std::isnan(e[1])); // previous state had an empty side
}

TEST(Ofi, WindowValidation) {
    const auto st = tiny_states();
    EXPECT_THROW(lobmod::ofi_at(st, 1, 0), std::invalid_argument);
    EXPECT_THROW(lobmod::ofi_at(st, 1, 3), std::invalid_argument); // no fit
    EXPECT_THROW(lobmod::ofi_at(st, 7, 1), std::invalid_argument); // range
}

TEST(Features, RowCountAndColumnsOnBundledStream) {
    const auto& fs = testdata::pipeline().fs;
    // n_test pinned by the golden inputs: 1305 = size - floor(0.7 * size)
    EXPECT_EQ(fs.size() - testdata::pipeline().train_idx.size(), 1305u);
    for (const auto& row : fs.X) ASSERT_EQ(row.size(), 5u);
    ASSERT_EQ(fs.y.size(), fs.X.size());
    ASSERT_EQ(fs.fwd_ret.size(), fs.X.size());
    ASSERT_EQ(fs.spread.size(), fs.X.size());
    for (int y : fs.y) {
        ASSERT_GE(y, 0);
        ASSERT_LE(y, 2);
    }
    // spot-check the pinned label rule on every row
    const auto& cfg = testdata::config();
    for (std::size_t i = 0; i < fs.size(); ++i) {
        const double d = fs.fwd_ret[i];
        const int expect = d > cfg.epsilon ? 2 : (d < -cfg.epsilon ? 0 : 1);
        ASSERT_EQ(fs.y[i], expect) << "row " << i;
    }
}

TEST(Features, QueueImbalanceBounds) {
    const auto& fs = testdata::pipeline().fs;
    for (const auto& row : fs.X) {
        ASSERT_GE(row[1], -1.0); // queue imbalance in [-1, 1]
        ASSERT_LE(row[1], 1.0);
        ASSERT_GT(row[2], 0.0);  // spread positive (book uncrossed)
        ASSERT_GE(row[3], 0.0);  // depth ratio non-negative
    }
}

TEST(Features, EmptyStatesErrors) {
    lobmod::BookStates st = tiny_states();
    // horizon larger than the stream: no emittable rows
    EXPECT_THROW(lobmod::compute_features(st, testdata::config()),
                 std::invalid_argument);
}

TEST(Features, ConfigValidation) {
    lobmod::FeatureConfig cfg = testdata::config();
    cfg.validate();
    lobmod::FeatureConfig bad = cfg;
    bad.ofi_window = 0;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = cfg;
    bad.epsilon = -0.1;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = cfg;
    bad.learning_rate = 0.0;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = cfg;
    bad.train_frac = 1.0;
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    bad = cfg;
    bad.hidden = {16, 0};
    EXPECT_THROW(bad.validate(), std::invalid_argument);
    EXPECT_THROW(lobmod::load_feature_config("/nonexistent.json"),
                 std::invalid_argument);
}

TEST(Features, ConfigPinnedValues) {
    const auto& cfg = testdata::config();
    EXPECT_EQ(cfg.ofi_window, 50);
    EXPECT_EQ(cfg.ret_window, 20);
    EXPECT_EQ(cfg.horizon, 10);
    EXPECT_DOUBLE_EQ(cfg.epsilon, 0.25);
    EXPECT_EQ(cfg.depth_k, 3);
    ASSERT_EQ(cfg.hidden.size(), 2u);
    EXPECT_EQ(cfg.hidden[0], 16);
    EXPECT_EQ(cfg.hidden[1], 8);
    EXPECT_DOUBLE_EQ(cfg.learning_rate, 0.1);
    EXPECT_EQ(cfg.epochs, 800);
    EXPECT_DOUBLE_EQ(cfg.train_frac, 0.7);
}

TEST(Features, ChronologicalSplit) {
    const auto [tr, te] = lobmod::chronological_split(10, 0.7);
    ASSERT_EQ(tr.size(), 7u);
    ASSERT_EQ(te.size(), 3u);
    EXPECT_EQ(tr.front(), 0u);
    EXPECT_EQ(tr.back(), 6u);
    EXPECT_EQ(te.front(), 7u); // strictly after the train block: no leakage
    EXPECT_EQ(te.back(), 9u);
    EXPECT_THROW(lobmod::chronological_split(1, 0.7), std::invalid_argument);
    EXPECT_THROW(lobmod::chronological_split(10, 0.0), std::invalid_argument);
    EXPECT_THROW(lobmod::chronological_split(10, 1.0), std::invalid_argument);
    EXPECT_THROW(lobmod::chronological_split(3, 0.01), std::invalid_argument);
}

// ------------------------------------------------------------ review additions

TEST(Ofi, WindowBoundaryIsPinned) {
    // PT-3: error iff event_index < window; event_index == window is valid
    lobmod::BookStates st;
    for (int i = 0; i < 10; ++i) {
        st.best_bid.push_back(100.0);
        st.best_ask.push_back(101.0);
        st.bid_size.push_back(5.0);
        st.ask_size.push_back(5.0);
        st.bid_depth_k.push_back(5.0);
        st.ask_depth_k.push_back(5.0);
        st.mid.push_back(100.5);
        st.microprice.push_back(100.5);
        st.spread.push_back(1.0);
    }
    EXPECT_THROW(lobmod::ofi_at(st, 3, 4), std::invalid_argument);
    const double v = lobmod::ofi_at(st, 4, 4); // sums e_1..e_4: finite, zero
    EXPECT_TRUE(std::isfinite(v));
    EXPECT_DOUBLE_EQ(v, 0.0);
    EXPECT_THROW(lobmod::ofi_at(st, 9, 10), std::invalid_argument);
    EXPECT_DOUBLE_EQ(lobmod::ofi_at(st, 9, 9), 0.0);
}

} // namespace
