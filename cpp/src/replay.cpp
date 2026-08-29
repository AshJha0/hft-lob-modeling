#include "lobmod/replay.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lobmod {

namespace {

/// Split one CSV line on commas (the pinned schema has no quoting).
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) out.push_back(field);
    if (!line.empty() && line.back() == ',') out.emplace_back();
    return out;
}

std::int64_t parse_i64(const std::string& s) {
    std::size_t pos = 0;
    const long long v = std::stoll(s, &pos);
    if (pos != s.size()) {
        throw std::invalid_argument("malformed integer field: " + s);
    }
    return static_cast<std::int64_t>(v);
}

char parse_char(const std::string& s) {
    if (s.size() != 1) {
        throw std::invalid_argument("malformed one-character field: " + s);
    }
    return s[0];
}

} // namespace

std::vector<Event> read_events_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::invalid_argument("cannot open event CSV: " + path);
    }
    std::string line;
    if (!std::getline(in, line) || line == std::string()) {
        throw std::invalid_argument("empty event CSV: " + path);
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "seq,type,side,price,size,order_id") {
        throw std::invalid_argument("unexpected event CSV header: " + line);
    }
    std::vector<Event> events;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto f = split_csv(line);
        if (f.size() != 6) {
            throw std::invalid_argument("malformed event row: " + line);
        }
        events.push_back(Event{parse_i64(f[0]), parse_char(f[1]), parse_char(f[2]),
                               parse_i64(f[3]), parse_i64(f[4]), parse_i64(f[5])});
    }
    return events;
}

void apply_event(LimitOrderBook& book, const Event& event) {
    switch (event.etype) {
    case 'L':
        book.limit(event.side, static_cast<double>(event.price), event.size,
                   event.order_id);
        break;
    case 'C':
        book.cancel(event.order_id);
        break;
    case 'M':
        book.market(event.side, event.size);
        break;
    default:
        throw std::invalid_argument(std::string("unknown event type '") +
                                    event.etype + "'");
    }
}

BookStates replay(const std::vector<Event>& events, int depth_k, double tick_size) {
    if (events.empty()) {
        throw std::invalid_argument("event stream is empty");
    }
    const std::size_t n = events.size();
    BookStates st;
    for (auto* v : {&st.best_bid, &st.best_ask, &st.bid_size, &st.ask_size,
                    &st.bid_depth_k, &st.ask_depth_k, &st.mid, &st.microprice,
                    &st.spread}) {
        v->resize(n);
    }
    LimitOrderBook book(tick_size);
    for (std::size_t i = 0; i < n; ++i) {
        apply_event(book, events[i]);
        const auto b = book.best_bid();
        const auto a = book.best_ask();
        st.best_bid[i] = b ? static_cast<double>(*b) : std::nan("");
        st.best_ask[i] = a ? static_cast<double>(*a) : std::nan("");
        st.bid_size[i] = static_cast<double>(book.best_bid_size());
        st.ask_size[i] = static_cast<double>(book.best_ask_size());
        st.bid_depth_k[i] = static_cast<double>(book.depth('B', depth_k));
        st.ask_depth_k[i] = static_cast<double>(book.depth('S', depth_k));
        st.mid[i] = book.mid();
        st.microprice[i] = book.microprice();
        st.spread[i] = book.spread();
    }
    return st;
}

} // namespace lobmod
