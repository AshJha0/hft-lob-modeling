#include "lobmod/replay.hpp"
#include "lobmod/csv.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lobmod {

namespace {

constexpr const char* kHeader = "seq,type,side,price,size,order_id";

char parse_code(const std::string& s, const char* allowed, const char* what,
                const std::string& context) {
    if (s.size() == 1) {
        for (const char* a = allowed; *a != '\0'; ++a) {
            if (*a == s[0]) return s[0];
        }
    }
    throw std::invalid_argument(std::string("invalid ") + what + " '" + s +
                                "' in " + context);
}

} // namespace

std::vector<Event> read_events_csv(const std::string& path) {
    const auto lines = read_csv_lines(path, kHeader);
    std::vector<Event> events;
    events.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string context = "event row " + std::to_string(i + 1);
        const auto f = split_csv_fields(lines[i]);
        if (f.size() != 6) {
            throw std::invalid_argument("malformed " + context + ": expected 6 fields, got " +
                                        std::to_string(f.size()) + ": '" + lines[i] + "'");
        }
        events.push_back(Event{parse_int64_strict(f[0], context),
                               parse_code(f[1], "LCM", "event type", context),
                               parse_code(f[2], "BS", "side", context),
                               parse_int64_strict(f[3], context),
                               parse_int64_strict(f[4], context),
                               parse_int64_strict(f[5], context)});
    }
    return events;
}

void write_events_csv(const std::string& path, const std::vector<Event>& events) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::invalid_argument("cannot write event CSV: " + path);
    }
    out << kHeader << '\n';
    for (const Event& e : events) {
        out << e.seq << ',' << e.etype << ',' << e.side << ',' << e.price << ','
            << e.size << ',' << e.order_id << '\n';
    }
    if (!out) {
        throw std::invalid_argument("write failed for event CSV: " + path);
    }
}

void apply_event(LimitOrderBook& book, const Event& event) {
    switch (event.etype) {
    case 'L':
        book.limit_ticks(event.side, event.price, event.size, event.order_id);
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

BookStates replay(const std::vector<Event>& events, int depth_k) {
    if (depth_k <= 0) {
        throw std::invalid_argument("depth_k must be a positive integer, got " +
                                    std::to_string(depth_k));
    }
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
    LimitOrderBook book(1.0, /*record_trades=*/false);
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
