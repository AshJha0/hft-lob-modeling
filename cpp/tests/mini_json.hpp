#ifndef LOBMOD_TESTS_MINI_JSON_HPP
#define LOBMOD_TESTS_MINI_JSON_HPP

/// \file mini_json.hpp
/// \brief Tiny self-contained JSON reader for the flat golden.json schema
///        ({"cases": [{"name", "inputs": {...}, "expect": {...}, "tol"}]}).
///
/// Supports objects, arrays, strings (no escapes beyond \" \\ \/ \n \t),
/// numbers, true/false/null — more than enough for the bundled files.
/// Test-only code; not part of the lobmod library.

#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_json {

class Value;
using ValuePtr = std::shared_ptr<Value>;

/// One parsed JSON value (tagged union).
class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<ValuePtr> array;
    std::map<std::string, ValuePtr> object;

    /// Object member access; throws on a missing key or non-object.
    const Value& at(const std::string& key) const {
        if (type != Type::Object) throw std::runtime_error("not a JSON object");
        const auto it = object.find(key);
        if (it == object.end()) throw std::runtime_error("missing key: " + key);
        return *it->second;
    }
    bool has(const std::string& key) const {
        return type == Type::Object && object.count(key) > 0;
    }
    double as_number() const {
        if (type != Type::Number) throw std::runtime_error("not a JSON number");
        return number;
    }
    const std::string& as_string() const {
        if (type != Type::String) throw std::runtime_error("not a JSON string");
        return str;
    }
};

/// Recursive-descent parser over an in-memory buffer.
class Parser {
public:
    explicit Parser(std::string text) : s_(std::move(text)) {}

    ValuePtr parse() {
        ValuePtr v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) throw std::runtime_error("trailing JSON content");
        return v;
    }

private:
    void skip_ws() {
        while (pos_ < s_.size() &&
               std::isspace(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }

    char peek() {
        skip_ws();
        if (pos_ >= s_.size()) throw std::runtime_error("unexpected end of JSON");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) {
            throw std::runtime_error(std::string("expected '") + c + "' at " +
                                     std::to_string(pos_));
        }
        ++pos_;
    }

    ValuePtr parse_value() {
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    ValuePtr parse_object() {
        expect('{');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Object;
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            const ValuePtr key = parse_string();
            expect(':');
            v->object[key->str] = parse_value();
            const char c = peek();
            ++pos_;
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("expected ',' or '}'");
        }
        return v;
    }

    ValuePtr parse_array() {
        expect('[');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Array;
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v->array.push_back(parse_value());
            const char c = peek();
            ++pos_;
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("expected ',' or ']'");
        }
        return v;
    }

    ValuePtr parse_string() {
        expect('"');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::String;
        while (pos_ < s_.size() && s_[pos_] != '"') {
            char c = s_[pos_++];
            if (c == '\\') {
                if (pos_ >= s_.size()) throw std::runtime_error("bad escape");
                const char e = s_[pos_++];
                switch (e) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                default: throw std::runtime_error("unsupported escape");
                }
            }
            v->str.push_back(c);
        }
        if (pos_ >= s_.size()) throw std::runtime_error("unterminated string");
        ++pos_; // closing quote
        return v;
    }

    ValuePtr parse_bool() {
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Bool;
        if (s_.compare(pos_, 4, "true") == 0) {
            v->boolean = true;
            pos_ += 4;
        } else if (s_.compare(pos_, 5, "false") == 0) {
            v->boolean = false;
            pos_ += 5;
        } else {
            throw std::runtime_error("bad literal");
        }
        return v;
    }

    ValuePtr parse_null() {
        if (s_.compare(pos_, 4, "null") != 0) throw std::runtime_error("bad literal");
        pos_ += 4;
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Null;
        return v;
    }

    ValuePtr parse_number() {
        skip_ws();
        std::size_t end = pos_;
        while (end < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[end])) ||
                s_[end] == '-' || s_[end] == '+' || s_[end] == '.' ||
                s_[end] == 'e' || s_[end] == 'E')) {
            ++end;
        }
        if (end == pos_) throw std::runtime_error("bad number");
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Number;
        v->number = std::stod(s_.substr(pos_, end - pos_));
        pos_ = end;
        return v;
    }

    std::string s_;
    std::size_t pos_ = 0;
};

/// Parse the JSON file at `path`.
inline ValuePtr parse_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open JSON file: " + path);
    std::stringstream buf;
    buf << in.rdbuf();
    return Parser(buf.str()).parse();
}

} // namespace mini_json

#endif // LOBMOD_TESTS_MINI_JSON_HPP
