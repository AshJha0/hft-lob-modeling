package com.quant.lobmod;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal recursive-descent JSON parser for the bundled flat config /
 * golden-value files.
 *
 * <p>Supports the full JSON grammar needed by {@code data/*.json} (objects,
 * arrays, strings with escapes, numbers, booleans, null). Numbers parse as
 * {@link Double}, objects as {@link LinkedHashMap}, arrays as
 * {@link ArrayList}. Package-private: an implementation detail of config
 * loading and the test suite, not part of the public API surface.</p>
 */
final class MiniJson {

    private final String src;
    private int pos;

    private MiniJson(String src) {
        this.src = src;
        this.pos = 0;
    }

    /** Parses a complete JSON document; trailing garbage is an error. */
    static Object parse(String text) {
        MiniJson p = new MiniJson(text);
        p.skipWs();
        Object value = p.parseValue();
        p.skipWs();
        if (p.pos != p.src.length()) {
            throw new IllegalStateException("trailing characters at offset " + p.pos);
        }
        return value;
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }

    private char peek() {
        if (pos >= src.length()) {
            throw new IllegalStateException("unexpected end of JSON input");
        }
        return src.charAt(pos);
    }

    private void expect(char c) {
        if (peek() != c) {
            throw new IllegalStateException(
                    "expected '" + c + "' at offset " + pos + ", got '" + peek() + "'");
        }
        pos++;
    }

    private Object parseValue() {
        char c = peek();
        switch (c) {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"':
                return parseString();
            case 't':
                literal("true");
                return Boolean.TRUE;
            case 'f':
                literal("false");
                return Boolean.FALSE;
            case 'n':
                literal("null");
                return null;
            default:
                return parseNumber();
        }
    }

    private void literal(String word) {
        if (!src.startsWith(word, pos)) {
            throw new IllegalStateException("invalid literal at offset " + pos);
        }
        pos += word.length();
    }

    private Map<String, Object> parseObject() {
        Map<String, Object> map = new LinkedHashMap<>();
        expect('{');
        skipWs();
        if (peek() == '}') {
            pos++;
            return map;
        }
        while (true) {
            skipWs();
            String key = parseString();
            skipWs();
            expect(':');
            skipWs();
            map.put(key, parseValue());
            skipWs();
            char c = peek();
            if (c == ',') {
                pos++;
            } else if (c == '}') {
                pos++;
                return map;
            } else {
                throw new IllegalStateException("expected ',' or '}' at offset " + pos);
            }
        }
    }

    private List<Object> parseArray() {
        List<Object> list = new ArrayList<>();
        expect('[');
        skipWs();
        if (peek() == ']') {
            pos++;
            return list;
        }
        while (true) {
            skipWs();
            list.add(parseValue());
            skipWs();
            char c = peek();
            if (c == ',') {
                pos++;
            } else if (c == ']') {
                pos++;
                return list;
            } else {
                throw new IllegalStateException("expected ',' or ']' at offset " + pos);
            }
        }
    }

    private String parseString() {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = src.charAt(pos++);
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                char esc = src.charAt(pos++);
                switch (esc) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> throw new IllegalStateException("bad escape '\\" + esc + "'");
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Double parseNumber() {
        int start = pos;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'
                    || (c >= '0' && c <= '9')) {
                pos++;
            } else {
                break;
            }
        }
        if (start == pos) {
            throw new IllegalStateException("invalid number at offset " + pos);
        }
        return Double.parseDouble(src.substring(start, pos));
    }
}
