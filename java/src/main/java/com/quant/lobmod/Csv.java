package com.quant.lobmod;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Pinned CSV parsing primitives shared by the event-stream and Hawkes-time
 * readers (API_SPEC.md section 2.4).
 *
 * <p>Rules (identical in every language port):</p>
 * <ul>
 *   <li>the header line must match exactly; a trailing {@code \r} is
 *       stripped from every line first (CRLF files are accepted);</li>
 *   <li>trailing blank lines are ignored; a blank line before the last data
 *       row is an error;</li>
 *   <li>integer fields match {@code -?[0-9]+} and must fit a signed 64-bit
 *       integer; time fields are finite decimal floats;</li>
 *   <li>every violation throws {@link IllegalArgumentException}; file
 *       problems surface as {@link UncheckedIOException}.</li>
 * </ul>
 */
public final class Csv {

    private static final Pattern INT_RE = Pattern.compile("-?[0-9]+");
    private static final Pattern FLOAT_RE =
            Pattern.compile("[+-]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?");

    private Csv() {
    }

    /**
     * Reads a pinned-schema CSV: validates the header, strips {@code \r},
     * drops trailing blank lines and returns the remaining data lines.
     *
     * @throws IllegalArgumentException on a bad header or a blank line before
     *     the last data row
     * @throws UncheckedIOException if the file cannot be read
     */
    public static List<String> readCsvLines(Path path, String expectedHeader) {
        String text;
        try {
            text = Files.readString(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        String[] raw = text.split("\n", -1);
        List<String> lines = new ArrayList<>(raw.length);
        for (String ln : raw) {
            lines.add(ln.endsWith("\r") ? ln.substring(0, ln.length() - 1) : ln);
        }
        while (!lines.isEmpty() && lines.get(lines.size() - 1).isEmpty()) {
            lines.remove(lines.size() - 1);
        }
        if (lines.isEmpty() || !lines.get(0).equals(expectedHeader)) {
            throw new IllegalArgumentException("unexpected CSV header in " + path + ": '"
                    + (lines.isEmpty() ? "<empty file>" : lines.get(0))
                    + "'; expected '" + expectedHeader + "'");
        }
        List<String> data = new ArrayList<>(lines.subList(1, lines.size()));
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).isEmpty()) {
                throw new IllegalArgumentException("blank line at row " + (i + 1) + " of " + path
                        + " (only trailing blank lines are allowed)");
            }
        }
        return data;
    }

    /**
     * Strict signed 64-bit integer parse ({@code -?[0-9]+}, in range).
     *
     * @throws IllegalArgumentException otherwise
     */
    public static long parseInt64Strict(String field, String context) {
        if (!INT_RE.matcher(field).matches()) {
            throw new IllegalArgumentException(
                    "malformed integer field '" + field + "' in " + context);
        }
        try {
            return Long.parseLong(field);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "integer field '" + field + "' out of 64-bit range in " + context, e);
        }
    }

    /**
     * Strict finite double parse (decimal or exponent notation only).
     *
     * @throws IllegalArgumentException otherwise
     */
    public static double parseDoubleStrict(String field, String context) {
        if (!FLOAT_RE.matcher(field).matches()) {
            throw new IllegalArgumentException("malformed time field '" + field + "' in " + context);
        }
        double v = Double.parseDouble(field);
        if (!Double.isFinite(v)) {
            throw new IllegalArgumentException("non-finite time '" + field + "' in " + context);
        }
        return v;
    }

    /** Writes {@code text} to {@code path} (UTF-8, {@code \n} endings). */
    static void writeText(Path path, String text) {
        try {
            Files.writeString(path, text, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
