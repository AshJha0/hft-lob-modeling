package com.quant.lobmod;

/**
 * One order-flow event. All integer fields are 64-bit ({@code long}) per the
 * cross-language contract; nothing is narrowed.
 *
 * @param seq 1-based sequence number
 * @param etype {@code 'L'} (limit add), {@code 'C'} (cancel by id) or
 *     {@code 'M'} (market order)
 * @param side {@code 'B'} or {@code 'S'} (for {@code 'C'} it echoes the
 *     resting order's side, unused by replay)
 * @param price price in <em>ticks</em> ({@code 'L'} only; 0 otherwise) —
 *     never a currency price
 * @param size order size ({@code 'L'}/{@code 'M'}; 0 for {@code 'C'})
 * @param orderId id of the order ({@code 'L'}: new id; {@code 'C'}: id to
 *     cancel; {@code 'M'}: 0)
 */
public record Event(long seq, char etype, char side, long price, long size, long orderId) {
}
