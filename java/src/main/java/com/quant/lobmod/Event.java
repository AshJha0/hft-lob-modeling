package com.quant.lobmod;

/**
 * One order-flow event.
 *
 * @param seq 1-based sequence number
 * @param etype {@code 'L'} (limit add), {@code 'C'} (cancel by id) or
 *     {@code 'M'} (market order)
 * @param side {@code 'B'} or {@code 'S'} (for {@code 'C'} it echoes the
 *     resting order's side, unused by replay)
 * @param price price in ticks ({@code 'L'} only; 0 otherwise)
 * @param size order size ({@code 'L'}/{@code 'M'}; 0 for {@code 'C'})
 * @param orderId id of the order ({@code 'L'}: new id; {@code 'C'}: id to
 *     cancel; {@code 'M'}: 0)
 */
public record Event(int seq, char etype, char side, int price, int size, int orderId) {
}
