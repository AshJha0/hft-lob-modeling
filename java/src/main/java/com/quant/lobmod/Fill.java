package com.quant.lobmod;

/**
 * One execution against a resting order.
 *
 * @param makerId id of the resting (passive) order that was hit
 * @param price execution price in <em>ticks</em> (the resting order's price)
 * @param size executed quantity (integer, &gt; 0)
 */
public record Fill(int makerId, int price, int size) {
}
