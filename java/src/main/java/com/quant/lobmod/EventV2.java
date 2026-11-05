package com.quant.lobmod;

/**
 * One event of schema v2, header
 * {@code seq,ts_ns,type,side,price,size,order_id,new_order_id}.
 *
 * @param seq 1-based sequence number
 * @param tsNs non-negative nanoseconds (0 when lifted from v1)
 * @param etype {@code L}, {@code C}, {@code M}, or {@code R} reduce,
 *     {@code U} replace, {@code E} execute
 * @param side {@code 'B'} or {@code 'S'} (echoes the resting side for
 *     C/R/U/E)
 * @param price price in ticks (L and U; 0 otherwise)
 * @param size L/M size; R size removed; U new size; E executed; C 0
 * @param orderId L new id; C/R/U/E resting id; M 0
 * @param newOrderId U replacement id; 0 otherwise
 */
public record EventV2(long seq, long tsNs, char etype, char side, long price, long size,
        long orderId, long newOrderId) {
}
