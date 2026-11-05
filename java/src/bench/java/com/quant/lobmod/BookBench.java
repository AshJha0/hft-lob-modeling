package com.quant.lobmod;

import java.lang.management.ManagementFactory;
import java.util.Arrays;

/**
 * Plain timing loop over the reference book. Method and limits are in
 * docs/LATENCY.md. Usage: BookBench [ops] [live_orders] [repeats]
 */
public final class BookBench {
    private BookBench() {
    }

    private static long state = 42;

    private static long next() {
        long z = state += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private static int below(int n) {
        return (int) Long.remainderUnsigned(next(), n);
    }

    /** Adds away from the touch, cancels of live orders, some market orders. */
    private static long pass(int ops, int live) {
        LimitOrderBook book = new LimitOrderBook(1.0, false);
        long[] ids = new long[2 * live + 1];
        int count = 0;
        long nextId = 0;
        long sum = 0;
        for (int i = 0; i < ops; i++) {
            int r = below(100);
            if (count < live || (r < 52 && count < 2 * live)) {
                boolean buy = (next() & 1) != 0;
                long px = 100_000 + (buy ? -1 - below(500) : 1 + below(500));
                book.limitTicks(buy ? LimitOrderBook.BUY : LimitOrderBook.SELL, px, 1 + below(100), ++nextId);
                ids[count++] = nextId;
            } else if (r < 98) {
                int at = below(count);
                long id = ids[at];
                ids[at] = ids[--count];
                if (book.hasOrder(id)) {
                    sum += book.cancel(id);
                }
            } else {
                char side = (next() & 1) != 0 ? LimitOrderBook.BUY : LimitOrderBook.SELL;
                sum += book.market(side, 1 + below(300)).remainder();
            }
        }
        return sum;
    }

    public static void main(String[] args) {
        int ops = args.length > 0 ? Integer.parseInt(args[0]) : 2_000_000;
        int live = args.length > 1 ? Integer.parseInt(args[1]) : 20_000;
        int repeats = args.length > 2 ? Integer.parseInt(args[2]) : 7;
        var threads = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        long tid = Thread.currentThread().threadId();
        long sink = 0;
        for (int i = 0; i < 3; i++) {
            sink += pass(ops, live); // warm-up for the JIT
        }
        double[] ns = new double[repeats];
        double[] bytes = new double[repeats];
        for (int i = 0; i < repeats; i++) {
            long b0 = threads.getThreadAllocatedBytes(tid);
            long t0 = System.nanoTime();
            sink += pass(ops, live);
            ns[i] = (double) (System.nanoTime() - t0) / ops;
            bytes[i] = (double) (threads.getThreadAllocatedBytes(tid) - b0) / ops;
        }
        Arrays.sort(ns);
        Arrays.sort(bytes);
        System.out.printf("Java %s ops=%d live=%d repeats=%d%n",
                System.getProperty("java.version"), ops, live, repeats);
        System.out.println("| book | median ns/op | best ns/op | median bytes/op |");
        System.out.println("|---|---|---|---|");
        System.out.printf("| reference | %.1f | %.1f | %.1f |%n", ns[repeats / 2], ns[0], bytes[repeats / 2]);
        System.out.println("checksum " + sink);
    }
}
