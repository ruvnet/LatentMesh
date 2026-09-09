#include "latentmesh_air/runtime.h"
#include <string.h>

int lm_rt_init(lm_rt_t *r, lm_rt_clock_fn clock, void *user) {
    if (!r || !clock) return 0;
    memset(r, 0, sizeof(*r));
    atomic_init(&r->observer.head, 0u);
    atomic_init(&r->observer.tail, 0u);
    atomic_init(&r->observer.dropped, 0u);
    if (!atomic_is_lock_free(&r->observer.head) ||
        !atomic_is_lock_free(&r->observer.tail) ||
        !atomic_is_lock_free(&r->observer.dropped)) return 0;
    r->clock = clock; r->clock_user = user;
    r->policy_budget_ns = 100000u; r->receiver_budget_ns = 150000u;
    return 1;
}
int lm_rt_publish(lm_rt_queue_t *q, const lm_air_block_t *b) {
    unsigned h = atomic_load_explicit(&q->head, memory_order_relaxed);
    unsigned t = atomic_load_explicit(&q->tail, memory_order_acquire);
    if (h - t >= LM_RT_QUEUE_CAPACITY) {
        atomic_fetch_add_explicit(&q->dropped, 1u, memory_order_relaxed);
        return 0;
    }
    q->blocks[h % LM_RT_QUEUE_CAPACITY] = *b;
    atomic_store_explicit(&q->head, h + 1u, memory_order_release);
    return 1;
}
int lm_rt_take(lm_rt_queue_t *q, lm_air_block_t *b) {
    unsigned t = atomic_load_explicit(&q->tail, memory_order_relaxed);
    unsigned h = atomic_load_explicit(&q->head, memory_order_acquire);
    if (h == t) return 0;
    *b = q->blocks[t % LM_RT_QUEUE_CAPACITY];
    atomic_store_explicit(&q->tail, t + 1u, memory_order_release);
    return 1;
}
static int elapsed(lm_rt_t *r, uint64_t start, uint64_t *duration, uint64_t budget) {
    uint64_t end = r->clock(r->clock_user);
    *duration = end >= start ? end - start : UINT64_MAX;
    return *duration <= budget;
}
lm_rt_intent_t lm_rt_decide(lm_rt_t *r, lm_rt_intent_t baseline) {
    lm_rt_intent_t candidate = baseline;
    uint64_t start;
    int ok;
    r->last_policy_ns = 0u;
    if (!r->policy || r->policy_open) { ++r->policy_fallbacks; return baseline; }
    start = r->clock(r->clock_user);
    ok = r->policy(r->policy_user, &baseline, &candidate);
    ok = elapsed(r, start, &r->last_policy_ns, r->policy_budget_ns) && ok;
    if (!ok || candidate.generation != baseline.generation || candidate.priority > 15u) {
        ++r->policy_fallbacks;
        if (++r->policy_misses >= 8u) r->policy_open = 1u;
        return baseline;
    }
    r->policy_misses = 0u;
    return candidate;
}
lm_air_status_t lm_rt_ingest(lm_rt_t *r, lm_air_rx_t *rx, const lm_air_block_t *b) {
    lm_air_block_t candidate;
    const lm_air_block_t *selected = b;
    uint64_t start;
    int ok = 0;
    if (!r || !rx || !b || !r->clock) return LM_AIR_ERR_ARGUMENT;
    r->last_receiver_ns = 0u;
    if (r->receiver && !r->receiver_open) {
        candidate = *b;
        start = r->clock(r->clock_user);
        ok = r->receiver(r->receiver_user, b, &candidate);
        ok = elapsed(r, start, &r->last_receiver_ns, r->receiver_budget_ns) && ok;
        ok = ok && candidate.raw_len == b->raw_len && candidate.bit_len == b->bit_len &&
            candidate.fec == b->fec && candidate.interleave_rows == b->interleave_rows;
        if (ok) { selected = &candidate; r->receiver_misses = 0u; }
        else if (++r->receiver_misses >= 8u) r->receiver_open = 1u;
    }
    if (!ok) ++r->receiver_fallbacks;
    (void)lm_rt_publish(&r->observer, selected);
    return lm_air_rx_ingest_block(rx, selected);
}
