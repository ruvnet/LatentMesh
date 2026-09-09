#include "latentmesh_air/runtime.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
static uint64_t now, step;
static uint64_t clock_fn(void *u) { (void)u; now += step; return now; }
static unsigned delivered;
static lm_air_status_t receive(void *u, const lm_air_message_t *m) {
    (void)u; assert(m->body_len == 5); assert(memcmp(m->body, "hello", 5) == 0);
    ++delivered; return LM_AIR_OK;
}
static int policy(void *u, const lm_rt_intent_t *in, lm_rt_intent_t *out) {
    (void)u; *out = *in; out->priority = 9; return 1;
}
static int stale(void *u, const lm_rt_intent_t *in, lm_rt_intent_t *out) {
    policy(u, in, out); ++out->generation; return 1;
}
static int identity(void *u, const lm_air_block_t *in, lm_air_block_t *out) {
    (void)u; *out = *in; return 1;
}
int main(void) {
    lm_rt_t runtime;
    lm_air_tx_t tx; lm_air_rx_t rx;
    lm_air_profile_config_t profile; lm_air_message_t message;
    lm_air_block_t block, observed;
    lm_rt_intent_t base = {1, 3}; unsigned i;
    assert(lm_rt_init(&runtime, clock_fn, NULL));
    runtime.policy = policy; step = 10;
    assert(lm_rt_decide(&runtime, base).priority == 9);
    runtime.policy = stale;
    assert(lm_rt_decide(&runtime, base).priority == 3);
    runtime.policy = policy; step = 100001;
    for (i=0; i<8; ++i) assert(lm_rt_decide(&runtime, base).priority == 3);
    assert(runtime.policy_open);
    assert(lm_rt_init(&runtime, clock_fn, NULL));
    runtime.receiver = identity;
    assert(lm_air_profile_defaults(LM_AIR_PROFILE_HF_BPSK, &profile) == LM_AIR_OK);
    assert(lm_air_tx_init(&tx, 7, 1, &profile, NULL) == LM_AIR_OK);
    assert(lm_air_rx_init(&rx, receive, NULL, NULL) == LM_AIR_OK);
    memset(&message, 0, sizeof(message));
    message.body = (const uint8_t *)"hello"; message.body_len = 5; message.class_id = 1;
    assert(lm_air_tx_begin(&tx, &message) == LM_AIR_OK);
    assert(lm_air_tx_poll(&tx, &block) == LM_AIR_COMPLETE);
    step = 10;
    assert(lm_rt_ingest(&runtime, &rx, &block) == LM_AIR_COMPLETE);
    assert(delivered == 1);
    assert(lm_rt_take(&runtime.observer, &observed));
    assert(memcmp(&block, &observed, sizeof(block)) == 0);
    assert(lm_rt_ingest(&runtime, &rx, &block) == LM_AIR_ERR_REPLAY);
    lm_air_rx_reset(&rx); step = 150001;
    assert(lm_rt_ingest(&runtime, &rx, &block) == LM_AIR_COMPLETE);
    assert(runtime.receiver_fallbacks == 1);
    while (lm_rt_take(&runtime.observer, &observed)) {}
    for (i=0; i<LM_RT_QUEUE_CAPACITY; ++i) assert(lm_rt_publish(&runtime.observer, &block));
    assert(!lm_rt_publish(&runtime.observer, &block));
    assert(atomic_load(&runtime.observer.dropped) == 1);
    puts("runtime integration: PASS (CPU fixture; no OTA qualification)");
    return 0;
}
