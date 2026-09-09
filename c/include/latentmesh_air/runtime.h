#ifndef LATENTMESH_AIR_RUNTIME_H
#define LATENTMESH_AIR_RUNTIME_H
#include "latentmesh_air/air.h"
#include <stdatomic.h>

/* Host-only C11 SPSC observer queue. One producer and one consumer. All other
 * runtime state is lane-owned. Callbacks are trusted, synchronous CPU code;
 * this API neither preempts callbacks nor accepts asynchronous CUDA work. */
#define LM_RT_QUEUE_CAPACITY 8u
typedef uint64_t (*lm_rt_clock_fn)(void *user);
typedef struct {
    uint64_t generation;
    uint8_t priority;
} lm_rt_intent_t;
typedef int (*lm_rt_policy_fn)(void *, const lm_rt_intent_t *, lm_rt_intent_t *);
typedef int (*lm_rt_receiver_fn)(void *, const lm_air_block_t *, lm_air_block_t *);
typedef struct {
    lm_air_block_t blocks[LM_RT_QUEUE_CAPACITY];
    atomic_uint head, tail;
    atomic_uint dropped;
} lm_rt_queue_t;
typedef struct {
    lm_rt_clock_fn clock;
    void *clock_user;
    lm_rt_policy_fn policy;
    void *policy_user;
    lm_rt_receiver_fn receiver;
    void *receiver_user;
    uint64_t policy_budget_ns, receiver_budget_ns;
    unsigned policy_misses, receiver_misses;
    unsigned policy_open, receiver_open;
    uint64_t policy_fallbacks, receiver_fallbacks;
    uint64_t last_policy_ns, last_receiver_ns;
    lm_rt_queue_t observer;
} lm_rt_t;

/* Init and configuration only while quiescent. No allocation in dispatch. */
int lm_rt_init(lm_rt_t *, lm_rt_clock_fn, void *);
int lm_rt_publish(lm_rt_queue_t *, const lm_air_block_t *);
int lm_rt_take(lm_rt_queue_t *, lm_air_block_t *);
lm_rt_intent_t lm_rt_decide(lm_rt_t *, lm_rt_intent_t conventional);
/* A candidate is never exposed to the stateful receiver unless synchronous
 * completion met its deadline and its framing metadata passed validation.
 * Accepted candidates still pass the existing CRC/auth/replay pipeline.
 * CRC failure is returned (not retried against already-mutated state). */
lm_air_status_t lm_rt_ingest(lm_rt_t *, lm_air_rx_t *, const lm_air_block_t *);
#endif
