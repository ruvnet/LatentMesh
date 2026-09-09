#define _POSIX_C_SOURCE 200809L
#include "latentmesh_air/runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static uint64_t clock_ns(void *u) {
    struct timespec t; (void)u;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) abort();
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static int policy(void *u, const lm_rt_intent_t *a, lm_rt_intent_t *b) {
    (void)u; *b = *a; return 1;
}
static int compare(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
int main(void) {
    lm_rt_t r; uint64_t samples[20000]; size_t i;
    lm_rt_intent_t input = {1, 3};
    if (!lm_rt_init(&r, clock_ns, NULL)) return 1;
    r.policy = policy;
    for (i=0; i<20000; ++i) {
        uint64_t start = clock_ns(NULL);
        input = lm_rt_decide(&r, input);
        samples[i] = clock_ns(NULL) - start;
    }
    qsort(samples, 20000, sizeof(*samples), compare);
    printf("{\"evidence\":\"cpu_idle_microbenchmark\",\"count\":20000,"
           "\"b_p50_ns\":%llu,\"b_p999_ns\":%llu,\"production_qualified\":false}\n",
           (unsigned long long)samples[10000], (unsigned long long)samples[19980]);
    return 0;
}
