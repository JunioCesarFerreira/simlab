/**
 * @file    metrics-packet.h
 * @brief   Report format and time helpers shared by the root and the nodes (firmware v2).
 *
 * Firmware v2 differs from v1 (rpl-udp-csma, rpl-udp-tsch) in how metrics are
 * obtained; see docs/markdown/FIRMWARE_METRICS.md. The points that shape this
 * header:
 *
 *  - Every field has a fixed width (stdint types). v1 used `unsigned int` and
 *    `unsigned long`, whose size depends on the platform, so the "packed"
 *    layout was not actually portable.
 *  - The packet starts with a type byte. v1 told a ping from a report by
 *    comparing the payload length, which breaks the moment two structures
 *    happen to have the same size.
 *  - The node reports raw cumulative counters. Everything derived from them
 *    (energy in mJ, per-interval deltas) is computed by the root, so a lost
 *    report loses a sample, never the energy of its interval.
 *  - The only latency is the one-way delay of the reports themselves, node to
 *    root, measured against the simulation clock (fw_sim_us below). v1 had
 *    three (rtt_latency, r2n_latency, n2r_latency), obtained from probes the
 *    root sent to the nodes; none of them could be compared across MACs.
 *
 * The same file is used, byte for byte, by rpl-udp-csma-v2 and rpl-udp-tsch-v2:
 * the application is identical and only the MAC layer (Makefile) differs.
 */

#ifndef METRICS_PACKET_H_
#define METRICS_PACKET_H_

#include <stdint.h>
#include "contiki.h"
#include "sys/clock.h"
#include "sys/rtimer.h"

#if MAC_CONF_WITH_TSCH
#include "net/mac/tsch/tsch.h"
#endif

#define SIMLAB_FIRMWARE_VERSION 2

/** Payload types, first byte of every UDP payload exchanged by this firmware. */
enum {
    PKT_REPORT = 1    /* node -> root: metrics report */
};

/**
 * Metrics report, node -> root. 52 bytes; keep it well under the 802.15.4
 * payload limit, a fragmented report would change what is being measured.
 *
 * Counters are cumulative since boot and are never reset.
 */
typedef struct {
    uint8_t  type;          /* PKT_REPORT                                                       */
    uint32_t seq;           /* reports generated so far, this one included (also "total_sent") */

    /* Simulation time (fw_sim_us) at the moment the report was generated, before it is handed
     * to the stack. The root subtracts it from the time the report arrives. */
    uint64_t send_us;

    /* Energest time in each state, microseconds since boot. */
    uint64_t cpu_us;
    uint64_t lpm_us;
    uint64_t tx_us;
    uint64_t rx_us;

    uint32_t bytes_tx;       /* UDP payload bytes this node handed to the stack (own reports)   */
    int16_t  rssi;           /* RSSI of the last frame the radio received before the report (dBm) */
    uint8_t  lqi;            /* LQI of that same frame (0-255)                                  */
} __attribute__((packed)) node_metrics_packet_t;

/* ------------------------------ time helpers ------------------------------ */

/*
 * fw_sim_us() reads the Cooja simulation clock. It is only a shared reference
 * inside Cooja; on real hardware every mote has its own rtimer and two
 * readings could not be subtracted. Refuse to build rather than produce
 * latencies that look fine and are not.
 */
#ifndef CONTIKI_TARGET_COOJA
#error "fw_sim_us() needs Cooja: only there do all motes share the rtimer clock"
#endif

/**
 * Simulation time in microseconds: the same value on every mote, whatever the MAC.
 *
 * Cooja writes the simulation time into every mote's rtimer before it runs
 * (ContikiClock.doActionsBeforeTick -> simRtimerCurrentTicks), and
 * rtimer_arch_now() returns it. clock_time() is not a shared reference: the
 * template starts every mote after a random delay of up to <motedelay_us>
 * (1 s) and its clock counts from its own boot, so two motes' clock_time()
 * readings differ by up to a second. That is why v1's r2n_latency and
 * n2r_latency, which subtracted two clock_time() readings on CSMA, were not
 * delays (see docs/markdown/FIRMWARE_METRICS.md).
 */
static inline uint64_t fw_sim_us(void)
{
    return ((uint64_t)RTIMER_NOW() * 1000000ULL) / RTIMER_SECOND;
}

/** Local clock in milliseconds. Starts at the mote's own boot, not shared. */
static inline uint64_t fw_local_ms(void)
{
    return ((uint64_t)clock_time() * 1000ULL) / CLOCK_SECOND;
}

/**
 * Network time in milliseconds, or (uint64_t)-1 when the mote cannot transmit.
 *
 * TSCH: it derives from the ASN, so it is the same on every associated mote;
 * it is (uint64_t)-1 while the mote is not associated.
 * CSMA: there is no shared time, this is the local clock.
 *
 * Used for the time column of the CSV (root_time_now) and the [RPL] log lines,
 * not for latency.
 */
static inline uint64_t fw_net_ms(void)
{
#if MAC_CONF_WITH_TSCH
    uint64_t ticks = tsch_get_network_uptime_ticks();
    if(ticks == (uint64_t)-1) {
        return (uint64_t)-1;   /* not associated */
    }
    return (ticks * 1000ULL) / CLOCK_SECOND;
#else
    return fw_local_ms();
#endif
}

#endif /* METRICS_PACKET_H_ */
