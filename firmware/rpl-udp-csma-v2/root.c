/**
 * @file    root.c
 * @brief   RPL root (firmware v2): turns each node report into one JSON line.
 *
 * For every metrics report received, print one JSON line on the serial port
 * (Cooja mote 1). pylib/cooja_builder/csv_converter.py turns the lines into
 * the CSV that the data conversion evaluates.
 *
 * The root sends nothing to the nodes. v1 probed every node with a ping to
 * measure latency; those probes were extra traffic that the MACs handled very
 * differently (the CSMA neighbour queue dropped them, the single TSCH shared
 * cell could not carry them), so the latency they produced said more about the
 * probe than about the application. Latency is now a property of the reports
 * themselves: the node stamps each with the simulation time, the root
 * subtracts it from the time the report arrives.
 *
 * All derived quantities (energy, per-report deltas, latency) are computed
 * here from the node's raw counters. Metric definitions:
 * docs/markdown/FIRMWARE_METRICS.md
 */

#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/uip.h"
#include "net/ipv6/simple-udp.h"
#include "net/ipv6/uiplib.h"
#include <stdio.h>
#include <string.h>

#include "metrics-packet.h"

//------------------------ Configuration Constants ----------------------------

#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678

/* Nodes the root can keep counters for. v1 had 100 and dereferenced a null
 * pointer for the 101st. */
#ifndef ROOT_MAX_MOTES
#define ROOT_MAX_MOTES 200
#endif

/*
 * Energy model: E[mJ] = I[mA] * V[V] * t[s]. The currents are the datasheet
 * figures v1 already used (MSP430 active / LPM, CC2420 TX / RX); v1 multiplied
 * them by time and called the result mJ, which left out the supply voltage.
 * Override any of them in project-conf.h to change the model; nothing in the
 * nodes depends on them.
 */
#ifndef ENERGY_SUPPLY_V
#define ENERGY_SUPPLY_V  3.0
#endif
#ifndef ENERGY_CPU_MA
#define ENERGY_CPU_MA    1.8
#endif
#ifndef ENERGY_LPM_MA
#define ENERGY_LPM_MA    0.0545
#endif
#ifndef ENERGY_TX_MA
#define ENERGY_TX_MA     17.4
#endif
#ifndef ENERGY_RX_MA
#define ENERGY_RX_MA     19.7
#endif

#define U(x) ((unsigned long)(x))

//------------------------ Per-node state -------------------------------------

typedef struct {
    uip_ipaddr_t addr;

    uint32_t reports_rx;        /* "server_received": reports that reached the root */
    uint32_t report_bytes_rx;   /* "server_bytes_rx": cumulative report payload     */

    /* newest report seen and its cumulative Energest times, to turn them into per-report deltas */
    uint32_t last_seq;
    uint64_t last_cpu_us, last_lpm_us, last_tx_us, last_rx_us;
} mote_t;

static mote_t motes[ROOT_MAX_MOTES];
static uint16_t n_motes = 0;
static uint8_t  table_full_warned = 0;

static struct simple_udp_connection udp_conn;

//------------------------ Node table -----------------------------------------

/** Finds the node, or adds it. NULL when the table is full. */
static mote_t *mote_get(const uip_ipaddr_t *addr) {
    for (uint16_t i = 0; i < n_motes; i++) {
        if (memcmp(&motes[i].addr, addr, sizeof(uip_ipaddr_t)) == 0) {
            return &motes[i];
        }
    }
    if (n_motes >= ROOT_MAX_MOTES) {
        if (!table_full_warned) {
            printf("No space for mote counter!\n");
            table_full_warned = 1;
        }
        return NULL;
    }
    mote_t *m = &motes[n_motes++];
    memset(m, 0, sizeof(*m));
    memcpy(&m->addr, addr, sizeof(uip_ipaddr_t));
    return m;
}

//------------------------ Reports --------------------------------------------

/** Growth of a cumulative counter since the newest report seen so far. */
static uint64_t delta_us(uint64_t current, uint64_t *last) {
    uint64_t d = (current >= *last) ? current - *last : 0;
    *last = current;
    return d;
}

static void handle_report(const uip_ipaddr_t *sender_addr, const uint8_t *data, uint16_t datalen) {
    /* Read the clock first: nothing below may add to the delay being measured. */
    uint64_t rx_us = fw_sim_us();

    if (datalen != sizeof(node_metrics_packet_t)) {
        return;
    }
    mote_t *m = mote_get(sender_addr);
    if (m == NULL) {
        return;
    }
    node_metrics_packet_t r;
    memcpy(&r, data, sizeof(r));   /* the payload may not be aligned */

    m->reports_rx++;
    m->report_bytes_rx += datalen;

    /* One-way delay of this report, node to root: application to application, so it
     * includes queueing, MAC access (CSMA backoff or the wait for a TSCH slot),
     * retransmissions and every hop. Both readings are the simulation clock, so no
     * assumption about the path or about synchronised clocks is involved. */
    char latency_s[24];
    if (rx_us >= r.send_us) {
        snprintf(latency_s, sizeof(latency_s), "%.3f", (double)(rx_us - r.send_us) / 1000.0);
    } else {
        snprintf(latency_s, sizeof(latency_s), "null");   /* cannot happen; never invent a number */
    }

    /* hops = links crossed to reach the root. The TTL leaves the node at UIP_TTL and each
     * forwarder takes one off, so a direct child arrives with UIP_TTL: 1 link. */
    uint8_t ttl = UIP_IP_BUF->ttl;
    unsigned int hops = (ttl <= UIP_TTL) ? (unsigned int)(UIP_TTL - ttl) + 1U : 1U;

    /* Energy of the interval since the newest report that arrived before this one. A report
     * that arrives after a later one (TSCH reorders) covers an interval that later report
     * already accounted for, so it adds none: counting it again would double the interval,
     * and treating its smaller counters as a reboot would add the whole run. */
    uint64_t d_cpu = 0, d_lpm = 0, d_tx = 0, d_rx = 0;
    if (r.seq > m->last_seq) {
        m->last_seq = r.seq;
        d_cpu = delta_us(r.cpu_us, &m->last_cpu_us);
        d_lpm = delta_us(r.lpm_us, &m->last_lpm_us);
        d_tx  = delta_us(r.tx_us,  &m->last_tx_us);
        d_rx  = delta_us(r.rx_us,  &m->last_rx_us);
    }
    double cpu_mj = ENERGY_CPU_MA * ENERGY_SUPPLY_V * (double)d_cpu / 1e6;
    double lpm_mj = ENERGY_LPM_MA * ENERGY_SUPPLY_V * (double)d_lpm / 1e6;
    double tx_mj  = ENERGY_TX_MA  * ENERGY_SUPPLY_V * (double)d_tx  / 1e6;
    double rx_mj  = ENERGY_RX_MA  * ENERGY_SUPPLY_V * (double)d_rx  / 1e6;

    /* The time this line's energy was spent in: the MCU is always either active or in LPM,
     * so CPU + LPM time is the node's elapsed time over the same interval (Cooja has no deep
     * LPM). Summed over lines it is the node-time the energy covers, so
     * sum(total_energy_mj) / sum(energy_time_ms) is a mean power, comparable between runs
     * whose nodes reported for different lengths of time. */
    double energy_time_ms = (double)(d_cpu + d_lpm) / 1000.0;

    /* Time of this line: the root's network time (TSCH) or local clock (CSMA). */
    uint64_t now = fw_net_ms();
    if (now == (uint64_t)-1) {
        now = fw_local_ms();
    }

    char addr_str[UIPLIB_IPV6_MAX_STR_LEN];
    uiplib_ipaddr_snprint(addr_str, sizeof(addr_str), sender_addr);

    uint32_t lost = (r.seq >= m->reports_rx) ? r.seq - m->reports_rx : 0;

    printf("{\"node\":\"%s\", \"fw_version\":%d, "
           "\"cpu_energy_mj\":%.3f, \"lpm_energy_mj\":%.3f, "
           "\"radio_tx_energy_mj\":%.3f, \"radio_rx_energy_mj\":%.3f, "
           "\"total_energy_mj\":%.3f, \"energy_time_ms\":%.3f, "
           "\"tx_time_ms\":%.3f, \"rx_time_ms\":%.3f, "
           "\"total_sent\":%lu, \"bytes_tx\":%lu, "
           "\"latency\":%s, \"hops\":%u, \"lqi\":%u, \"rssi\":%d, "
           "\"server_received\":%lu, \"server_bytes_rx\":%lu, \"reports_lost\":%lu, "
           "\"root_time_now\":%lu}\n",
           addr_str, SIMLAB_FIRMWARE_VERSION,
           cpu_mj, lpm_mj, tx_mj, rx_mj, cpu_mj + lpm_mj + tx_mj + rx_mj, energy_time_ms,
           (double)r.tx_us / 1000.0, (double)r.rx_us / 1000.0,
           U(r.seq), U(r.bytes_tx),
           latency_s, hops, (unsigned int)r.lqi, (int)r.rssi,
           U(m->reports_rx), U(m->report_bytes_rx), U(lost),
           U(now));
}

//------------------------ UDP Receive Callback -------------------------------

static void udp_rx_callback(struct simple_udp_connection *c,
                            const uip_ipaddr_t *sender_addr,
                            uint16_t sender_port,
                            const uip_ipaddr_t *receiver_addr,
                            uint16_t receiver_port,
                            const uint8_t *data,
                            uint16_t datalen)
{
    if (datalen >= 1 && data[0] == PKT_REPORT) {
        handle_report(sender_addr, data, datalen);
    }
}

//------------------------ Main Process ---------------------------------------

PROCESS(udp_server_process, "UDP server");
AUTOSTART_PROCESSES(&udp_server_process);

PROCESS_THREAD(udp_server_process, ev, data)
{
    PROCESS_BEGIN();

    printf("UDP Root process started\n");

    NETSTACK_ROUTING.root_start();
    simple_udp_register(&udp_conn, UDP_SERVER_PORT, NULL, UDP_CLIENT_PORT, udp_rx_callback);

    /* Everything happens in the receive callback; keep the process alive. */
    while(1) {
        PROCESS_WAIT_EVENT();
    }

    PROCESS_END();
}
