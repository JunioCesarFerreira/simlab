/**
 * @file    node.c
 * @brief   Sensor node (firmware v2): sends the root a report every ~10 s.
 *
 * The report carries the simulation time at which it was generated, plus
 * cumulative counters since boot: time spent in each Energest state, bytes
 * sent, the last RSSI/LQI. The node receives nothing from the application:
 * the root does not probe it, so the traffic is the reports and nothing else.
 *
 * What the node deliberately does NOT do, compared with v1:
 *  - it does not convert time into energy: the root does, with documented
 *    constants, so the model can be changed without re-running anything;
 *  - it does not reset its counters when it reports: a report that is lost
 *    costs one sample, not the interval it covered;
 *  - it does not compute latency: the root does, from send_us.
 *
 * Metric definitions: docs/markdown/FIRMWARE_METRICS.md
 */

#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "net/ipv6/uip-ds6.h"
#include "net/ipv6/uip-ds6-route.h"
#include "net/ipv6/uiplib.h"
#include "sys/energest.h"
#include "random.h"
#include <stdio.h>

#include "metrics-packet.h"

//------------------------ Configuration Constants ----------------------------

#define UDP_CLIENT_PORT   8765
#define UDP_SERVER_PORT   5678

#define BASE_INTERVAL (10 * CLOCK_SECOND)               ///< Base interval between reports
/* Drawn again for every period. v1 drew it once, so two nodes whose draws were
 * close collided in every period for the whole run. */
#define JITTER        (random_rand() % (CLOCK_SECOND))

//------------------------ State ----------------------------------------------

static struct simple_udp_connection udp_conn;

static uint32_t reports_sent    = 0;   /* reports generated (attempted)        */
static uint32_t bytes_tx_acc    = 0;   /* UDP payload bytes handed to the stack */

//------------------------ Utility Functions ----------------------------------

/** Energest time of one state, in microseconds, whatever ENERGEST_SECOND is. */
static uint64_t energest_us(energest_type_t type) {
    return (energest_type_time(type) * 1000000ULL) / ENERGEST_SECOND;
}

/** Prints the node's own link-local address (once, when it first reports). */
static void print_own_link_local(void) {
    uip_ds6_addr_t *ll_addr = uip_ds6_get_link_local(ADDR_PREFERRED);
    if (ll_addr != NULL) {
        char addr_str[UIPLIB_IPV6_MAX_STR_LEN];
        uiplib_ipaddr_snprint(addr_str, sizeof(addr_str), &ll_addr->ipaddr);
        printf("Sensor IPv6 = %s\n", addr_str);
    } else {
        printf("Sensor Loop: No link-local address available\n");
    }
}

/**
 * @brief Logs the RPL preferred parent whenever it changes (DODAG instrumentation).
 *
 * One line per parent switch: "[RPL] t=<ms> node=<ll> parent=<ll>". The time is
 * the node's network time (TSCH) or local clock (CSMA), in milliseconds.
 * pylib/rpl_dodag.py rebuilds the tree from these.
 */
static void log_parent_if_changed(void) {
    static uip_ipaddr_t last_parent;
    static uint8_t has_parent = 0;

    const uip_ipaddr_t *parent = uip_ds6_defrt_choose();
    if (parent == NULL) {
        return;
    }
    if (has_parent && uip_ipaddr_cmp(&last_parent, parent)) {
        return; /* parent unchanged */
    }
    uip_ipaddr_copy(&last_parent, parent);
    has_parent = 1;

    char self_str[UIPLIB_IPV6_MAX_STR_LEN] = "";
    char parent_str[UIPLIB_IPV6_MAX_STR_LEN];
    uip_ds6_addr_t *ll_addr = uip_ds6_get_link_local(ADDR_PREFERRED);
    if (ll_addr != NULL) {
        uiplib_ipaddr_snprint(self_str, sizeof(self_str), &ll_addr->ipaddr);
    }
    uiplib_ipaddr_snprint(parent_str, sizeof(parent_str), parent);

    uint64_t t = fw_net_ms();
    if (t == (uint64_t)-1) {
        t = fw_local_ms();
    }
    printf("[RPL] t=%lu node=%s parent=%s\n", (unsigned long)t, self_str, parent_str);
}

/** Fills the report from the current cumulative counters. Nothing is reset. */
static void fill_report(node_metrics_packet_t *r) {
    radio_value_t rssi = 0, lqi = 0;

    energest_flush();   /* bring the state that is running right now up to date */
    NETSTACK_RADIO.get_value(RADIO_PARAM_LAST_RSSI, &rssi);
    NETSTACK_RADIO.get_value(RADIO_PARAM_LAST_LINK_QUALITY, &lqi);

    r->type          = PKT_REPORT;
    r->seq           = reports_sent;
    r->send_us       = fw_sim_us();   /* last thing before the stack gets the packet */

    r->cpu_us        = energest_us(ENERGEST_TYPE_CPU);
    r->lpm_us        = energest_us(ENERGEST_TYPE_LPM);
    r->tx_us         = energest_us(ENERGEST_TYPE_TRANSMIT);
    r->rx_us         = energest_us(ENERGEST_TYPE_LISTEN);

    r->bytes_tx      = bytes_tx_acc;
    r->rssi          = (int16_t)rssi;
    r->lqi           = (uint8_t)lqi;
}

//------------------------ Main Process ---------------------------------------

PROCESS(udp_client_process, "UDP client");
AUTOSTART_PROCESSES(&udp_client_process);

PROCESS_THREAD(udp_client_process, ev, data)
{
    static struct etimer periodic_timer;
    static node_metrics_packet_t report;
    static uint8_t announced = 0;
    uip_ipaddr_t dest_ipaddr;

    PROCESS_BEGIN();

    simple_udp_register(&udp_conn, UDP_CLIENT_PORT, NULL, UDP_SERVER_PORT, NULL);
    etimer_set(&periodic_timer, BASE_INTERVAL + JITTER);

    NETSTACK_MAC.on();

    while(1) {
        PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));

        log_parent_if_changed();

        /* TSCH: a mote that is not associated cannot transmit, and the stack may
         * still hold a stale parent for a while; do not queue reports it cannot send. */
        if (fw_net_ms() != (uint64_t)-1 &&
            NETSTACK_ROUTING.node_is_reachable() &&
            NETSTACK_ROUTING.get_root_ipaddr(&dest_ipaddr)) {

            if (!announced) {
                print_own_link_local();
                announced = 1;
            }

            reports_sent++;
            bytes_tx_acc += sizeof(report);   /* counted before filling: the report includes itself */
            fill_report(&report);

            simple_udp_sendto(&udp_conn, &report, sizeof(report), &dest_ipaddr);
        } else {
            printf("Not reachable yet\n");
        }

        etimer_set(&periodic_timer, BASE_INTERVAL + JITTER);
    }

    PROCESS_END();
}
