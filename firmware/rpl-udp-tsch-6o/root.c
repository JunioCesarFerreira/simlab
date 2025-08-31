/**
 * @file    root.c
 * @brief   RPL root for a Contiki-NG WSN collecting node metrics via UDP.
 *
 * This process periodically sends probe (ping) packets to known nodes.
 * Each node replies with execution and network metrics, including energy consumption,
 * latency, RSSI, LQI, and routing hops. The root logs and computes RTT-based latency.
 */

#include "contiki.h"
#include "net/routing/routing.h"
#include "net/netstack.h"
#include "net/ipv6/simple-udp.h"
#include "net/ipv6/uiplib.h"
#include "net/mac/tsch/tsch.h"
#include <stdio.h>

#include "metrics-packet.h"

//------------------------ Configuration Constants ----------------------------

#define UDP_CLIENT_PORT 8765
#define UDP_SERVER_PORT 5678
#define MAX_MOTES       100
#define SEND_INTERVAL   (10 * CLOCK_SECOND)

// Default TTL for IPv6/RPL — typically 64
#define DEFAULT_TTL_HOP_COUNTER 64

//------------------------ Structures and Global Variables --------------------

static struct etimer periodic_timer;
static ping_packet_t ping_pkt = { 0, 0 };
static struct simple_udp_connection udp_conn;

/**
 * @struct mote_t
 * @brief Internal structure for tracking node statistics.
 */
typedef struct {
    uip_ipaddr_t addr;
    uint32_t rx_count;
    uint32_t tx_count;
    uint32_t latency; 
    uint32_t last_volume_bytes;   /* para throughput incremental*/
    uint32_t last_rtt_ms;         /* RTT da última sondagem     */
    uint16_t index;
    char used;
} mote_t;

static mote_t motes[MAX_MOTES];

//------------------------ Utility Functions ----------------------------------

/**
 * @brief Compares two IPv6 addresses.
 */
static int compare_ipaddr(const uip_ipaddr_t *a, const uip_ipaddr_t *b) {
    return memcmp(a, b, sizeof(uip_ipaddr_t));
}

/**
 * @brief Searches for an existing mote entry or allocates a new one.
 */
static mote_t* rx_handle_mote_counters(const uip_ipaddr_t *sender_addr) {
    uint8_t found = 0;
    mote_t* ptr = NULL;
    for (int i = 0; i < MAX_MOTES; i++) {
        if (motes[i].used) {
            if (compare_ipaddr(sender_addr, &motes[i].addr) == 0) {
                motes[i].rx_count++;
                motes[i].index = i;
                found = 1;
                ptr = &motes[i];
                break;
            }
        } else {
            memcpy(&motes[i].addr, sender_addr, sizeof(uip_ipaddr_t));
            motes[i].rx_count = 1;
            motes[i].tx_count = 0;
            motes[i].used = 1;
            motes[i].index = i;
            found = 1;
            ptr = &motes[i];
            break;
        }
    }
    if (!found) {
        printf("No space for mote counter!\n");
    }
    return ptr;
}

/**
 * @brief Sends a ping packet to all known nodes.
 *
 * Each node should echo the same packet (PONG).
 * The root will use the RTT to estimate actual latency.
 */
static void send_ping_to_all_nodes(void) {
    ping_pkt.ping_seq++;
    ping_pkt.send_timestamp = tsch_get_network_uptime_ticks();

    for (int i = 0; i < MAX_MOTES; i++) {
        if (motes[i].used) {
            motes[i].tx_count++;
            simple_udp_sendto(&udp_conn, &ping_pkt, sizeof(ping_pkt), &motes[i].addr);
        }
    }
}

/**
 * @brief Prints metrics received from a node.
 *
 * Can be formatted as JSON or tabular output depending on preprocessor flags.
 */
static void metrics_print(char* addr_str, 
                          node_metrics_packet_t* metrics,
                          mote_t* scp_mote,
                          uint64_t now,
                          uint8_t hops,
                          uint16_t datalen) 
{
    /* ---- métricas primárias vindas do nó ---- */
    uint32_t cpu     = metrics->cpu_energy_mJ;
    uint32_t lpm     = metrics->lpm_energy_mJ;
    uint32_t tx_e    = metrics->radio_tx_energy_mJ;
    uint32_t rx_e    = metrics->radio_rx_energy_mJ;
    uint32_t etot    = cpu + lpm + tx_e + rx_e;
    uint32_t latency = metrics->from_root_to_node_latency;      /* ms */

    /* ---- métrica de volume/throughput calculada aqui ---- */
    uint32_t volume      = metrics->volume_bytes;
    uint32_t delta_bytes = volume - scp_mote->last_volume_bytes;
    scp_mote->last_volume_bytes = volume;

    uint32_t rtt_ms      = scp_mote->last_rtt_ms ? scp_mote->last_rtt_ms : 1; /* evita /0 */
    uint32_t throughput_bps = (delta_bytes * 8 * 1000UL) / rtt_ms;            /* Eq. 2.28 */

    /* ---- único log CSV/JSON ---- */
    printf(
        "{\"node\":\"%s\",\"cpu_mj\":%u,\"total_mj\":%u,\"latency_ms\":%u,\"volume_B\":%u,\"response_ms\":%u,\"throughput_bps\":%u,\"root_time_now\":%lu}\n",
        addr_str, cpu, etot, latency, volume, rtt_ms, throughput_bps, now);
}

//------------------------ UDP Receive Callback -------------------------------

/**
 * @brief Callback triggered when a UDP packet is received from a node.
 *
 * - If it is a PONG, compute RTT and update latency.
 * - If it is a metrics report, log it.
 */
static void udp_rx_callback(struct simple_udp_connection *c,
                            const uip_ipaddr_t *sender_addr,
                            uint16_t sender_port,
                            const uip_ipaddr_t *receiver_addr,
                            uint16_t receiver_port,
                            const uint8_t *data,
                            uint16_t datalen) 
{
    char addr_str[UIPLIB_IPV6_MAX_STR_LEN];
    uiplib_ipaddr_snprint(addr_str, sizeof(addr_str), sender_addr);

    //printf("UDP Packet received from %s\n", addr_str);
    
    mote_t* scp_mote = rx_handle_mote_counters(sender_addr);
    uint64_t now = tsch_get_network_uptime_ticks();

    if (datalen == sizeof(ping_packet_t)) {
        ping_packet_t *received_pkt = (ping_packet_t *)data;
        uint64_t rtt = now - received_pkt->send_timestamp;
        motes[scp_mote->index].latency = rtt / 2;  // Approximate one-way latency
        motes[scp_mote->index].last_rtt_ms = (uint32_t)(rtt);
    } 
    else if (datalen == sizeof(node_metrics_packet_t)) {
        node_metrics_packet_t *metrics = (node_metrics_packet_t *)data;
        uint8_t hops = UIP_IP_BUF->ttl;
        metrics_print(addr_str, metrics, scp_mote, now, hops, datalen);
    } 
    else {
        printf("Received bytes %d\n", datalen);
    }
}

//------------------------ Main Process ---------------------------------------

/**
 * @process udp_server_process
 * Main process for:
 *  - starting the RPL root
 *  - initializing the UDP socket
 *  - sending ping packets periodically to known nodes
 */
PROCESS(udp_server_process, "UDP server");
AUTOSTART_PROCESSES(&udp_server_process);

PROCESS_THREAD(udp_server_process, ev, data)
{
    PROCESS_BEGIN();

    printf("UDP Root process started\n");

    for (int i = 0; i < MAX_MOTES; i++) {
        motes[i].used = 0;
        motes[i].latency = 0;
    }

    NETSTACK_ROUTING.root_start();
    simple_udp_register(&udp_conn, UDP_SERVER_PORT, NULL, UDP_CLIENT_PORT, udp_rx_callback);
    etimer_set(&periodic_timer, SEND_INTERVAL);

    while(1) {
        PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));
        send_ping_to_all_nodes();
        etimer_reset(&periodic_timer);
    }

    PROCESS_END();
}
