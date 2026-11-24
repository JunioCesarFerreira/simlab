#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_

#define ENERGEST_CONF_ON 1

/*
 * TSCH only (CSMA builds do not read it): one shared cell every 3 timeslots
 * instead of Contiki-NG's default of 7, the value of Contiki-NG's own 6TiSCH
 * example. With 7, 26 nodes reporting every 10 s congest the shared cell: the
 * queues toward the relays fill up, RPL's DAO-ACKs time out, nodes repair
 * locally and stop reporting for minutes. With 3 the same instances deliver
 * what CSMA delivers. Measurements: docs/markdown/FIRMWARE_METRICS.md, section 8.
 */
#define TSCH_SCHEDULE_CONF_DEFAULT_LENGTH 3

/*
 * Neighbour and route tables sized for SimLab's networks instead of the Cooja
 * platform's 300. It changes no output and makes Cooja several times faster.
 * On every tick Cooja copies the mote's whole RAM (.data and .bss) into the
 * native library and back, because all motes of a type share one library. With
 * 300 entries the .bss was 272-285 KB, 158 KB of it TSCH neighbour queues, and
 * the copy took 87% of the simulation's time under TSCH; with 64 it is
 * 118-131 KB.
 *
 * The output is identical while no table fills, which holds up to 63 motes,
 * root included. All neighbour tables share one pool of addresses: a node can
 * hold every other mote (an entry outlives a neighbour that moves away) plus
 * TSCH's two virtual neighbours (EB and broadcast), motes + 1 in all. The
 * root's source-routing table holds every node of the DODAG and the root itself.
 * Past 63 motes a full table evicts a neighbour and runs diverge from the
 * 300-entry build. Measurements: firmware/README.md, "Simulation speed".
 */
#define NBR_TABLE_CONF_MAX_NEIGHBORS 64
#define NETSTACK_MAX_ROUTE_ENTRIES 64

#endif /* PROJECT_CONF_H_ */
