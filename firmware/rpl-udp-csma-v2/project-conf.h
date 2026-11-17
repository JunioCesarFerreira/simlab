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

#endif /* PROJECT_CONF_H_ */
