#ifndef _COMMON_H_
#define _COMMON_H_

#include <netinet/in_systm.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <net/ethernet.h>     /* the L2 protocols */
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <linux/if_packet.h>

//#include <linux/ipv6.h>
#define NUM_REQUESTS_PER_BATCH 32
#define IPV4_REG_NUM 16

struct pkt_hdr_normal 
{
    /////////////////// Ethernet /////////////////// 
    // Ethernet is 6 bytes. Break down into first half / second half (Store 4 bytes per half for performance)
    u_int32_t ether_dhost_1;
    u_int32_t ether_dhost_2;
    u_int32_t ether_shost_1;
    u_int32_t ether_shost_2;
    u_int32_t ether_type;

    /////////////////// IP /////////////////// 
    u_int32_t ip_version;
    u_int32_t ip_tos; 
    u_int32_t ip_tot_len; 
    u_int32_t ip_id;
    u_int32_t ip_frag_off; 
    u_int32_t ip_ttl; 
    u_int32_t ip_protocol; 
    u_int32_t ip_check; 
    u_int32_t ip_saddr;
    u_int32_t ip_daddr; 

    /////////////////// UDP ///////////////////
    u_int32_t udp_source;
    u_int32_t udp_dest;
    u_int32_t udp_len;
    u_int32_t udp_check;
};




struct pkt_hdr_batch 
{
    /////////////////// Ethernet /////////////////// 
    // Ethernet is 6 bytes. Break down into first half / second half (Store 4 bytes per half for performance)
    u_int32_t ether_dhost_1[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_dhost_2[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_shost_1[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_shost_2[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_type[NUM_REQUESTS_PER_BATCH];

    /////////////////// IP /////////////////// 
    u_int32_t ip_version[NUM_REQUESTS_PER_BATCH];
    u_int32_t ip_tos[NUM_REQUESTS_PER_BATCH]; 
    u_int32_t ip_tot_len[NUM_REQUESTS_PER_BATCH]; 
    u_int32_t ip_id[NUM_REQUESTS_PER_BATCH];
    u_int32_t ip_frag_off[NUM_REQUESTS_PER_BATCH]; 
    u_int32_t ip_ttl[NUM_REQUESTS_PER_BATCH]; 
    u_int32_t ip_protocol[NUM_REQUESTS_PER_BATCH]; 
    u_int32_t ip_check[NUM_REQUESTS_PER_BATCH]; 
    u_int32_t ip_saddr[NUM_REQUESTS_PER_BATCH];
    u_int32_t ip_daddr[NUM_REQUESTS_PER_BATCH]; 

    /////////////////// UDP ///////////////////
    u_int32_t udp_source[NUM_REQUESTS_PER_BATCH];
    u_int32_t udp_dest[NUM_REQUESTS_PER_BATCH];
    u_int32_t udp_len[NUM_REQUESTS_PER_BATCH];
    u_int32_t udp_check[NUM_REQUESTS_PER_BATCH];
};

/////////////// IPv4
struct ip_header 
{
  u_int8_t	version;			/* version */			// Small hack, version+ihl = 8 bits, so replace ihl with 8bit version

  u_int8_t	tos;			/* type of service */
  u_int16_t	tot_len;			/* total length */
  u_int16_t	id;			/* identification */
  u_int16_t	frag_off;			/* fragment offset field */
  u_int8_t	ttl;			/* time to live */
  u_int8_t	protocol;			/* protocol */
  u_int16_t	check;			/* checksum */

  u_int32_t saddr;
  u_int32_t daddr;	/* source and dest address */
};

/*
 * Udp protocol header.
 * Per RFC 768, September, 1981.
 */
struct udp_header 
{
    u_int16_t source;		/* source port */
    u_int16_t dest;		/* destination port */
    u_int16_t len;		/* udp length */
    u_int16_t check;		/* udp checksum */
};

struct pkt_hdr 
{
    ether_header eh;
    struct ip_header iph;
    udp_header uh;
} __attribute__((packed));


/////////////// IPv6
//struct ipv6hdr 
//{
//	uint8_t version;
//    uint8_t flow_lbl[3];
//    uint16_t payload_len;
//    uint8_t nexthdr;
//    uint8_t hop_limit;
//    struct in6_addr saddr;
//    struct in6_addr daddr;
//};

/**
 * IPv6 Header: size = 40 bits
 */
struct ipv6_hdr {
    uint32_t vtc_flow;     /**< IP version, traffic class & flow label. */
    uint16_t payload_len;  /**< IP packet length - includes sizeof(ip_header). */
    uint8_t  proto;        /**< Protocol, next header. */
    uint8_t  hop_limits;   /**< Hop limits. */
    uint8_t  src_addr[16]; /**< IP address of source host. */
    uint8_t  dst_addr[16]; /**< IP address of destination host(s). */
} __attribute__((__packed__));

struct ipv6_pkt_hdr 
{
	ether_header eh;
	struct ipv6_hdr iph;
	udp_header uh;	
} __attribute__((packed));

struct ipv6_pkt_hdr_normal 
{
    /////////////////// Ethernet /////////////////// 
    // Ethernet is 6 bytes. Break down into first half / second half (Store 4 bytes per half for performance)
    u_int32_t ether_dhost_1;
    u_int32_t ether_dhost_2;
    u_int32_t ether_shost_1;
    u_int32_t ether_shost_2;
    u_int32_t ether_type;

    /////////////////// IP /////////////////// 
    uint32_t ip_vtc_flow;     /**< IP version, traffic class & flow label. */
    uint32_t ip_payload_len;  /**< IP packet length - includes sizeof(ip_header). */
    uint32_t ip_proto;        /**< Protocol, next header. */
    uint32_t ip_hop_limits;   /**< Hop limits. */

    uint32_t ip_saddr1;
    uint32_t ip_saddr2;
    uint32_t ip_saddr3;
    uint32_t ip_saddr4;
    uint32_t ip_daddr1;
    uint32_t ip_daddr2;
    uint32_t ip_daddr3;
    uint32_t ip_daddr4;

    /////////////////// UDP ///////////////////
    u_int32_t udp_source;
    u_int32_t udp_dest;
    u_int32_t udp_len;
    u_int32_t udp_check;
};

struct ipv6_pkt_hdr_swizzle 
{
    /////////////////// Ethernet /////////////////// 
    // Ethernet is 6 bytes. Break down into first half / second half (Store 4 bytes per half for performance)
    u_int32_t ether_dhost_1[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_dhost_2[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_shost_1[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_shost_2[NUM_REQUESTS_PER_BATCH];
    u_int32_t ether_type[NUM_REQUESTS_PER_BATCH];

    /////////////////// IP /////////////////// 
    uint32_t ip_vtc_flow[NUM_REQUESTS_PER_BATCH];     /**< IP version, traffic class & flow label. */
    uint32_t ip_payload_len[NUM_REQUESTS_PER_BATCH];  /**< IP packet length - includes sizeof(ip_header). */
    uint32_t ip_proto[NUM_REQUESTS_PER_BATCH];        /**< Protocol, next header. */
    uint32_t ip_hop_limits[NUM_REQUESTS_PER_BATCH];   /**< Hop limits. */

    uint32_t ip_saddr1[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_saddr2[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_saddr3[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_saddr4[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_daddr1[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_daddr2[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_daddr3[NUM_REQUESTS_PER_BATCH];
    uint32_t ip_daddr4[NUM_REQUESTS_PER_BATCH];

    /////////////////// UDP ///////////////////
    u_int32_t udp_source[NUM_REQUESTS_PER_BATCH];
    u_int32_t udp_dest[NUM_REQUESTS_PER_BATCH];
    u_int32_t udp_len[NUM_REQUESTS_PER_BATCH];
    u_int32_t udp_check[NUM_REQUESTS_PER_BATCH];
};





#endif 
