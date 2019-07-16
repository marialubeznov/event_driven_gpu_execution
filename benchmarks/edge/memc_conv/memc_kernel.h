#ifndef __MEMC_KERNE_H__
#define __MEMC_KERNEL_H__

#include "memc_shared.h"

#define NOT_GET(p) ( p[0] != 'g' || p[1] != 'e' || p[2] != 't' ) 

#define SET_VALUE_HDR(p)  p[0] = 'V'; p[1] = 'A'; p[2] = 'L'; p[3] = 'U'; \
                            p[4] = 'E'; p[5] = ' '; 

#define MAIN_WARP(t) (t==0)

#define VALUE_HDR_SIZE 6
#define ETH_ALEN 6

// TODO: The coalesced packet load is currently hardcoded to 256 requests per sub-group 
//       and 512 threads per group. This likely doesn't need to change, but it could be 
//       made configurable. 

// # pkts per group / [# threads in a group / WARP_SIZE ]
//
#define NUM_REQ_PER_LOOP    16
#define WARP_SIZE           32

// 42B hdr + 14B memc header = 56B per header / 4B per thread = 14 threads per header
#define THREAD_PER_HDR_COPY 14 



typedef struct _ether_header{
  u_int8_t  ether_dhost[ETH_ALEN];    /* destination eth addr    */
  u_int8_t  ether_shost[ETH_ALEN];    /* source ether addr    */
  u_int16_t ether_type;                /* packet type ID field    */
}ether_header;
// 14

typedef struct _ip_header {
  u_int8_t    version;            /* version */    // Version+ihl = 8 bits, so replace ihl with 8bit version
  //u_int32_t ihl:4;            /* header length */

  u_int8_t    tos;                /* type of service */
  u_int16_t    tot_len;            /* total length */
  u_int16_t    id;                    /* identification */
  u_int16_t    frag_off;            /* fragment offset field */
  u_int8_t    ttl;                /* time to live */
  u_int8_t    protocol;            /* protocol */
  u_int16_t    check;                /* checksum */

  u_int16_t saddr1;             /* source and dest address */
  u_int16_t saddr2;
  u_int16_t daddr1;
  u_int16_t daddr2;

}ip_header;
// 14 + 20 = 34
//
typedef struct _udp_header {
  u_int16_t    source;        /* source port */
  u_int16_t    dest;        /* destination port */
  u_int16_t    len;        /* udp length */
  u_int16_t    check;        /* udp checksum */
}udp_header;
// 14 + 20 + 8 = 42


typedef struct UdpPktHdr {
    ether_header    eh;
    ip_header       iph;
    udp_header      udph;
} UdpPktHdr;

typedef struct _memc_hdr_{
    u_int8_t hdr[14]; // Only 8 Bytes, but padding an extra 4 bytes for memcpy purposes
}memcHdr;

typedef struct _pkt_memc_hdr_{
    UdpPktHdr udp;
    memcHdr mch;
} MemcPktHdr;

typedef struct _pkt_info_{
    unsigned _valueIdx;                 // CPU VA pointer to found item
    unsigned _valueLength;    // Total length of response packet => Packet UDP header + "VALUE " + key + suffix + data (with "\r\n")
    int _hv;        // Hash value
    int _isGetReq;
    MemcPktHdr _nmch; // Packet header + memc 8 Byte header
} PktInfo;

typedef struct _key_ {
    unsigned _len;
    char _key[MAX_KEY_SIZE];
} Key;







#endif /* __MEMC_KERNEL_H__ */
