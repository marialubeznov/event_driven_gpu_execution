#ifndef __MEMC_SHARED_H__
#define __MEMC_SHARED_H__

#include <stdint.h>
//#define DEBUG

//#define BUFFER_SIZE 1500
#define BUFFER_SIZE 128

#define RX_BUFFER_SIZE BUFFER_SIZE
#define TX_BUFFER_SIZE BUFFER_SIZE

#define HDR_SIZE 64
#define KEY_SIZE 16
#define VAL_SIZE 8

#define PKT_STRIDE 72

// HASH TABLES
#define MAX_KEY_SIZE 32
#define MAX_VAL_SIZE 8

#define HASH_POWER  14
#define KEY_HASH_MASK 0x000000FF

#define HASH_SET_ASSOC   8

#define hashsize(n) ((unsigned int)1<<(n))
#define hashmask(n) (hashsize(n)-1)
#define numsets() (hashsize(HASH_POWER) / HASH_SET_ASSOC)

#define MEMC_REG_NUM 24

#define UNLOCKED            0    // No lock set
#define SHARED_LOCK         1    // GET request(s) have the item locked
#define PRIVATE_LOCK        (-1)    // SET request has the item locked. Only a single PRIVATE_LOCK can be obtained at

#define UDP_PORT        9960

#define G_HTONS(val) (u_int16_t) ((((u_int16_t)val >> 8) & 0x00FF ) | (((u_int16_t)val << 8) & 0xFF00) )
#define G_NTOHS(val) (G_HTONS(val))

#define G_HTONL(val) (u_int32_t) ( (((u_int32_t)val & 0xFF000000) >> 24 ) | \
                                   (((u_int32_t)val & 0x00FF0000) >> 8  ) | \
                                   (((u_int32_t)val & 0x0000FF00) << 8  ) | \
                                   (((u_int32_t)val & 0x000000FF) << 24))

#define G_NTOHL(val) (G_HTONL(val))

/*************************************/
#define MAX_THREADS_PER_BLOCK   32 //256 // Number of threads per request group
#define NUM_REQUESTS_PER_GROUP  16 //256 // Do not change
/*************************************/

#define NUM_REQUESTS_PER_BATCH  32 //512

#define NUM_THREADS_PER_GROUP   NUM_REQUESTS_PER_GROUP*2
#define NUM_GROUPS NUM_REQUESTS_PER_BATCH / NUM_REQUESTS_PER_GROUP


/***********************************************/
/***********************************************/
// Bob Jenkin's hash from baseline Memcached
/***********************************************/
/***********************************************/
#define rot(x,k) (((x)<<(k)) ^ ((x)>>(32-(k))))

#define memcached_mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

#define final(a,b,c) \
{ \
   c ^= b; c -= rot(b,14); \
   a ^= c; a -= rot(c,11); \
   b ^= a; b -= rot(a,25); \
   c ^= b; c -= rot(b,16); \
   a ^= c; a -= rot(c,4);  \
   b ^= a; b -= rot(a,14); \
   c ^= b; c -= rot(b,24); \
}


/*******************************************************************************/
/*******************************************************************************/
/******************************* Structures ************************************/
/*******************************************************************************/
/*******************************************************************************/
typedef unsigned rel_time_t;


typedef struct GpuGetPkt {
    unsigned char _pkt[RX_BUFFER_SIZE];
} GpuGetPkt;

typedef struct _gpuPrimaryHashtable_
{
    unsigned _valueIdx;
    rel_time_t _lastAccessedTime;
    unsigned _valid;
    unsigned _keyHash;          // 8-bit key hash - using 4 bytes to keep everything aligned
    unsigned _keyLength;
    unsigned _valueLength;
    char _key[MAX_KEY_SIZE];
} gpuPrimaryHashtable;


typedef struct SetRequest {
	uint16_t _keyLength;
	uint16_t _valueLength;
    uint8_t _key[MAX_KEY_SIZE];
    uint8_t _value[MAX_VAL_SIZE];
} SetRequest;

typedef struct MemcValue 
{
    unsigned _valid;
    uint8_t _value[MAX_VAL_SIZE]; 
} MemcValue;


#endif /* __MEMC_SHARED_H__ */
