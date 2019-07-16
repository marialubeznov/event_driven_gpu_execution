#ifndef __EDGE_HELPER_H__
#define __EDGE_HELPER_H__

typedef int* EDGE_PTR;
typedef unsigned long long EDGE_ADDR_TYPE;


#define NULL_EVENT_ID 0
#define WARMUP_EVENT_ID 1

enum GPUEventType {
    EDGE_NULL = 0,
    EDGE_WARMUP,
    EDGE_ENABLE, 
    EDGE_DISABLE,
    EDGE_USER_EVENT,
    EDGE_RELEASE_BAR,
    EDGE_NUM_EVENT_TYPES
};

////////////////////////////////////////////////////////////////////////////////
/// Function macros 
////////////////////////////////////////////////////////////////////////////////
#define EDGE_READ(addr)         *(EDGE_PTR)addr
#define EDGE_VOL_READ(addr)     *(volatile EDGE_PTR)addr
#define EDGE_WRITE(addr, val)   *(EDGE_PTR)addr = val;
#define EDGE_SET_FLAG(addr)     *(EDGE_PTR)addr = 1;
#define EDGE_OP(addr)           (((size_t)addr >= (size_t)EDGE_RANGE_BEGIN) && \
                                 ((size_t)addr <  (size_t)EDGE_RANGE_END))

////////////////////////////////////////////////////////////////////////////////
/// EDGE memory mapped address range
////////////////////////////////////////////////////////////////////////////////
#define EDGE_RANGE_BEGIN        (EDGE_ADDR_TYPE)0x00001000
#define EDGE_RANGE_END          (EDGE_ADDR_TYPE)0x00002000

////////////////////////////////////////////////////////////////////////////////
/// Memory mapped addresses
////////////////////////////////////////////////////////////////////////////////
#define EDGE_BEGIN_INT          (EDGE_ADDR_TYPE)0x00001000
#define EDGE_COMPLETE_INT       (EDGE_ADDR_TYPE)0x00001004
#define EDGE_READ_EVENT_TYPE    (EDGE_ADDR_TYPE)0x00001008
#define EDGE_READ_EVENT_ID      (EDGE_ADDR_TYPE)0x0000100C

#define EDGE_SCHEDULE_EVENT     (EDGE_ADDR_TYPE)0x00001010
#define EDGE_CONFIG_EVENT       (EDGE_ADDR_TYPE)0x00001014
#define EDGE_COMPLETE_EVENT     (EDGE_ADDR_TYPE)0x00001018

#define EDGE_GET_PARAM_BASE_ADDR (EDGE_ADDR_TYPE)0x00001100

////////////////////////////////////////////////////////////////////////////////
/// DEBUG
////////////////////////////////////////////////////////////////////////////////
#define EDGE_READ_SM_ID         (EDGE_ADDR_TYPE)0x00001F00
#define EDGE_DELAY_INT          (EDGE_ADDR_TYPE)0x00001F04

#define EDGE_READ_NUM_DELAY_OPS (EDGE_ADDR_TYPE)0x00001F08

#endif /* __EDGE_HELPER_H__ */
