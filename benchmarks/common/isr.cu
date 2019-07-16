#include <stdio.h>
#include <cuda.h>
#include "edge_helper.h"

__device__ void clock_block(clock_t clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
}

__global__ void ISR()
{
    int tid = threadIdx.x;
    int eventType = -1;
    //int smID = -1;
    //clock_block(1000);
    if( tid == 0 ) {
        EDGE_SET_FLAG(EDGE_BEGIN_INT);                  // Signal start of the interrupt
        eventType = EDGE_READ(EDGE_READ_EVENT_TYPE);    // Get the interrupt type
        //smID = EDGE_READ(EDGE_READ_SM_ID);              // Get the corresponding interrupt SM ID
        
        // EventType 0 is a NULL Event, just return
        //EDGE_SET_FLAG(EDGE_DELAY_INT);

        // Just delay for a bit, changes the number of cycles per interrupt
#if 0
        int x = 0;
        int nReadsDelay = EDGE_READ(EDGE_READ_NUM_DELAY_OPS);
        for( unsigned i=0; i<nReadsDelay; ++i ) {
            x += *(volatile int*)EDGE_READ_SM_ID;   // Volatile read to ensure read happens
            if( x > (unsigned)0xEF33FFFF )          // Some large number to break 
                break;
        }       
#endif

        // For debug NULL Event, should never get here
        if( eventType == EDGE_USER_EVENT || eventType == EDGE_WARMUP ) {
#if 0
            EDGE_WRITE(EDGE_CONFIG_EVENT, x);
#else
            EDGE_SET_FLAG(EDGE_CONFIG_EVENT);
#endif
            EDGE_SET_FLAG(EDGE_SCHEDULE_EVENT); // Schedule the event
            EDGE_SET_FLAG(EDGE_COMPLETE_INT);   // Signal completion of the interrupt
        } else if ( eventType == EDGE_RELEASE_BAR ) {
            asm("bar.sync 2;"); // Blanket release all barriers to start
            EDGE_SET_FLAG(EDGE_COMPLETE_INT);   // Signal completion of the interrupt
        }

    }
} 


