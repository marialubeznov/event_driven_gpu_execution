#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "memc_kernel.h"
#include "memc_shared.h"

extern __device__ gpuPrimaryHashtable* gPrimaryHashtable;
extern __device__ int* gLocks;
extern __device__ MemcValue* gValueHeap;
extern __device__ int* gDebugPtr;

// Main functions
__global__ void initDataStructures();
__global__ void memcGetKernel(char* rxBuffers, char* txBuffers, unsigned nPkts);
__global__ void initHashTable(SetRequest* inputs, unsigned N, unsigned timestamp);

// Helper functions
__device__ unsigned int hash(char const * key, size_t length, const unsigned int initval);
__device__ int fast_memcmp(const void *key1, const void *key2, int num);
__device__ int short_memcmp(const void *key1, const void *key2, int num);
__device__ int partial_cksum(unsigned char *buf, unsigned nbytes, int sum);
__device__ int cksum_hdr_len_only(unsigned char *buf, int sum);
__device__ int g_in_cksum(unsigned char *buf, unsigned nbytes, int sum);
__device__ int wrapsum (u_int32_t sum) ;


template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
                                    const int numImages, const int numFilters,
                                    const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                    const int moduleStride, const int numModulesY, const int numModulesX, const int imgStride,
                                    const float scaleTargets, const float scaleOutputs, const int conv);


template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const int conv);

__global__ void emptyKernel()
{

}

/*******************************************************************************/
/*******************************************************************************/
/****************************** GPU Globals ************************************/
/*******************************************************************************/
/*******************************************************************************/
__global__ void initDataStructures()
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int htSize = hashsize(HASH_POWER);
    if( tid < htSize ) {
        gpuPrimaryHashtable* mEntry = &gPrimaryHashtable[tid];
        mEntry->_valueIdx = tid;
    }
}

// This function expects a single warp (or at the least, a single thread block)
__global__ void simpleMemcGetKernel(char* packets, unsigned nPkts, int* reg_buffer)
{
    int tid = threadIdx.x;

    int ehs = sizeof(ether_header);
    int ips = sizeof(ip_header);
    int udps = sizeof(udp_header);
    unsigned networkSize = ehs + ips + udps;
    int *reqPtr = NULL;
    int *pktHdrPtr = NULL;
    char *pktHdr = NULL;
                                             
    ether_header *eh;
    ip_header *iph;
    udp_header *udp;
    char *payload;
    char *key;
    unsigned nKey = 0;

    int count = 0;
    u_int16_t check = 0;

    // Not storing in shared memory any more
    PktInfo pi;
    int itemFound = 0;

    pktHdr = packets + tid*RX_BUFFER_SIZE;
    eh = (ether_header*)pktHdr;
    iph = (ip_header *)(pktHdr + ehs);
    udp = (udp_header *)(pktHdr + ehs + ips);
    payload = pktHdr + networkSize + 8; 

    pi._isGetReq = 1;
    
    if(G_NTOHS(udp->dest) != UDP_PORT){
        pi._isGetReq = 0;
    }

    // Verify Checksum
    // Lower 4-bits of version is the ip_header length (ihl)
    if( pi._isGetReq && iph->check != 0 ) {
        check = wrapsum(g_in_cksum((unsigned char *)iph, (iph->version & 0x0F)<<2, 0));
        if( check != 0 ) { 
            pi._isGetReq = 0;
        }
    }

    if( pi._isGetReq ) {
        if( NOT_GET(payload) )
            pi._isGetReq = 0;
    }
                                           
    key = payload+4; // Move passed "get "
    if( pi._isGetReq ) {
        // key is guaranteed to be a minimum of 16 bytes
        count = 16;
                                                                                      
        // Then search for the end condition
        while( (key[count] != '\r') || (key[count+1] != '\n') ){
            count++;
        }
                                                                                      
        // Check if key is too large
        if(count >= MAX_KEY_SIZE){
            pi._isGetReq = 0;
        }
    }
                                                                                      
    // Set the key length
    nKey = count;

    // Now do the Memcached lookup
    int found = 0; 
    volatile char *key_t;
    unsigned key_hash_t;
    volatile gpuPrimaryHashtable *m_gph;
    unsigned set_index;
    unsigned set_hv_index;
    unsigned key_hash = 0;
    int is_locked = 0;
    volatile int old_lock_val = -1;
    volatile int new_lock_val = 0;
    volatile int new_old_lock_val = 0;

    // Compute the hash
    unsigned hv = hash(key, nKey, 0);
    key_hash = hv & KEY_HASH_MASK; // Compute the hash mask for this key

    // Compute the set index for the hash and the corresponding index into the hash table
    unsigned num_sets = numsets(); 
    set_index = hv % num_sets;                  // Calculate the set index for this hash value
    set_hv_index = set_index * HASH_SET_ASSOC;    // Move to the correct location in the hash table for this set

    // Soft mutex for each GET request. Multiple shared_locks, only single private_lock.
    // Grab the shared lock for the set
    while( !is_locked ) {
        old_lock_val = *(volatile int*)(gLocks + set_index);
        if(old_lock_val != -1){ // TEST
            new_lock_val = old_lock_val+1;
            new_old_lock_val = atomicCAS(&gLocks[set_index], old_lock_val, new_lock_val); // and TEST and SET
            if( new_old_lock_val == old_lock_val ) {
                is_locked = 1;
            }
        }
    }

    // Set initial response length if item isn't found
    pi._valueLength = 0;

    /************************ Critical Section ************************/
    for( unsigned i=0; i<HASH_SET_ASSOC; ++i ){
        m_gph = (volatile gpuPrimaryHashtable *)&gPrimaryHashtable[set_hv_index + i];

        if( m_gph->_valid > 0 ) {
            key_t = (volatile char *)m_gph->_key;

            // New - First check key hash. If equal, then do key comparison. Otherwise, no way they're equal.
            key_hash_t = m_gph->_keyHash;
            if( key_hash == key_hash_t ) {
                //found = fast_memcmp((const void *)key, (const void *)key_t, nKey);
                found = short_memcmp((const void *)key, (const void *)key_t, nKey);

                if( found ) {
                    pi._valueIdx = m_gph->_valueIdx;
                    pi._valueLength = m_gph->_valueLength; 
                    itemFound = 1;
                    m_gph->_valid = 2; // Update hash table entry to say that last access was a GET request
                    break;
                }
            }
        }
    }

    // Unlock the set
    atomicSub(&gLocks[set_index], 1);
    /************************ End Critical Section ************************/

    // Copy in "VALUE "
    //SET_VALUE_HDR(pi[logicalId]._nmch.mch.hdr);
    // Need to verify if this will work? 
    payload = pktHdr + networkSize; 
    SET_VALUE_HDR(payload); 
    char* txPtr;
    txPtr = payload + VALUE_HDR_SIZE;

    //atomicAdd(&gDebugPtr[1], nKey); 

    // Populate response packet
    if (itemFound == 1) {
        atomicAdd(&gDebugPtr[1], 1); 
        for (unsigned i=0; i<pi._valueLength; ++i) {
            //txPtr[i] = gValueHeap[pi._valueIdx]._value[i];
            txPtr[i] = 0xFF;
        }
    } else {
        txPtr[0] = 0x66;
    }

    // Finally, create the response packet header
    u_int16_t  ether_swap;
    u_int16_t ip_addr1;
    u_int16_t ip_addr2;
    u_int16_t udp_port;

    // Swap ether
    for(unsigned i=0; i<ETH_ALEN; i+=2){
        ether_swap = *(uint16_t*)(eh->ether_shost + i);
        *(uint16_t*)(eh->ether_shost + i) = *(uint16_t*)(eh->ether_dhost + i);
        *(uint16_t*)(eh->ether_dhost + i) = ether_swap;
    }

    unsigned pktLength = sizeof(UdpPktHdr) + VALUE_HDR_SIZE + pi._valueLength;

    // Swap IP
    ip_addr1 = iph->saddr1;
    ip_addr2 = iph->saddr2;
    iph->saddr1 = iph->daddr1;
    iph->saddr2 = iph->daddr2;
    iph->daddr1 = ip_addr1;
    iph->daddr2 = ip_addr2;
    iph->tot_len = G_HTONS(pktLength - sizeof(ether_header));
    iph->check = 0;

    // Swap UDP port
    udp_port = udp->source;
    udp->source = udp->dest;
    udp->dest = udp_port;
    udp->len = G_HTONS(pktLength - sizeof(ether_header) - sizeof(ip_header));
    udp->check = 0;

    iph->check = wrapsum(g_in_cksum((unsigned char *)iph, (iph->version & 0x0F)<<2, 0));

}

__global__ void simpleMemcGetKernel_save_regs(char* packets, unsigned nPkts, int* reg_buffer)
{
    int tid = threadIdx.x;

    for (int i=0; i<MEMC_REG_NUM; i++) {
        reg_buffer[tid * MEMC_REG_NUM + i] = tid;
    }

    int ehs = sizeof(ether_header);
    int ips = sizeof(ip_header);
    int udps = sizeof(udp_header);
    unsigned networkSize = ehs + ips + udps;
    int *reqPtr = NULL;
    int *pktHdrPtr = NULL;
    char *pktHdr = NULL;
                                             
    ether_header *eh;
    ip_header *iph;
    udp_header *udp;
    char *payload;
    char *key;
    unsigned nKey = 0;

    int count = 0;
    u_int16_t check = 0;

    // Not storing in shared memory any more
    PktInfo pi;
    int itemFound = 0;

    pktHdr = packets + tid*RX_BUFFER_SIZE;
    eh = (ether_header*)pktHdr;
    iph = (ip_header *)(pktHdr + ehs);
    udp = (udp_header *)(pktHdr + ehs + ips);
    payload = pktHdr + networkSize + 8; 

    pi._isGetReq = 1;
    
    if(G_NTOHS(udp->dest) != UDP_PORT){
        pi._isGetReq = 0;
    }

    // Verify Checksum
    // Lower 4-bits of version is the ip_header length (ihl)
    if( pi._isGetReq && iph->check != 0 ) {
        check = wrapsum(g_in_cksum((unsigned char *)iph, (iph->version & 0x0F)<<2, 0));
        if( check != 0 ) { 
            pi._isGetReq = 0;
        }
    }

    if( pi._isGetReq ) {
        if( NOT_GET(payload) )
            pi._isGetReq = 0;
    }
                                           
    key = payload+4; // Move passed "get "
    if( pi._isGetReq ) {
        // key is guaranteed to be a minimum of 16 bytes
        count = 16;
                                                                                      
        // Then search for the end condition
        while( (key[count] != '\r') || (key[count+1] != '\n') ){
            count++;
        }
                                                                                      
        // Check if key is too large
        if(count >= MAX_KEY_SIZE){
            pi._isGetReq = 0;
        }
    }
                                                                                      
    // Set the key length
    nKey = count;

    // Now do the Memcached lookup
    int found = 0; 
    volatile char *key_t;
    unsigned key_hash_t;
    volatile gpuPrimaryHashtable *m_gph;
    unsigned set_index;
    unsigned set_hv_index;
    unsigned key_hash = 0;
    int is_locked = 0;
    volatile int old_lock_val = -1;
    volatile int new_lock_val = 0;
    volatile int new_old_lock_val = 0;

    // Compute the hash
    unsigned hv = hash(key, nKey, 0);
    key_hash = hv & KEY_HASH_MASK; // Compute the hash mask for this key

    // Compute the set index for the hash and the corresponding index into the hash table
    unsigned num_sets = numsets(); 
    set_index = hv % num_sets;                  // Calculate the set index for this hash value
    set_hv_index = set_index * HASH_SET_ASSOC;    // Move to the correct location in the hash table for this set

    // Soft mutex for each GET request. Multiple shared_locks, only single private_lock.
    // Grab the shared lock for the set
    while( !is_locked ) {
        old_lock_val = *(volatile int*)(gLocks + set_index);
        if(old_lock_val != -1){ // TEST
            new_lock_val = old_lock_val+1;
            new_old_lock_val = atomicCAS(&gLocks[set_index], old_lock_val, new_lock_val); // and TEST and SET
            if( new_old_lock_val == old_lock_val ) {
                is_locked = 1;
            }
        }
    }

    // Set initial response length if item isn't found
    pi._valueLength = 0;

    /************************ Critical Section ************************/
    for( unsigned i=0; i<HASH_SET_ASSOC; ++i ){
        m_gph = (volatile gpuPrimaryHashtable *)&gPrimaryHashtable[set_hv_index + i];

        if( m_gph->_valid > 0 ) {
            key_t = (volatile char *)m_gph->_key;

            // New - First check key hash. If equal, then do key comparison. Otherwise, no way they're equal.
            key_hash_t = m_gph->_keyHash;
            if( key_hash == key_hash_t ) {
                //found = fast_memcmp((const void *)key, (const void *)key_t, nKey);
                found = short_memcmp((const void *)key, (const void *)key_t, nKey);

                if( found ) {
                    pi._valueIdx = m_gph->_valueIdx;
                    pi._valueLength = m_gph->_valueLength; 
                    itemFound = 1;
                    m_gph->_valid = 2; // Update hash table entry to say that last access was a GET request
                    break;
                }
            }
        }
    }

    // Unlock the set
    atomicSub(&gLocks[set_index], 1);
    /************************ End Critical Section ************************/

    // Copy in "VALUE "
    //SET_VALUE_HDR(pi[logicalId]._nmch.mch.hdr);
    // Need to verify if this will work? 
    payload = pktHdr + networkSize; 
    SET_VALUE_HDR(payload); 
    char* txPtr;
    txPtr = payload + VALUE_HDR_SIZE;

    //atomicAdd(&gDebugPtr[1], nKey); 

    // Populate response packet
    if (itemFound == 1) {
        atomicAdd(&gDebugPtr[1], 1); 
        for (unsigned i=0; i<pi._valueLength; ++i) {
            //txPtr[i] = gValueHeap[pi._valueIdx]._value[i];
            txPtr[i] = 0xFF;
        }
    } else {
        txPtr[0] = 0x66;
    }

    // Finally, create the response packet header
    u_int16_t  ether_swap;
    u_int16_t ip_addr1;
    u_int16_t ip_addr2;
    u_int16_t udp_port;

    // Swap ether
    for(unsigned i=0; i<ETH_ALEN; i+=2){
        ether_swap = *(uint16_t*)(eh->ether_shost + i);
        *(uint16_t*)(eh->ether_shost + i) = *(uint16_t*)(eh->ether_dhost + i);
        *(uint16_t*)(eh->ether_dhost + i) = ether_swap;
    }

    unsigned pktLength = sizeof(UdpPktHdr) + VALUE_HDR_SIZE + pi._valueLength;

    // Swap IP
    ip_addr1 = iph->saddr1;
    ip_addr2 = iph->saddr2;
    iph->saddr1 = iph->daddr1;
    iph->saddr2 = iph->daddr2;
    iph->daddr1 = ip_addr1;
    iph->daddr2 = ip_addr2;
    iph->tot_len = G_HTONS(pktLength - sizeof(ether_header));
    iph->check = 0;

    // Swap UDP port
    udp_port = udp->source;
    udp->source = udp->dest;
    udp->dest = udp_port;
    udp->len = G_HTONS(pktLength - sizeof(ether_header) - sizeof(ip_header));
    udp->check = 0;

    iph->check = wrapsum(g_in_cksum((unsigned char *)iph, (iph->version & 0x0F)<<2, 0));

    //restore regs
    for (int i=0; i<MEMC_REG_NUM; i++) {
        tid = reg_buffer[tid * MEMC_REG_NUM + i];
    }
}



/*******************************************************************************/
/*******************************************************************************/
/*********************** Main Memcached GET kernel *****************************/
/*******************************************************************************/
/*******************************************************************************/
__global__ void memcGetKernel(char* rxBuffers, char* txBuffers, unsigned nPkts)
{
	int globalId = blockDim.x*blockDim.y*blockIdx.x + blockDim.x*threadIdx.y + threadIdx.x;
	int localId = blockDim.x*threadIdx.y + threadIdx.x;
	int logicalId = threadIdx.x; 
	int groupId = blockIdx.x;
	int threadType = threadIdx.y;
    int debugIdx = logicalId + groupId*256; // First block is 0->256, second block is 256->512

    int timestamp = 0; // Move to kernel parameter

	__shared__ PktInfo pi[NUM_REQUESTS_PER_GROUP];
	__shared__ int itemFound[NUM_REQUESTS_PER_GROUP];

#ifdef SHARED_KEY
    __shared__ Key keys[NUM_REQUESTS_PER_GROUP];
#else
    Key keys;
#endif

    unsigned long long rxBufferStart = (unsigned long long)rxBuffers + (groupId * RX_BUFFER_SIZE * NUM_REQUESTS_PER_GROUP);

    /**********************************************************/
    /********************** PARSE PACKET **********************/
    /**********************************************************/
    //parsePkt( rxBufferStart, localId, logicalId, threadType, pi, &keys[logicalId] );
    // __device__ void parsePkt(unsigned long long rxBufferStart, int localId, int logicalId, int threadType, PktInfo* pi, Key* gKey)
    int ehs = sizeof(ether_header);
    int ips = sizeof(ip_header);
    int udps = sizeof(udp_header);
    unsigned networkSize = ehs + ips + udps;

    int *reqPtr = NULL;
    int *pktHdrPtr = NULL;
    char *pktHdr = NULL;

    ether_header *eh;
    ip_header *iph;
    udp_header *udp;
    char *payload;
    char *key;
    int count = 0;
    u_int16_t check = 0;

    int reqInd = (int)(localId / WARP_SIZE); // Which warp do you belong to?
    reqInd *= NUM_REQ_PER_LOOP;
    int wTid = localId % WARP_SIZE;
    int maskedInd = wTid % THREAD_PER_HDR_COPY;

    for( unsigned i=0; i<NUM_REQ_PER_LOOP; ++i ) {
        reqPtr = (int *)( rxBufferStart + ((reqInd + i)*RX_BUFFER_SIZE) );
        pktHdrPtr = (int *)(&pi[reqInd + i]._nmch);
        pktHdrPtr[maskedInd] = reqPtr[maskedInd];
    }	

    itemFound[logicalId] = 0;
    pi[logicalId]._isGetReq = 1;

	__syncthreads();

    if( MAIN_WARP(threadType) ) {
        pktHdr = (char *)&pi[logicalId]._nmch;
        iph = (ip_header *)(pktHdr + ehs);
        udp = (udp_header *)(pktHdr + ehs + ips);

        payload = (char *)(rxBufferStart + (logicalId*RX_BUFFER_SIZE));

        payload += (networkSize+8);

        if(G_NTOHS(udp->dest) != UDP_PORT){
            pi[logicalId]._isGetReq = 0;
        }

        // Verify Checksum
        // Lower 4-bits of version is the ip_header length (ihl)
        if( pi[logicalId]._isGetReq && iph->check != 0 ) {
            check = wrapsum(g_in_cksum((unsigned char *)iph, (iph->version & 0x0F)<<2, 0));
            if( check != 0 ) { 
                pi[logicalId]._isGetReq = 0;
            }
        }
        
        //gDebugPtr[debugIdx] = pi[logicalId]._isGetReq;//itemFound[logicalId]; //pi[logicalId]._valueLength;

        if( pi[logicalId]._isGetReq ) {
            if( NOT_GET(payload) )
                pi[logicalId]._isGetReq = 0;
        }

        key = payload+4; // Move passed "get "
        if( pi[logicalId]._isGetReq ) {
            // key is guaranteed to be a minimum of 16 bytes, load in 16 bytes as shorts.
            for( unsigned i=0; i<8; i++, count += 2 ) {
#ifdef SHARED_KEY
                ((short *)(keys[logicalId]._key))[i] = ((short *)(key))[i];
#else
                ((short *)(keys._key))[i] = ((short *)(key))[i];
#endif
            }

            // Then load in the rest, searching for the end condition
            while( (key[count] != '\r') || (key[count+1] != '\n') ){
#ifdef SHARED_KEY                
                keys[logicalId]._key[count] = key[count];
#else
                keys._key[count] = key[count];
#endif
                count++;
            }

            // Check if key is too large
            if(count >= MAX_KEY_SIZE){
                pi[logicalId]._isGetReq = 0;
            }
        }

        // Set the key length
#ifdef SHARED_KEY        
        keys[logicalId]._len = count;
#else
        keys._len = count;
#endif
    }

    /**********************************************************/
    /**********************************************************/
    /**********************************************************/

    __syncthreads();
    //__threadfence();

    /***************************************************************************/
    /********************** PROCESS GET & CREATE RESPONSE **********************/
    /***************************************************************************/    
    //__device__ int processGetRequest( PktInfo* pi, rel_time_t time, Key* gKey )

    if( pi[logicalId]._isGetReq ) {
        if( MAIN_WARP(threadType) ) {
            //itemFound[liogicalId] = processGetRequest( &pi[logicalId], timestamp, &keys[logicalId] );
            int found = 0; 
#ifdef SHARED_KEY            
            size_t nkey = keys[logicalId]._len;
            char *key = keys[logicalId]._key;
#else
            size_t nkey = keys._len;
            char *key = keys._key;
#endif
            volatile char *key_t;
            unsigned key_hash_t;
            volatile gpuPrimaryHashtable *m_gph;
            unsigned set_index;
            unsigned set_hv_index;
            unsigned key_hash = 0;
            int is_locked = 0;
            volatile int old_lock_val = -1;
            volatile int new_lock_val = 0;
            volatile int new_old_lock_val = 0;

            // Compute the hash
            unsigned hv = hash(key, nkey, 0);
            key_hash = hv & KEY_HASH_MASK; // Compute the hash mask for this key

            // Compute the set index for the hash and the corresponding index into the hash table
            unsigned num_sets = numsets(); 
            set_index = hv % num_sets;                  // Calculate the set index for this hash value
            set_hv_index = set_index * HASH_SET_ASSOC;    // Move to the correct location in the hash table for this set


            // Soft mutex for each GET request. Multiple shared_locks, only single private_lock.
            // Grab the shared lock for the set
            while( !is_locked ) {
                //old_lock_val = gLocks[set_index];
                old_lock_val = *(volatile int*)(gLocks + set_index);
                if(old_lock_val != -1){ // TEST
                    new_lock_val = old_lock_val+1;
                    new_old_lock_val = atomicCAS(&gLocks[set_index], old_lock_val, new_lock_val); // and TEST and SET
                    if( new_old_lock_val == old_lock_val ) {
                        is_locked = 1;
                    }
                }
            }

            // Set initial response length if item isn't found
            //pi->_pktLength = TX_BUFFER_SIZE;
            pi[logicalId]._valueLength = 0;

            /************************ Critical Section ************************/
            for( unsigned i=0; i<HASH_SET_ASSOC; ++i ){
                m_gph = (volatile gpuPrimaryHashtable *)&gPrimaryHashtable[set_hv_index + i];

                if( m_gph->_valid > 0 ) {
                    key_t = (volatile char *)m_gph->_key;

                    // New - First check key hash. If equal, then do key comparison. Otherwise, no way they're equal.
                    key_hash_t = m_gph->_keyHash;
                    if( key_hash == key_hash_t ) {
                        found = fast_memcmp((const void *)key, (const void *)key_t, nkey);

                        if( found ) {
                            pi[logicalId]._valueIdx = m_gph->_valueIdx;
                            pi[logicalId]._valueLength = m_gph->_valueLength; 
                            itemFound[logicalId] = 1;

                            //m_gph->_lastAccessedTime = timestamp; // Possible Race Condition if multiple GETs updating this concurrently, but don't care who wins
                            m_gph->_valid = 2; // Update hash table entry to say that last access was a GET request

                            break;
                        }
                    }
                }
            }

            // Unlock the set
            atomicSub(&gLocks[set_index], 1);
            /************************ End Critical Section ************************/
        
        } else {
            //__device__ void createResponseHdr(PktInfo* pi)
            //createResponseHdr( &pi[logicalId] );
            // Elements to swap
            u_int8_t  ether_swap;
            u_int16_t ip_addr1;
            u_int16_t ip_addr2;
            u_int16_t udp_port;

            char *header = (char *)(&pi[logicalId]._nmch);

            eh = (ether_header *)header;
            iph = (ip_header *)&header[14];
            udp = (udp_header *)&header[34];

            // Swap ether
            uint16_t t;
            for(unsigned i=0; i<ETH_ALEN; i+=2){
                t = *(uint16_t*)(eh->ether_shost + i);
                *(uint16_t*)(eh->ether_shost + i) = *(uint16_t*)(eh->ether_dhost + i);
                *(uint16_t*)(eh->ether_dhost + i) = t;
            }

            // Swap IP
            ip_addr1 = iph->saddr1;
            ip_addr2 = iph->saddr2;
            iph->saddr1 = iph->daddr1;
            iph->saddr2 = iph->daddr2;
            iph->daddr1 = ip_addr1;
            iph->daddr2 = ip_addr2;
            iph->check = 0;

            // Swap UDP port
            udp_port = udp->source;
            udp->source = udp->dest;
            udp->dest = udp_port;
            udp->check = 0;

            // Calculate an initial partial checksum without the IP header length field.
            // This will be added in afterwards
            iph->check = partial_cksum((unsigned char *)iph, 4*(iph->version & 0x0F), 0);

            // Copy in "VALUE "
            SET_VALUE_HDR(pi[logicalId]._nmch.mch.hdr);
        }
    }

    /**********************************************************/
    /**********************************************************/
    /**********************************************************/

    __syncthreads();

    /***********************************************************************/
    /********************** POPULATE RESPONSE PACKETS **********************/
    /***********************************************************************/    

    //__device__ void populateResponsePkt(char* tx, PktInfo* pi, int localId, int logicalId, int groupId, int* itemFound, unsigned threadType, Key* gKey)
    //populateResponsePkt( txBuffers, pi, localId, logicalId, groupId, itemFound, threadType, &keys[logicalId] );

    char* txPtr = NULL;
    char* tx = txBuffers + ( groupId * TX_BUFFER_SIZE * NUM_REQUESTS_PER_GROUP );

    if( MAIN_WARP(threadType) ) {
        if( itemFound[logicalId] == 1 ) {
            // Store the actual value in the response packet
            txPtr = tx + (logicalId * TX_BUFFER_SIZE); // Point to correct Tx buffer
            txPtr += (sizeof(UdpPktHdr) + VALUE_HDR_SIZE); // Move to right spot in the Tx buffer

            for( unsigned i=0; i<pi[logicalId]._valueLength; ++i ) {
                txPtr[i] = gValueHeap[pi[logicalId]._valueIdx]._value[i];
            }

            //atomicAdd(&gDebugPtr[1], 1); // How many evictions do we have?
        }
        
    } else {
        // Compute the proper packet length and checksum
        char *header = (char *)(&pi[logicalId]._nmch);
        ip_header *iph = (ip_header *)&header[sizeof(ether_header)];
        udp_header *uh = (udp_header *)&header[sizeof(ether_header) + sizeof(ip_header)];

        unsigned pktLength = sizeof(UdpPktHdr) + VALUE_HDR_SIZE + pi[logicalId]._valueLength;
        // Update response packet lengths and compute IP checksum
        iph->tot_len = G_HTONS(( pktLength - sizeof(ether_header)) );
        uh->len = G_HTONS( (pktLength - sizeof(ether_header) - sizeof(ip_header)) );

        // Already computed a partial checksum without the IP header length field.
        // Add the updated length to the checksum. 
        iph->check = wrapsum(cksum_hdr_len_only((unsigned char *)iph, iph->check));
    }

    __syncthreads();

    reqInd = (int)(localId / WARP_SIZE); // Which warp this thread belongs to
    reqInd *= NUM_REQ_PER_LOOP;
    wTid = localId % WARP_SIZE;
    maskedInd = wTid % (THREAD_PER_HDR_COPY-2);

    int idx = logicalId + groupId*256;
    //gDebugPtr[idx] = itemFound[logicalId]; //pi[logicalId]._valueLength;
    
    // Finally, store packet response headers from shared to global memory
    for( unsigned i=0; i<NUM_REQ_PER_LOOP; ++i ) {
        int* pktHdrPtr = (int*)(&pi[reqInd + i]._nmch);
        int* txPtrBlock = (int*)(tx + ((reqInd+i) * TX_BUFFER_SIZE));
        txPtrBlock[maskedInd] = pktHdrPtr[maskedInd];
    }
}


/*******************************************************************************/
/*******************************************************************************/
/*********************** Main Memcached SET kernel *****************************/
/*******************************************************************************/
/*******************************************************************************/
// Basically the SET kernel, but performs N sets concurrently. 
__global__ void initHashTable(SetRequest* inputs, unsigned N, unsigned timestamp)
{
    int ctaTid = blockDim.x*blockIdx.x + threadIdx.x;
    int wid = ctaTid / 32; // Warp id
    int lid = ctaTid % 32; // Local thread id within a warp
    
    int localWid = threadIdx.x / 32;

	bool isLocked = false;

    unsigned insertHvIdx = (unsigned)-1;
    unsigned oldestItemHv = (unsigned)-1;
    int oldestItemTime = 0xFFFFFFFF;
    int freeFound = 0;
    unsigned hv = 0;
    unsigned setIdx;
    unsigned setHvIdx;
    int setKeyHash;

    __shared__ unsigned warpInsertHvIdx[8];

    volatile gpuPrimaryHashtable* tempGph;


	SetRequest* setReq = &inputs[wid];	
    if( (wid < N) && (lid == 0) ) { // Only 1 thread per warp
       
        hv = hash( (const char*)setReq->_key, setReq->_keyLength, 0);
		setIdx = hv % numsets();
        setHvIdx = setIdx * HASH_SET_ASSOC;
        setKeyHash = hv & KEY_HASH_MASK; // Calcualte the hash mask

	    // Obtain the lock
	    while( !isLocked ) {
            int old_lock_val = atomicCAS(&gLocks[setIdx], UNLOCKED, PRIVATE_LOCK);
            if( old_lock_val == UNLOCKED ) {
                isLocked = true;
            }
        }

        for( unsigned i=0; i<HASH_SET_ASSOC; ++i ) {
            tempGph = (volatile gpuPrimaryHashtable *)&gPrimaryHashtable[setHvIdx + i]; // Index into the hashtable at this set
            if( tempGph->_valid > 0 ){ // This hash location is already occupied, check the next location
                // First check key hash. If equal, then do key comparison. Otherwise, no way they're equal.
                if( tempGph->_keyHash == setKeyHash ) {
                    // If key hash matches, check complete key
                    int ret = fast_memcmp((const void *)setReq->_key, (const void *)tempGph->_key, setReq->_keyLength);

                    if( ret == 1 ) {
                        // If matches, select this entry to overwrite. Set matching key-value pair to evict.
                        // This is required to ensure correct ordering on the CPU post processing

                        // Treat this the same as an LRU evict
                        oldestItemTime = tempGph->_lastAccessedTime;
                        oldestItemHv = (setHvIdx + i);
                        freeFound = 0;
                        break;
                    }
                }

                // If no hit, update LRU status for this set                
                if( (tempGph->_lastAccessedTime < oldestItemTime) || (oldestItemHv == -1) ) {
                    oldestItemTime = tempGph->_lastAccessedTime;
                    oldestItemHv = (setHvIdx + i);
                }
            } else {
                // No need to search the whole set if an invalid entry is found
                freeFound = 1;
                insertHvIdx = (setHvIdx + i);
                break;
            }
        }

        if( !freeFound ) {
            insertHvIdx = oldestItemHv;
            atomicAdd(&gDebugPtr[0], 1); // How many evictions do we have?
        }

        warpInsertHvIdx[localWid] = insertHvIdx;
    }

    insertHvIdx = warpInsertHvIdx[localWid];

    // Can't sync here yet, still have a lock that warps may be blocked on

    volatile gpuPrimaryHashtable* gph = (volatile gpuPrimaryHashtable*)&gPrimaryHashtable[insertHvIdx];
   
    // Block memory copy with all threads in the warp (max key size of 16 with this code)
#if 0
    unsigned* tmpSrc = (unsigned*)setReq->_key;
    unsigned* tmpDst = (unsigned*)gph->_key;
    if( lid < (setReq->_keyLength / sizeof(unsigned)) ) {
        tmpDst[lid] = tmpSrc[lid];
    }

    tmpSrc = (unsigned*)setReq->_value;
    tmpDst = (unsigned*)gValueHeap[gph->_valueIdx]._value;
    if( lid < (setReq->_valueLength / sizeof(unsigned)) ) {
        tmpDst[lid] = tmpSrc[lid];
    }
#else
    char* tmpSrc = (char*)setReq->_key;
    char* tmpDst = (char*)gph->_key;
    if( lid < setReq->_keyLength ) {
        tmpDst[lid] = tmpSrc[lid];
    }

    char* ctmpSrc = (char*)setReq->_value;
    char* ctmpDst = (char*)gValueHeap[gph->_valueIdx]._value;
    if( lid < setReq->_valueLength ) {
        ctmpDst[lid] = ctmpSrc[lid];
    }
#endif
    // Single thread per warp
    
    if( (wid < N) && (lid == 0) ) {
        if( !freeFound ) {
            // EVICT THE ITEM -- anything to do?
        }

        // Update the hash table entry
        gph->_keyHash = setKeyHash;
        gph->_keyLength = setReq->_keyLength;
        gph->_valueLength = setReq->_valueLength;
        gph->_lastAccessedTime = (unsigned)timestamp;
        gph->_valid = 1;
    
        // Unlock the set
        atomicCAS(&gLocks[setIdx], -1, UNLOCKED); 
       
        //gDebugPtr[wid] = (int)&gValueHeap[gph->_valueIdx]._value; //setReq->_keyLength; //gph->_valueIdx;
    }
}



// NOTE: This requires key lengths to be in increments 4 bytes
__device__ int fast_memcmp(const void *key1, const void *key2, int num){

    const unsigned *p1 = (const unsigned* )key1;
    const unsigned *p2 = (const unsigned* )key2;

    int main_loop = num / sizeof(int);

    for(unsigned i=0; i<main_loop; i++){
        if(*(p1+i) != *(p2+i)){
            return 0;
        }
    }

    return 1;
}

__device__ int short_memcmp(const void *key1, const void *key2, int num){

    const short *p1 = (const short*)key1;
    const short *p2 = (const short*)key2;

    int main_loop = num / sizeof(short);

    for(unsigned i=0; i<main_loop; i++){
        if(*(p1+i) != *(p2+i)){
            return 0;
        }
    }

    return 1;
}


__device__ unsigned int hash(char const * key,  /* the key to hash */
                    size_t length,              /* length of the key */
                    const unsigned int initval  /* initval */){

  unsigned int a,b,c;                                          /* internal state */
  union { const char *ptr; size_t i; } u;     /* needed for Mac Powerbook G4 */

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  u.ptr = key;
  if (((u.i & 0x3) == 0)) {
       unsigned int const * k = ( unsigned int const *)key;

    /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
    while (length > 12)
    {
      a += k[0];
      b += k[1];
      c += k[2];
      memcached_mix(a,b,c);
      length -= 12;
      k += 3;
    }

    switch(length)
    {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;  /* zero length strings require no mixing */
    }
 } else if (((u.i & 0x1) == 0)) {
      unsigned short const * k = (unsigned short const *)key;                           /* read 16-bit chunks */
      unsigned char const * k8;

   /*--------------- all but last block: aligned reads and different mixing */
    while (length > 12)
    {
      a += k[0] + (((unsigned int)k[1])<<16);
      b += k[2] + (((unsigned int)k[3])<<16);
      c += k[4] + (((unsigned int)k[5])<<16);
      memcached_mix(a,b,c);
      length -= 12;
      k += 6;
    }

    /*----------------------------- handle the last (probably partial) block */
    k8 = (  unsigned char const *)k;
    switch(length)
    {
    case 12: c+=k[4]+(((unsigned int)k[5])<<16);
             b+=k[2]+(((unsigned int)k[3])<<16);
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 11: c+=((unsigned int)k8[10])<<16;     /* @fallthrough */
    /* no break */
    case 10: c+=k[4];                       /* @fallthrough@ */
             b+=k[2]+(((unsigned int)k[3])<<16);
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 9 : c+=k8[8];                      /* @fallthrough */
    case 8 : b+=k[2]+(((unsigned int)k[3])<<16);
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 7 : b+=((unsigned int)k8[6])<<16;      /* @fallthrough */
    case 6 : b+=k[2];
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 5 : b+=k8[4];                      /* @fallthrough */
    case 4 : a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 3 : a+=((unsigned int)k8[2])<<16;      /* @fallthrough */
    case 2 : a+=k[0];
             break;
    case 1 : a+=k8[0];
             break;
    case 0 : return c;  /* zero length strings require no mixing */
    }

  } else {                        /* need to read the key one byte at a time */
       unsigned char const * k = ( unsigned char const *)key;

    /*--------------- all but the last block: affect some 32 bits of (a,b,c) */
    while (length > 12)
    {
      a += k[0];
      a += ((unsigned int)k[1])<<8;
      a += ((unsigned int)k[2])<<16;
      a += ((unsigned int)k[3])<<24;
      b += k[4];
      b += ((unsigned int)k[5])<<8;
      b += ((unsigned int)k[6])<<16;
      b += ((unsigned int)k[7])<<24;
      c += k[8];
      c += ((unsigned int)k[9])<<8;
      c += ((unsigned int)k[10])<<16;
      c += ((unsigned int)k[11])<<24;
      memcached_mix(a,b,c);
      length -= 12;
      k += 12;
    }

    /*-------------------------------- last block: affect all 32 bits of (c) */
    switch(length)                   /* all the case statements fall through */
    {
    case 12: c+=((unsigned int)k[11])<<24;
    case 11: c+=((unsigned int)k[10])<<16;
    case 10: c+=((unsigned int)k[9])<<8;
    case 9 : c+=k[8];
    case 8 : b+=((unsigned int)k[7])<<24;
    case 7 : b+=((unsigned int)k[6])<<16;
    case 6 : b+=((unsigned int)k[5])<<8;
    case 5 : b+=k[4];
    case 4 : a+=((unsigned int)k[3])<<24;
    case 3 : a+=((unsigned int)k[2])<<16;
    case 2 : a+=((unsigned int)k[1])<<8;
    case 1 : a+=k[0];
             break;
    case 0 : return c;  /* zero length strings require no mixing */
    }
  }

  final(a,b,c);
  return c;             /* zero length strings require no mixing */
}


///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Helper functions ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


// This checksum skips the ip_header length field, but adds up everything else.
// Later we can add in the length. Used to overlap independent computation to
// reduce processing latency
__device__ int partial_cksum(unsigned char *buf, unsigned nbytes, int sum)
{
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    if(i != 2){ // Bytes 2&3 are the IP header length field, skip it
        sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + i)));
        /* Add carry. */
        if(sum > 0xFFFF)
            sum -= 0xFFFF;
    }
  }

  /* If there's a single byte left over, checksum it, too.   Network
     byte order is big-endian, so the remaining byte is the high byte. */
  if(i < nbytes) {
    sum += buf [i] << 8;
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  return sum;
}


// Only add up the ip header length once we know the response packet size
__device__ int cksum_hdr_len_only(unsigned char *buf, int sum)
{
    sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + 2)));
    if(sum > 0xFFFF)
        sum -= 0xFFFF;

    return sum;
}

// Full checksum
/*
 * Checksum routine for Internet Protocol family headers (C Version)
 *
 * Borrowed from DHCPd
 */
__device__ int g_in_cksum(unsigned char *buf, unsigned nbytes, int sum) 
{
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + i)));
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  /* If there's a single byte left over, checksum it, too.   Network
     byte order is big-endian, so the remaining byte is the high byte. */
  if(i < nbytes) {
    sum += buf [i] << 8;
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  return sum;
}

/* ******************************************* */

__device__ int wrapsum (u_int32_t sum) 
{
  sum = ~sum & 0xFFFF;
  return G_NTOHS(sum);
}

/***********************************************/
/***********************************************/
/***********************************************/
#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

////// PREFERS SHARED in config (cudaFuncCachePreferShared)


#define CEIL(x, y) ( (x)/(y) + ( (x)%(y) ? 1 : 0 ) )


template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
                                    const int numImages, const int numFilters,
                                    const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                    const int moduleStride, const int numModulesY, const int numModulesX, const int imgStride,
                                    const float scaleTargets, const float scaleOutputs, const int conv) {
    __shared__ float shFilters[B_Y*numColors][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = blockIdx.y % blocksPerModule;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += filtersPerThread * B_Y * blockFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesY * numModulesX
            + myImgIdx;


    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    for (int p = 0; p < filterPixels; p += B_Y) {
        /*
         * Load B_Y pixels from B_Y*filtersPerThread filters
         */
        if (shFilterLoadY < B_Y) {
            #pragma unroll
            for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                if (p + p2 + shFilterLoadY < filterPixels) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                    }
                }
            }
        }

        /*
         * Load B_Y pixels from B_X*imgsPerThread images
         */
        const int pixIdx = p + threadIdx.y;
        if (pixIdx < filterPixels) {
            const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
            const int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
            if (y >= 0 && y< imgSizeY && x >= 0 && x < imgSizeX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSizeX + x) + i * B_X];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < B_Y*numColors; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                }
            }

        }
        __syncthreads();
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
                }
            }
        }
    }


}


template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache, bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups,
                                       const float scaleTargets, const float scaleOutputs,
                                       const int conv) {
    __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;

    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }
//    __shared__ int imgPos[]
    for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
                }
            }

            /*
             * Load B_Y pixels from B_X*imgsPerThread images
             */
            const int pixIdx = p + threadIdx.y;
            if (pixIdx < filterPixels) {
                const int x = imgLoadModPosX + pixIdx % filterSize;
                const int y = imgLoadModPosY + pixIdx / filterSize;
                if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                    float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }
                }

            }
            __syncthreads();
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
                }
            }
        }
    }


}

void _filterActs(float *images, int images_cols, int images_rows, float *filters, int filters_cols, 
                int filters_rows,  float *targets, int targets_cols, int targets_rows,
                int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                int numImgColors, int numGroups, float scaleTargets, float scaleOutput, int conv, cudaStream_t stream, 
                bool warmup) {

    int numFilterColors = numImgColors / numGroups;      
    int numFilters = filters_cols;
    int numModules = numModulesY * numModulesX;
    int numImages = images_cols;
    int imgPixels = images_rows/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;
    
    assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
    assert(numGroups == 1 || numFilterColors % 2 == 0);
    assert(numFilters % (16 * numGroups) == 0);
    assert(numImgColors % numGroups == 0);
    assert(images_rows == imgPixels * numImgColors);
    assert(imgSizeY * imgSizeX == imgPixels);
    int numFiltersPerGroup = numFilters / numGroups;

    int imgStride = images_cols; // ???? //images.getStride(); // images does not need to be a contiguous matrix

    int filterPixels = filters_rows / (filterModuleMult * numFilterColors);
    int filterSize = int(sqrt(filterPixels));
    assert(filterSize * filterSize == filterPixels);
    assert(filters_rows == filterModuleMult * numFilterColors * filterPixels);

    // These routines don't handle the case when only part of the image is visited in the convolution
    assert(paddingStart <= 0);
    assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    assert(moduleStride <= filterSize);
    
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    dim3 blocks = numFiltersPerGroup % 32 == 0 ? dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 8))
                                               : dim3(DIVUP(numImages, 32 * imgsPerThread), (numModules * numFilters) / (4 * 4));

    if( warmup ) {
        blocks = dim3(4, 16);
    }

    dim3 threads(32, 4);
    bool checkImgBounds = numImages % (32*imgsPerThread) != 0;

    printf("blocks(%d, %d, %d), threads(%d, %d, %d)\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    
    /*
    if (scaleTargets == 0) {
        targets.resize(numFilters * numModules, numImages);
    } else {
        assert(targets.getNumRows() == numFilters * numModules);
        assert(targets.getNumCols() == numImages);
    }
    */


    assert(targets_rows == numFilters * numModules);
    assert(targets_cols == numImages);

    printf("\n\n");
    printf("filters.getNumCols = %d, filters.getnumrows = %d, images.getNumCols = %d, images.getNumRows = %d, targets.getNumcols = %d, targets.getNumrows = %d\n\n",
            filters_cols, filters_rows, images_cols, images_rows, targets_cols, targets_rows);

    printf("\n\n\n====== Kernel Parameters ======\n\n");

    printf("images = %p\n"
        "filters = %p\n"
        "targets = %p\n"
        "numImages = %d\n"
        "numFilters = %d\n"
        "imgSizeY = %d\n"
        "imgSizeX = %d\n"
        "filterSize = %d\n"
        "paddingStart = %d\n"
        "moduleStride = %d\n"
        "numModulesY = %d\n"
        "numModulesX = %d\n"
        "imgStride = %d\n"
        "scaleTargts = %lf\n"
        "scaleOutputs = %lf\n"
        "conv = %d\n"
        "numImgColors = %d\n"
        "imgsPerThread = %d\n"
        "numGroups = %d\n"
        "checkImgBounds = %d\n"
        "numFiltersPerGroup = %d\n"
        "blocks = %d, %d, %d\n"
        "threads = %d, %d, %d\n"
        "\n===================================\n",
        images, filters, targets,
        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart,
        moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv,
        numImgColors, imgsPerThread, numGroups, checkImgBounds, numFiltersPerGroup, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

#if 0    
    dim3 tmpBlocks(4, 64, 1);
        //filterActs_YxX_color < 4, 32, 1, 4, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
        filterActs_YxX_color<<<tmpBlocks, threads, 0, stream>>>(images, filters, targets, numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, 
#endif

    if (imgsPerThread == 4) {
        if (numImgColors <= 3) {
            assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
            if (scaleTargets == 0) { // don't scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            ////cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                         if (numFilters % 32 == 0) {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, true >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 8, 3, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, true >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 4, 3, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    } else {
                         if (numFilters % 32 == 0) {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, false >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 8, 3, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, false >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 4, 4, 3, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    }
                }
            } else { // do scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 1, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 1, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 3, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 3, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 8, 3, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 4, 4, 3, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }
            }
        } else {
            if (scaleTargets == 0) { // don't scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            } else { // do scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 8, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 4, 4, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    } else if (imgsPerThread == 2) {
        if (numImgColors <= 3) {
            assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
            if (scaleTargets == 0) { // don't scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                         if (numFilters % 32 == 0) {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, false, true >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 8, 3, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, false, true >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 4, 3, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    } else {
                         if (numFilters % 32 == 0) {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, false, false >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 8, 3, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, false, false >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 2, 4, 3, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    }
                }
            } else { // do scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 1, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 1, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 1, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 1, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 2, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 2, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 3, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 3, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 8, 3, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 8, 3, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 2, 4, 3, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 2, 4, 3, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }
            }
        } else {
            if (scaleTargets == 0) { // don't scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            } else { // do scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 8, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 2, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 2, 4, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    } else {
        if (numImgColors <= 3) {
            assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
            if (scaleTargets == 0) { // don't scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            printf("\n\n\n\ I AM HERE \n\n\n");
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, false, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, false, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                         if (numFilters % 32 == 0) {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, false, true >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 8, 3, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             printf("\n\n\n\nBING HERE\n\n\n\n");
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, false, true >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 4, 3, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    } else {
                         if (numFilters % 32 == 0) {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, false, false >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 8, 3, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         } else {
                             //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, false, false >, cudaFuncCachePreferShared);
                             filterActs_YxX_color < 4, 32, 1, 4, 3, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                         numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                         }
                    }
                }
            } else { // do scale
                if (numImgColors == 1) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 1, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 1, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 1, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 1, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                } else if (numImgColors == 2) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 2, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 2, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }  else if (numImgColors == 3) {
                    if (checkImgBounds) {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 3, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, true, true >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 3, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    } else {
                        if (numFilters % 32 == 0) {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 8, 3, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 8, 3, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        } else {
                            //cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 1, 4, 3, true, false >, cudaFuncCachePreferShared);
                            filterActs_YxX_color < 4, 32, 1, 4, 3, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                        numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
                        }
                    }
                }
            }
        } else {
            if (scaleTargets == 0) { // don't scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, true >, cudaFuncCachePreferShared);
                        printf("\n\n\n\n\n BING BING BING \n\n\n\n\n");
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, false, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, false, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            } else { // do scale
                if (checkImgBounds) {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, true >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, true, true > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                } else {
                    if (numFiltersPerGroup % 32 == 0) {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 8, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 8, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    } else {
                        //cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 1, 4, 2, false, false >, cudaFuncCachePreferShared);
                        filterActs_YxX_sparse < 4, 32, 1, 4, 2, true, false > <<<blocks, threads, 0, stream>>>(images, filters, targets,
                                    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
                    }
                }
            }
        }
    }

}

  

