#ifndef _PACKET_SWIZZLE_KERNEL_H_
#define _PACKET_SWIZZLE_KERNEL_H_

#include <stdio.h>
#include "common.h"


#include "rte_lpm.h"
#include "rte_lpm6.h"

/**< xia-router0 xge0,1    xia-router1 xge0,1
  *  xia-router0 xge2,3    xia-router1 xge2,3 */
__device__ long long src_mac_arr[8] = {0x6c10bb211b00, 0x6d10bb211b00, 0x64d2bd211b00, 0x65d2bd211b00,
                     0xc8a610ca0568, 0xc9a610ca0568, 0xa2a610ca0568, 0xa3a610ca0568};

__device__ long long dst_mac_arr[8] = {0x36d3bd211b00, 0x37d3bd211b00, 0x44d7a3211b00, 0x45d7a3211b00,
                     0xa8d6a3211b00, 0xa9d6a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00};

__device__ void save_register(int* buffer)
{    
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int* addr = &buffer[tid];
    int val = tid;
    int scale = 4;
    // Stores r1 to global memory at the correct address
    asm("st.global.u32 [%0], %%r1;" :: "l"(addr)  "r"(val), "r"(scale));
}    

__device__ void restore_register(int* buffer)
{    
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int* addr = &buffer[tid];
    int val = tid;
    int scale = 4;
    // Moves 'val' into r0
    // Multiplies r0 by 'scale' and stores it back into r0
    // Stores r0 to global memory at the correct address
    asm("ld.global.u32 %%r1, [%0];" :: "l"(addr));
}  

// __device__ void save_registers(int* buffer)
// {    
//     int tid = blockIdx.x*blockDim.x + threadIdx.x;
//     int* addr = &buffer[tid];
//     int val = tid;
//     int scale = 4;
//     asm("st.global.u32 [%0], %%r1;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 512], %%r2;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 1024], %%r3;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 1536], %%r4;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 2048], %%r5;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 2560], %%r6;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 3072], %%r7;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 3584], %%r8;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 4096], %%r9;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 4608], %%r10;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 5120], %%r11;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 5632], %%r12;" :: "l"(addr));
//     asm("st.global.u32 [%0 + 6144], %%r13;" :: "l"(addr));
// }    

// __device__ void restore_registers(int* buffer)
// {    
//     int tid = blockIdx.x*blockDim.x + threadIdx.x;
//     int* addr = &buffer[tid];
//     int val = tid;
//     int scale = 4;
//     asm("ld.global.u32 %%r1, [%0];" :: "l"(addr));
//     asm("ld.global.u32 %%r2, [%0 + 512];" :: "l"(addr));
//     asm("ld.global.u32 %%r3, [%0 + 1024];" :: "l"(addr));
//     asm("ld.global.u32 %%r4, [%0 + 1536];" :: "l"(addr));
//     asm("ld.global.u32 %%r5, [%0 + 2048];" :: "l"(addr));
//     asm("ld.global.u32 %%r6, [%0 + 2560];" :: "l"(addr));
//     asm("ld.global.u32 %%r7, [%0 + 3072];" :: "l"(addr));
//     asm("ld.global.u32 %%r8, [%0 + 3584];" :: "l"(addr));
//     asm("ld.global.u32 %%r9, [%0 + 4096];" :: "l"(addr));
//     asm("ld.global.u32 %%r10, [%0 + 4608];" :: "l"(addr));
//     asm("ld.global.u32 %%r11, [%0 + 5120];" :: "l"(addr));
//     asm("ld.global.u32 %%r12, [%0 + 5632];" :: "l"(addr));
//     asm("ld.global.u32 %%r13, [%0 + 6144];" :: "l"(addr));
// }                    

extern "C" __global__ void normal_kernel(pkt_hdr_normal* packet_batch, pkt_hdr_normal* response_batch, unsigned n_pkts)
{
    unsigned tid = threadIdx.x;
    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid == 0)
        printf("tid = %d. gid = %d. Number of packets = %u. Total num threads = %d\n", tid, gid, n_pkts, blockDim.x * gridDim.x);

    if (gid < n_pkts) {
        response_batch[gid].ether_dhost_1 = packet_batch[gid].ether_shost_1;
        response_batch[gid].ether_dhost_2 = packet_batch[gid].ether_shost_2;
        response_batch[gid].ether_shost_1 = packet_batch[gid].ether_dhost_1;
        response_batch[gid].ether_shost_2 = packet_batch[gid].ether_dhost_2;
        response_batch[gid].ether_type = packet_batch[gid].ether_type;

        response_batch[gid].ip_version = packet_batch[gid].ip_version;
        response_batch[gid].ip_tos = packet_batch[gid].ip_tos;
        response_batch[gid].ip_tot_len = packet_batch[gid].ip_tot_len;
        response_batch[gid].ip_id = packet_batch[gid].ip_id;
        response_batch[gid].ip_frag_off = packet_batch[gid].ip_frag_off;
        response_batch[gid].ip_ttl = packet_batch[gid].ip_ttl;
        response_batch[gid].ip_protocol = packet_batch[gid].ip_protocol;
        response_batch[gid].ip_check = packet_batch[gid].ip_check;
        response_batch[gid].ip_saddr = packet_batch[gid].ip_daddr;
        response_batch[gid].ip_daddr = packet_batch[gid].ip_saddr;
                                                           
        response_batch[gid].udp_source = packet_batch[gid].udp_dest;
        response_batch[gid].udp_dest = packet_batch[gid].udp_source;
        response_batch[gid].udp_len = packet_batch[gid].udp_len;
        response_batch[gid].udp_check = packet_batch[gid].udp_check;
    }

}

#define WARP_SIZE 32
extern "C" __global__ void normal_shmem_kernel(pkt_hdr_normal* packet_batch, pkt_hdr_normal* response_batch, unsigned n_pkts)
{
    unsigned tid = threadIdx.x;
    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned w_tid = threadIdx.x % WARP_SIZE;

    
    __shared__ pkt_hdr_normal shmem_packet_batch[NUM_REQUESTS_PER_BATCH];

    if (gid == 0)
        printf("tid = %d. gid = %d. Number of packets = %u. Total num threads = %d\n", tid, gid, n_pkts, blockDim.x * gridDim.x);


    unsigned num_requests_per_batch = NUM_REQUESTS_PER_BATCH;
    unsigned hdr_size = sizeof(pkt_hdr_normal);
    unsigned num_threads_per_packet = hdr_size / sizeof(int);
    unsigned total_num_warps = blockDim.x / WARP_SIZE;
    unsigned num_packets_per_warp = NUM_REQUESTS_PER_BATCH / total_num_warps; 

    if (gid == 0)
        printf("# threads per block = %d. # req/batch = %d. # threads / packet = %d. HDR size = %d\n" \
                "# threads per packet = %d, total # warps = %d, num_packets_per_warp = %d\n", 
                num_threads_per_packet, num_requests_per_batch, num_threads_per_packet, hdr_size,
                num_threads_per_packet, total_num_warps, num_packets_per_warp);

   
    unsigned global_req_ind = (unsigned)(gid/WARP_SIZE);
    global_req_ind *= num_packets_per_warp;
    unsigned sh_req_ind = (unsigned)(tid/WARP_SIZE);
    sh_req_ind *= num_packets_per_warp;
    unsigned masked_ind = w_tid % num_threads_per_packet;

    for (unsigned i=0; i<num_packets_per_warp; ++i) {
        int* req_ptr = (int*)(packet_batch + (global_req_ind + i));
        int* pkt_ptr = (int*)&shmem_packet_batch[sh_req_ind + i]; 
        pkt_ptr[masked_ind] = req_ptr[masked_ind];
    }

    // Swap source and dest
    if (gid < n_pkts) {
        int tmp1 = shmem_packet_batch[tid].ether_dhost_1;
        int tmp2 = shmem_packet_batch[tid].ether_dhost_2;
        shmem_packet_batch[tid].ether_dhost_1 = shmem_packet_batch[tid].ether_shost_1;
        shmem_packet_batch[tid].ether_dhost_2 = shmem_packet_batch[tid].ether_shost_2;
        shmem_packet_batch[tid].ether_shost_1 = tmp1;
        shmem_packet_batch[tid].ether_shost_2 = tmp2;

        tmp1 = shmem_packet_batch[tid].ip_daddr;
        shmem_packet_batch[tid].ip_daddr = shmem_packet_batch[tid].ip_saddr;
        shmem_packet_batch[tid].ip_saddr = tmp1;

        tmp1 = shmem_packet_batch[tid].udp_dest;
        shmem_packet_batch[tid].udp_dest = shmem_packet_batch[tid].udp_source;
        shmem_packet_batch[tid].udp_source = tmp1;
    }

    for (unsigned i=0; i<num_packets_per_warp; ++i) {
        int* pkt_ptr = (int*)&shmem_packet_batch[sh_req_ind + i];
        int* res_ptr = (int*)(response_batch + (global_req_ind + i));
        res_ptr[masked_ind] = pkt_ptr[masked_ind];
    }
}






extern "C" __global__ void packet_swizzle_kernel(pkt_hdr_batch* packet_batch, pkt_hdr_batch* response_batch, unsigned n_pkts)
{
    unsigned tid = threadIdx.x;
    unsigned bid = blockIdx.x;

    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid == 0)
        printf("tid = %d. gid = %d. Number of packets = %u. Total num threads = %d\n", tid, gid, n_pkts, blockDim.x * gridDim.x);

    pkt_hdr_batch* pkt = &packet_batch[bid];
    pkt_hdr_batch* res = &response_batch[bid];

    // PING Benchmark
    if (gid < n_pkts) {
        res->ether_dhost_1[tid] = pkt->ether_shost_1[tid];
        res->ether_dhost_2[tid] = pkt->ether_shost_2[tid];
        res->ether_shost_1[tid] = pkt->ether_dhost_1[tid];
        res->ether_shost_2[tid] = pkt->ether_dhost_2[tid];
        res->ether_type[tid] = pkt->ether_type[tid];
        
        res->ip_version[tid] = pkt->ip_version[tid];
        res->ip_tos[tid] = pkt->ip_tos[tid];
        res->ip_tot_len[tid] = pkt->ip_tot_len[tid];
        res->ip_id[tid] = pkt->ip_id[tid];
        res->ip_frag_off[tid] = pkt->ip_frag_off[tid];
        res->ip_ttl[tid] = pkt->ip_ttl[tid];
        res->ip_protocol[tid] = pkt->ip_protocol[tid];
        res->ip_check[tid] = pkt->ip_check[tid];
        res->ip_saddr[tid] = pkt->ip_daddr[tid];
        res->ip_daddr[tid] = pkt->ip_saddr[tid];

        res->udp_source[tid] = pkt->udp_dest[tid];
        res->udp_dest[tid] = pkt->udp_source[tid];
        res->udp_len[tid] = pkt->udp_len[tid];
        res->udp_check[tid] = pkt->udp_check[tid];
    }

}



__device__ uint16_t port_lookup(uint16_t* tbl24, uint16_t* tbl8, uint32_t ip)
{
    uint32_t tbl24_index = (ip >> 8);
    uint16_t tbl_entry;
    tbl_entry = tbl24[tbl24_index];

    if ((tbl_entry & RTE_LPM_VALID_EXT_ENTRY_BITMASK) == RTE_LPM_VALID_EXT_ENTRY_BITMASK) {
        unsigned tbl8_index = (uint8_t)ip + ((uint8_t) tbl_entry * RTE_LPM_TBL8_GROUP_NUM_ENTRIES);
        tbl_entry = tbl8[tbl8_index];
    }

    return tbl_entry;
}


extern "C" __global__ void ipv4_fwd_kernel_save_regs(pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts, int* reg_buffer, bool save_regs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    if (save_regs) {
        //save regs
        for (int i=0; i<IPV4_REG_NUM; i++) {
            //save_register(reg_buffer);
            reg_buffer[tid * IPV4_REG_NUM + i] = tid;
        }
    }
    
    if (gid < n_pkts) {
        uint32_t ip = packet_batch[gid].ip_daddr;
        uint16_t tbl_entry = port_lookup(tbl24, tbl8, ip);
        packet_batch[gid].ether_dhost_1 = (uint32_t)(dst_mac_arr[tbl_entry] >> 16);
        packet_batch[gid].ether_dhost_2 = (uint32_t)(dst_mac_arr[tbl_entry] & 0xFFFF);
        packet_batch[gid].ether_shost_1 = (uint32_t)(src_mac_arr[tbl_entry] >> 16);
        packet_batch[gid].ether_shost_2 = (uint32_t)(src_mac_arr[tbl_entry] & 0xFFFF);
    }

    if (save_regs) {
        //save regs
        for (int i=0; i<IPV4_REG_NUM; i++) {
            tid = reg_buffer[tid * IPV4_REG_NUM + i];
        }
    }
}

extern "C" __global__ void ipv4_fwd_kernel(pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts, int* reg_buffer, bool save_regs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    if (gid < n_pkts) {
        uint32_t ip = packet_batch[gid].ip_daddr;
        uint16_t tbl_entry = port_lookup(tbl24, tbl8, ip);
        packet_batch[gid].ether_dhost_1 = (uint32_t)(dst_mac_arr[tbl_entry] >> 16);
        packet_batch[gid].ether_dhost_2 = (uint32_t)(dst_mac_arr[tbl_entry] & 0xFFFF);
        packet_batch[gid].ether_shost_1 = (uint32_t)(src_mac_arr[tbl_entry] >> 16);
        packet_batch[gid].ether_shost_2 = (uint32_t)(src_mac_arr[tbl_entry] & 0xFFFF);
    }
}

extern "C" __global__ void swizzle_ipv4_fwd_kernel(pkt_hdr_batch* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    pkt_hdr_batch* pkt = &packet_batch[bid];
    if (gid < n_pkts) {        
        uint32_t ip = pkt->ip_daddr[tid];
        uint16_t tbl_entry = port_lookup(tbl24, tbl8, ip);
        pkt->ether_dhost_1[tid] = (uint32_t)(dst_mac_arr[tbl_entry] >> 16);
        //pkt->ether_dhost_2[tid] = (uint32_t)(dst_mac_arr[tbl_entry] & 0xFFFF);
        //pkt->ether_shost_1[tid] = (uint32_t)(src_mac_arr[tbl_entry] >> 16);
        //pkt->ether_shost_2[tid] = (uint32_t)(src_mac_arr[tbl_entry] & 0xFFFF);
    }
}

__device__ uint32_t ipv6_port_lookup(uint16_t* tbl24, uint16_t* tbl8, ipv6_pkt_hdr_normal* pkt)
{
    int status;                                    
    uint8_t first_byte;                            
    uint32_t tbl24_index, tbl8_index, tbl_entry;   
    
    first_byte = 3;
    uint32_t addr = pkt->ip_saddr1; 
    tbl24_index = (addr >> 8); 
                                                                                                      
    tbl_entry = tbl24[tbl24_index];
                                                                                                      
    uint32_t offset = 0;
    do {
        if ((tbl_entry & RTE_LPM6_VALID_EXT_ENTRY_BITMASK) == RTE_LPM6_VALID_EXT_ENTRY_BITMASK) {
            if (first_byte == 4) {
                addr = pkt->ip_saddr2;
                offset = 24; 
            } else if (first_byte == 8) {
                addr = pkt->ip_saddr3;
                offset = 24; 
            } else if (first_byte == 12) {
                addr = pkt->ip_saddr4;
                offset = 24; 
            }
                                                                                                      
            uint8_t x = (uint8_t)((addr >> offset) & 0xFF);
            tbl8_index = x + ((tbl_entry & RTE_LPM6_TBL8_BITMASK) * RTE_LPM6_TBL8_GROUP_NUM_ENTRIES);
            tbl_entry = tbl8[tbl8_index];
                                                                                                      
            first_byte++;
            offset -= 8;
            status = 1;
        } else {
            status = 0;
        }
    } while (status == 1);

    return tbl_entry;
}

extern "C" __global__ void ipv6_fwd_kernel(ipv6_pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_pkts) {
        ipv6_pkt_hdr_normal* pkt = &packet_batch[gid];
        uint32_t tbl_entry = ipv6_port_lookup(tbl24, tbl8, pkt); 
        
        packet_batch[gid].ether_dhost_1 = (uint32_t)(dst_mac_arr[tbl_entry] >> 16);
        packet_batch[gid].ether_dhost_2 = (uint32_t)(dst_mac_arr[tbl_entry] & 0xFFFF);
        packet_batch[gid].ether_shost_1 = (uint32_t)(src_mac_arr[tbl_entry] >> 16);
        packet_batch[gid].ether_shost_2 = (uint32_t)(src_mac_arr[tbl_entry] & 0xFFFF);
    }
}

__device__ uint32_t ipv6_port_lookup_swizzle(uint16_t* tbl24, uint16_t* tbl8, ipv6_pkt_hdr_swizzle* pkt, unsigned pkt_ind)
{
    int status;                                    
    uint8_t first_byte;                            
    uint32_t tbl24_index, tbl8_index, tbl_entry;   
    
    first_byte = LOOKUP_FIRST_BYTE;
    uint32_t addr = pkt->ip_saddr1[pkt_ind]; 
    tbl24_index = (addr >> 8); 
                                                                                                      
    tbl_entry = tbl24[tbl24_index];
                                                                                                      
    uint32_t offset = 0;
    do {
        if ((tbl_entry & RTE_LPM6_VALID_EXT_ENTRY_BITMASK) == RTE_LPM6_VALID_EXT_ENTRY_BITMASK) {
            if (first_byte == 4) {
                addr = pkt->ip_saddr2[pkt_ind];
                offset = 24; 
            } else if (first_byte == 8) {
                addr = pkt->ip_saddr3[pkt_ind];
                offset = 24; 
            } else if (first_byte == 12) {
                addr = pkt->ip_saddr4[pkt_ind];
                offset = 24; 
            }

            uint8_t x = (uint8_t)((addr >> offset) & 0xFF);
            tbl8_index = x + ((tbl_entry & RTE_LPM6_TBL8_BITMASK) * RTE_LPM6_TBL8_GROUP_NUM_ENTRIES);
            tbl_entry = tbl8[tbl8_index];
                                                                                                      
            first_byte++;
            offset -= 8;
            status = 1;
        } else {
            status = 0;
        }
    } while (status == 1);

    return tbl_entry;
}


extern "C" __global__ void swizzle_ipv6_fwd_kernel(ipv6_pkt_hdr_swizzle* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n_pkts) {
        ipv6_pkt_hdr_swizzle* pkt = &packet_batch[bid];
        uint32_t tbl_entry = ipv6_port_lookup_swizzle(tbl24, tbl8, pkt, tid);

        pkt->ether_dhost_1[tid] = (uint32_t)(dst_mac_arr[tbl_entry] >> 16);
        pkt->ether_dhost_2[tid] = (uint32_t)(dst_mac_arr[tbl_entry] & 0xFFFF);
        pkt->ether_shost_1[tid] = (uint32_t)(src_mac_arr[tbl_entry] >> 16);
        pkt->ether_shost_2[tid] = (uint32_t)(src_mac_arr[tbl_entry] & 0xFFFF);
    }
}





#endif // #ifndef _PACKET_SWIZZLE_KERNEL_H_
