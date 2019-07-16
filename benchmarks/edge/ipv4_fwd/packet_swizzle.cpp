#include "packet_swizzle.h"
#include "option_parser.h"
#include "config.h"
#include "common.h"
#include "util.h"
#include "ipv4.h"
#include "ipv6.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runNormalTest();
void runShmemTest();
void runSwizzleTest();
void run_ipv4_test(bool swizzle);
void run_ipv6_test(bool swizzle);

void init_swizzle_requests(struct kernel_args* args, bool alloc_response);
void init_normal_requests(struct kernel_args* args, bool alloc_response);
void init_ipv6_normal_requests(struct kernel_args* args, struct ipv6_prefix* prefix_arr, unsigned prefix_ind);
void generate_dummy_packet(struct pkt_hdr* pkt, unsigned gen_type);
void ipv6_generate_dummy_packet(struct ipv6_pkt_hdr* pkt, struct ipv6_prefix* pfa);
void print_pkt_hdr(struct pkt_hdr* hdr);
void print_swizzled_packet(struct pkt_hdr_batch* batch_ptr, unsigned pkt_ind);
void print_normal_packet(struct pkt_hdr_normal* pkt);
static u_int32_t wrapsum (u_int32_t sum);
int in_cksum(unsigned char* buf, unsigned nbytes, int sum);
static CUresult initCUDA(unsigned dev_num, CUfunction* kernel, const char* kernel_name);

void ipv6_normal_packet(struct ipv6_pkt_hdr_normal* pkt_hdr_normal_ptr, struct ipv6_pkt_hdr* pkt, unsigned pkt_ind);
void ipv6_swizzle_packet(struct ipv6_pkt_hdr_swizzle* pkt_hdr_batch_ptr, struct ipv6_pkt_hdr* pkt, unsigned pkt_ind);
void init_ipv6_swizzle_requests(struct kernel_args* args, struct ipv6_prefix* prefix_arr, unsigned prefix_ind);

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUstream cuStream;
CUfunction cuFunction; 
size_t totalGlobalMem;

CudaConfig* g_cuda_config = NULL;
char* g_file_path = NULL;

unsigned g_batch_size = NUM_REQUESTS_PER_BATCH;
unsigned g_num_batches = 65536;

/**< [xia-router0 - xge0,1,2,3], [xia-router1 - xge0,1,2,3] */
LL src_mac_arr[2][4] = {{0x36d3bd211b00, 0x37d3bd211b00, 0xa8d6a3211b00, 0xa9d6a3211b00},
						{0x44d7a3211b00, 0x45d7a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00}};

/**< [xia-router2 - xge0,1,4,5], [xia-router2 - xge2,3,6,7] */
LL dst_mac_arr[2][4] = {{0x6c10bb211b00, 0x6d10bb211b00, 0xc8a610ca0568, 0xc9a610ca0568},
						{0x64d2bd211b00, 0x65d2bd211b00, 0xa2a610ca0568, 0xa3a610ca0568}};

uint64_t rss_seed = 0xdeadbeef;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    g_file_path = argv[0];

    option_parser_t opp = option_parser_create();
    g_cuda_config = new CudaConfig();
    g_cuda_config->reg_options(opp); 

    option_parser_cmdline(opp, argc, (const char**)argv);
    option_parser_print(opp, stdout);
    
    switch (g_cuda_config->config_to_run) {
    case 0:
        cout << "NORMAL PACKET TEST" << endl;
        runNormalTest();
        break;
    case 1:
        cout << "SHARED MEM PACKET TEST" << endl;
        runShmemTest();
        break;
    case 2:
        cout << "SWIZZLE PACKET TEST" << endl;
        runSwizzleTest();
        break;
    case 3:
        cout << "NORMAL IPv4 TEST" << endl;
        run_ipv4_test(false);
        break;
    case 4:
        cout << "SWIZZLE IPv4 TEST" << endl;
        run_ipv4_test(true);
        break;
    case 5:
        cout << "NORMAL IPv6 TEST" << endl;
        run_ipv6_test(false);
        break;
    case 6:
        cout << "SWIZZLE IPv6 TEST" << endl;
        run_ipv6_test(true);
        break;

    default:
        cout << "Error: Undefined run config <" << g_cuda_config->config_to_run << ">" << endl;
        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}



void init_gpu(const char* kernel_name)
{
    checkCudaErrors(initCUDA(g_cuda_config->dev_num, &cuFunction, kernel_name));
    checkCudaErrors(cuStreamCreate(&cuStream, 0));
}

void cleanup(struct kernel_args* k_args)
{
    cuMemFree(k_args->g_packet_buffer);
    cuMemFree(k_args->g_response_buffer);
    free(k_args->h_packet_buffer);
    free(k_args->h_response_buffer);
    checkCudaErrors(cuStreamDestroy(cuStream));
    checkCudaErrors(cuCtxDestroy(cuContext));
}

void launch_kernel(struct kernel_args* k_args, bool response)
{
    checkCudaErrors(cuMemcpyHtoD(k_args->g_packet_buffer, k_args->h_packet_buffer, k_args->buffer_size));
                                                                                                                   
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    dim3 block(NUM_REQUESTS_PER_BATCH, 1, 1);
    dim3 grid(g_num_batches, 1, 1);
    
    unsigned n_packets = k_args->n_batches * k_args->batch_size;
    void* args[3] = {&k_args->g_packet_buffer, &k_args->g_response_buffer, &n_packets};
                                                                                                                    
    sdkStartTimer(&timer);
    checkCudaErrors(cuLaunchKernel(cuFunction, grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z, 0,
                                   cuStream, args, NULL));
    checkCudaErrors(cuStreamSynchronize(cuStream));
                                                                                                                    
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    if (response)
        checkCudaErrors(cuMemcpyDtoH((void *)k_args->h_response_buffer, k_args->g_response_buffer, k_args->buffer_size));
    else
        checkCudaErrors(cuMemcpyDtoH((void *)k_args->h_packet_buffer, k_args->g_packet_buffer, k_args->buffer_size));

}


bool validate_normal_packet(pkt_hdr_normal* pkt1, pkt_hdr_normal* pkt2)
{
    if (pkt1->ether_shost_1 != pkt2->ether_dhost_1 || 
            pkt1->ether_shost_2 != pkt2->ether_dhost_2 || 
            pkt1->ether_dhost_1 != pkt2->ether_shost_1 || 
            pkt1->ether_dhost_2 != pkt2->ether_shost_2 || 
            pkt1->ip_version != pkt2->ip_version || 
            pkt1->ip_tos != pkt2->ip_tos || 
            pkt1->ip_tot_len != pkt2->ip_tot_len || 
            pkt1->ip_id != pkt2->ip_id || 
            pkt1->ip_frag_off != pkt2->ip_frag_off || 
            pkt1->ip_ttl != pkt2->ip_ttl || 
            pkt1->ip_protocol != pkt2->ip_protocol || 
            pkt1->ip_check != pkt2->ip_check || 
            pkt1->ip_daddr != pkt2->ip_saddr || 
            pkt1->ip_saddr != pkt2->ip_daddr || 
            pkt1->udp_dest != pkt2->udp_source || 
            pkt1->udp_source != pkt2->udp_dest || 
            pkt1->udp_len != pkt2->udp_len || 
            pkt1->udp_check != pkt2->udp_check) {
        return false;
    }
    return true;
}

bool validate_swizzle_packet(pkt_hdr_batch* pkt1, pkt_hdr_batch* pkt2, unsigned index)
{

    if (pkt1->ether_shost_1[index] != pkt2->ether_dhost_1[index] || 
            pkt1->ether_shost_2[index] != pkt2->ether_dhost_2[index] || 
            pkt1->ether_dhost_1[index] != pkt2->ether_shost_1[index] || 
            pkt1->ether_dhost_2[index] != pkt2->ether_shost_2[index] || 
            pkt1->ip_version[index] != pkt2->ip_version[index] || 
            pkt1->ip_tos[index] != pkt2->ip_tos[index] || 
            pkt1->ip_tot_len[index] != pkt2->ip_tot_len[index] || 
            pkt1->ip_id[index] != pkt2->ip_id[index] || 
            pkt1->ip_frag_off[index] != pkt2->ip_frag_off[index] || 
            pkt1->ip_ttl[index] != pkt2->ip_ttl[index] || 
            pkt1->ip_protocol[index] != pkt2->ip_protocol[index] || 
            pkt1->ip_check[index] != pkt2->ip_check[index] || 
            pkt1->ip_daddr[index] != pkt2->ip_saddr[index] || 
            pkt1->ip_saddr[index] != pkt2->ip_daddr[index] || 
            pkt1->udp_dest[index] != pkt2->udp_source[index] || 
            pkt1->udp_source[index] != pkt2->udp_dest[index] || 
            pkt1->udp_len[index] != pkt2->udp_len[index] || 
            pkt1->udp_check[index] != pkt2->udp_check[index]) {
        return false;
    }
    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////

void run_ipv6_test(bool swizzle)
{
    struct kernel_args k_args;
    struct rte_lpm6 *lpm;
    struct ipv6_prefix *prefix_arr;

    /**< rte_lpm6_tbl_entry is 4 bytes */
    int entry_sz = sizeof(struct rte_lpm6_tbl_entry);
    int tbl24_bytes = RTE_LPM6_TBL24_NUM_ENTRIES * entry_sz;
    int tbl8_bytes = (IPV6_NUM_TBL8 * RTE_LPM6_TBL8_GROUP_NUM_ENTRIES) * entry_sz;

    lpm = ipv6_init(IPV6_XIA_R2_PORT_MASK, &prefix_arr, 1);
    int prefix_arr_i = rand() % IPV6_NUM_RAND_PREFIXES;

    if (!swizzle) {
        init_gpu("ipv6_fwd_kernel");
        init_ipv6_normal_requests(&k_args, prefix_arr, prefix_arr_i);
    } else {
        init_gpu("swizzle_ipv6_fwd_kernel");
        init_ipv6_swizzle_requests(&k_args, prefix_arr, prefix_arr_i);
    }

    CUdeviceptr gpu_tbl24 = 0;
    CUdeviceptr gpu_tbl8 = 0;
    
    /**< Alloc and copy tbl24 and tbl8 arrays to GPU memory */
    printf("\tGPU master: alloc tbl24 (size = %lf MB) on device\n", (float)tbl24_bytes / 1e6);
    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_tbl24, tbl24_bytes)); 
    checkCudaErrors(cuMemcpyHtoD(gpu_tbl24, lpm->tbl24, tbl24_bytes));
                                                                                               
    printf("\tGPU master: alloc tbl8 (size = %lf MB) on device\n", (float)tbl8_bytes / 1e6);
    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_tbl8, tbl8_bytes)); 
    checkCudaErrors(cuMemcpyHtoD(gpu_tbl8, lpm->tbl8, tbl8_bytes));

    checkCudaErrors(cuMemcpyHtoD(k_args.g_packet_buffer, k_args.h_packet_buffer, k_args.buffer_size));
                                                                                                                   
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    dim3 block(NUM_REQUESTS_PER_BATCH, 1, 1);
    dim3 grid(g_num_batches, 1, 1);
                                                                                                                           
    unsigned n_packets = k_args.n_batches * k_args.batch_size;
    void* args[4] = {&k_args.g_packet_buffer, &gpu_tbl24, &gpu_tbl8, &n_packets};
                                                                                                                    
    sdkStartTimer(&timer);
    checkCudaErrors(cuLaunchKernel(cuFunction, grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z, 0,
                                   cuStream, args, NULL));
    checkCudaErrors(cuStreamSynchronize(cuStream));
                                                                                                                    
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
                                                                                                                          
    checkCudaErrors(cuMemcpyDtoH((void *)k_args.h_packet_buffer, k_args.g_packet_buffer, k_args.buffer_size));

}

void run_ipv4_test(bool swizzle)
{
    struct kernel_args k_args;
    struct rte_lpm* lpm;


    if (!swizzle)  {
        init_gpu("ipv4_fwd_kernel");
        init_normal_requests(&k_args, false);
    } else {
        init_gpu("swizzle_ipv4_fwd_kernel");
        init_swizzle_requests(&k_args, false);
    }

    // Initialize the IPv4 forwarding table 
    lpm = ipv4_init();

    /**< XXX: HACK - Failed lookups should choose a random port. This hack
      *  overdoes it and directs *all* lookups to a random ports. */
    for (unsigned i = 0; i < RTE_LPM_TBL24_NUM_ENTRIES; i ++) {
        uint16_t *tbl24_entry = (uint16_t *) &(lpm->tbl24[i]);

        /**< If this entry does not point to a tbl8, randomize it. */
        if((*tbl24_entry & RTE_LPM_VALID_EXT_ENTRY_BITMASK) !=
                RTE_LPM_VALID_EXT_ENTRY_BITMASK) {
            *tbl24_entry = i & 3;
        }
    }

    /**< rte_lpm_tbl24_entry ~ rte_lpm_tbl8_entry ~ uint16_t */
    int entry_sz = sizeof(struct rte_lpm_tbl24_entry);
    int tbl24_bytes = RTE_LPM_TBL24_NUM_ENTRIES * entry_sz;
    int tbl8_bytes = RTE_LPM_TBL8_NUM_ENTRIES * entry_sz;
    
	CUdeviceptr gpu_tbl24 = 0;
	CUdeviceptr gpu_tbl8 = 0;
    
    /**< Alloc and copy tbl24 and tbl8 arrays to GPU memory */
    printf("\tGPU master: alloc tbl24 (size = %lf MB) on device\n", (float)tbl24_bytes / 1e6);
    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_tbl24, tbl24_bytes)); 
    checkCudaErrors(cuMemcpyHtoD(gpu_tbl24, lpm->tbl24, tbl24_bytes));

    printf("\tGPU master: alloc tbl8 (size = %lf MB) on device\n", (float)tbl8_bytes / 1e6);
    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_tbl8, tbl8_bytes)); 
    checkCudaErrors(cuMemcpyHtoD(gpu_tbl8, lpm->tbl8, tbl8_bytes));

    checkCudaErrors(cuMemcpyHtoD(k_args.g_packet_buffer, k_args.h_packet_buffer, k_args.buffer_size));
                                                                                                                   
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    dim3 block(NUM_REQUESTS_PER_BATCH, 1, 1);
    dim3 grid(g_num_batches, 1, 1);

    unsigned n_packets = k_args.n_batches * k_args.batch_size;
    void* args[4] = {&k_args.g_packet_buffer, &gpu_tbl24, &gpu_tbl8, &n_packets};
                                                                                                                    
    sdkStartTimer(&timer);
    checkCudaErrors(cuLaunchKernel(cuFunction, grid.x, grid.y, grid.z,
                                   block.x, block.y, block.z, 0,
                                   cuStream, args, NULL));
    checkCudaErrors(cuStreamSynchronize(cuStream));
                                                                                                                    
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
                                                                                                                          
    checkCudaErrors(cuMemcpyDtoH((void *)k_args.h_packet_buffer, k_args.g_packet_buffer, k_args.buffer_size));


}



void runNormalTest()
{
    struct kernel_args k_args;

    init_gpu("normal_kernel");
    init_normal_requests(&k_args, true);
    launch_kernel(&k_args, true); 

    // Validate buffer
    pkt_hdr_normal* packet_buffer = (pkt_hdr_normal*)k_args.h_packet_buffer;
    pkt_hdr_normal* response_buffer = (pkt_hdr_normal*)k_args.h_response_buffer;
    for (unsigned i=0; i<g_num_batches*NUM_REQUESTS_PER_BATCH; ++i) {
        if (!validate_normal_packet(&response_buffer[i], &packet_buffer[i])) {
            cout << "FAILED" << endl;
            cout << "\tbatch: " << i/NUM_REQUESTS_PER_BATCH << endl;
            cout << "\tpacket: " << i%NUM_REQUESTS_PER_BATCH << endl;
            goto DONE; 
        }
    }
    cout << "SUCCESS" << endl;

DONE:
    cleanup(&k_args);
}

void runShmemTest()
{
    struct kernel_args k_args;

    init_gpu("normal_shmem_kernel");
    init_normal_requests(&k_args, true);
    launch_kernel(&k_args, true); 

    // Validate buffer                                                               
    pkt_hdr_normal* packet_buffer = (pkt_hdr_normal*)k_args.h_packet_buffer;
    pkt_hdr_normal* response_buffer = (pkt_hdr_normal*)k_args.h_response_buffer;
    for (unsigned i=0; i<g_num_batches*NUM_REQUESTS_PER_BATCH; ++i) {
        if (!validate_normal_packet(&response_buffer[i], &packet_buffer[i])) {
            cout << "FAILED" << endl;
            cout << "\tbatch: " << i/NUM_REQUESTS_PER_BATCH << endl;
            cout << "\tpacket: " << i%NUM_REQUESTS_PER_BATCH << endl;
            goto DONE; 
        }
    }
    cout << "SUCCESS" << endl;
    
DONE:
    cleanup(&k_args);
}


void runSwizzleTest()
{
    struct kernel_args k_args;

    init_gpu("packet_swizzle_kernel");
    init_swizzle_requests(&k_args, true);
    launch_kernel(&k_args, true); 

    // Validate buffer
    pkt_hdr_batch* packet_buffer = (pkt_hdr_batch*)k_args.h_packet_buffer;
    pkt_hdr_batch* response_buffer = (pkt_hdr_batch*)k_args.h_response_buffer;
    for (unsigned i=0; i<g_num_batches; ++i) {
        pkt_hdr_batch* pkt = &packet_buffer[i];
        pkt_hdr_batch* res = &response_buffer[i];
        for (unsigned j=0; j<NUM_REQUESTS_PER_BATCH; ++j) {

            if (!validate_swizzle_packet(res, pkt, j)) {
                cout << "FAILED" << endl;
                cout << "\tbatch: " << i << endl;
                cout << "\tpacket: " << j << endl;
                print_swizzled_packet(pkt, j);
                print_swizzled_packet(res, j);
                goto DONE; 
            }
        }
    }
    cout << "SUCCESS" << endl;

DONE:
    cleanup(&k_args);
}

void normal_packet(struct pkt_hdr_normal* pkt_hdr_normal_ptr, struct pkt_hdr* pkt, unsigned pkt_ind)
{
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_dhost_1, pkt->eh.ether_dhost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_dhost_2, pkt->eh.ether_dhost + 4, 2); 
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_shost_1, pkt->eh.ether_shost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_shost_2, pkt->eh.ether_shost + 4, 2); 
    pkt_hdr_normal_ptr[pkt_ind].ether_type = (u_int32_t)pkt->eh.ether_type; 

    pkt_hdr_normal_ptr[pkt_ind].ip_version   = (u_int32_t)pkt->iph.version;
    pkt_hdr_normal_ptr[pkt_ind].ip_tos       = (u_int32_t)pkt->iph.tos;
    pkt_hdr_normal_ptr[pkt_ind].ip_tot_len   = (u_int32_t)pkt->iph.tot_len;
    pkt_hdr_normal_ptr[pkt_ind].ip_id        = (u_int32_t)pkt->iph.id;
    pkt_hdr_normal_ptr[pkt_ind].ip_frag_off  = (u_int32_t)pkt->iph.frag_off;
    pkt_hdr_normal_ptr[pkt_ind].ip_ttl       = (u_int32_t)pkt->iph.ttl;
    pkt_hdr_normal_ptr[pkt_ind].ip_protocol  = (u_int32_t)pkt->iph.protocol;
    pkt_hdr_normal_ptr[pkt_ind].ip_check     = (u_int32_t)pkt->iph.check;
    pkt_hdr_normal_ptr[pkt_ind].ip_saddr     = (u_int32_t)pkt->iph.saddr;
    pkt_hdr_normal_ptr[pkt_ind].ip_daddr     = (u_int32_t)pkt->iph.daddr;

    pkt_hdr_normal_ptr[pkt_ind].udp_source   = (u_int32_t)pkt->uh.source;
    pkt_hdr_normal_ptr[pkt_ind].udp_dest     = (u_int32_t)pkt->uh.dest; 
    pkt_hdr_normal_ptr[pkt_ind].udp_len      = (u_int32_t)pkt->uh.len; 
    pkt_hdr_normal_ptr[pkt_ind].udp_check    = (u_int32_t)pkt->uh.check; 
}


void swizzle_packet(struct pkt_hdr_batch* pkt_hdr_batch_ptr, struct pkt_hdr* pkt, unsigned pkt_ind)
{
       
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_dhost_1[pkt_ind], pkt->eh.ether_dhost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_dhost_2[pkt_ind], pkt->eh.ether_dhost + 4, 2); 
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_shost_1[pkt_ind], pkt->eh.ether_shost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_shost_2[pkt_ind], pkt->eh.ether_shost + 4, 2); 
    pkt_hdr_batch_ptr->ether_type[pkt_ind] = (u_int32_t)pkt->eh.ether_type; 

    pkt_hdr_batch_ptr->ip_version[pkt_ind]   = (u_int32_t)pkt->iph.version;
    pkt_hdr_batch_ptr->ip_tos[pkt_ind]       = (u_int32_t)pkt->iph.tos;
    pkt_hdr_batch_ptr->ip_tot_len[pkt_ind]   = (u_int32_t)pkt->iph.tot_len;
    pkt_hdr_batch_ptr->ip_id[pkt_ind]        = (u_int32_t)pkt->iph.id;
    pkt_hdr_batch_ptr->ip_frag_off[pkt_ind]  = (u_int32_t)pkt->iph.frag_off;
    pkt_hdr_batch_ptr->ip_ttl[pkt_ind]       = (u_int32_t)pkt->iph.ttl;
    pkt_hdr_batch_ptr->ip_protocol[pkt_ind]  = (u_int32_t)pkt->iph.protocol;
    pkt_hdr_batch_ptr->ip_check[pkt_ind]     = (u_int32_t)pkt->iph.check;
    pkt_hdr_batch_ptr->ip_saddr[pkt_ind]     = (u_int32_t)pkt->iph.saddr;
    pkt_hdr_batch_ptr->ip_daddr[pkt_ind]     = (u_int32_t)pkt->iph.daddr;

    pkt_hdr_batch_ptr->udp_source[pkt_ind]   = (u_int32_t)pkt->uh.source;
    pkt_hdr_batch_ptr->udp_dest[pkt_ind]     = (u_int32_t)pkt->uh.dest; 
    pkt_hdr_batch_ptr->udp_len[pkt_ind]      = (u_int32_t)pkt->uh.len; 
    pkt_hdr_batch_ptr->udp_check[pkt_ind]    = (u_int32_t)pkt->uh.check; 
}

void print_swizzled_packet(struct pkt_hdr_batch* batch_ptr, unsigned pkt_ind)
{
    cout << "\n------------------- " << pkt_ind << " -----------------------" << endl;
    cout << "Ether dhost 1: " << batch_ptr->ether_dhost_1[pkt_ind] << endl;
    cout << "Ether dhost 2: " << batch_ptr->ether_dhost_2[pkt_ind] << endl;
    cout << "Ether shost 1: " << batch_ptr->ether_shost_1[pkt_ind] << endl;
    cout << "Ether shost 2: " << batch_ptr->ether_shost_2[pkt_ind] << endl;
    cout << "Ether type: " << batch_ptr->ether_type[pkt_ind] << endl;
    cout << "--" << endl;
    cout << "IP version: " << batch_ptr->ip_version[pkt_ind] << endl;
    cout << "IP tos: " << batch_ptr->ip_tos[pkt_ind] << endl;
    cout << "IP tot_len: " << ntohs(batch_ptr->ip_tot_len[pkt_ind]) << endl;
    cout << "IP id: " << ntohs(batch_ptr->ip_id[pkt_ind]) << endl;
    cout << "IP frag_off: " << batch_ptr->ip_frag_off[pkt_ind] << endl;
    cout << "IP ttl: " << batch_ptr->ip_ttl[pkt_ind] << endl;
    cout << "IP protocol: " << batch_ptr->ip_protocol[pkt_ind] << endl;
    cout << "IP check: " << ntohs(batch_ptr->ip_check[pkt_ind]) << endl;
    cout << "IP saddr: " << ntohl(batch_ptr->ip_saddr[pkt_ind]) << endl;
    cout << "IP daddr: " << ntohl(batch_ptr->ip_daddr[pkt_ind]) << endl;
    cout << "--" << endl;
    cout << "UDP source: " << ntohs(batch_ptr->udp_source[pkt_ind]) << endl;
    cout << "UDP dest: " << ntohs(batch_ptr->udp_dest[pkt_ind]) << endl;
    cout << "UDP len: " << ntohs(batch_ptr->udp_len[pkt_ind]) << endl;
    cout << "UDP check: " << batch_ptr->udp_check[pkt_ind] << endl;
    cout << "--------------------------------------------------------------" << endl;
}

void print_normal_packet(struct pkt_hdr_normal* pkt)
{
    cout << "--------------------------------------------------------------" << endl;
    cout << "Ether dhost 1: " << pkt->ether_dhost_1 << endl;
    cout << "Ether dhost 2: " << pkt->ether_dhost_2 << endl;
    cout << "Ether shost 1: " << pkt->ether_shost_1 << endl;
    cout << "Ether shost 2: " << pkt->ether_shost_2 << endl;
    cout << "Ether type: " << pkt->ether_type << endl;
    cout << "--" << endl;
    cout << "IP version: " << pkt->ip_version << endl;
    cout << "IP tos: " << pkt->ip_tos << endl;
    cout << "IP tot_len: " << ntohs(pkt->ip_tot_len) << endl;
    cout << "IP id: " << ntohs(pkt->ip_id) << endl;
    cout << "IP frag_off: " << pkt->ip_frag_off << endl;
    cout << "IP ttl: " << pkt->ip_ttl << endl;
    cout << "IP protocol: " << pkt->ip_protocol << endl;
    cout << "IP check: " << ntohs(pkt->ip_check) << endl;
    cout << "IP saddr: " << ntohl(pkt->ip_saddr) << endl;
    cout << "IP daddr: " << ntohl(pkt->ip_daddr) << endl;
    cout << "--" << endl;
    cout << "UDP source: " << ntohs(pkt->udp_source) << endl;
    cout << "UDP dest: " << ntohs(pkt->udp_dest) << endl;
    cout << "UDP len: " << ntohs(pkt->udp_len) << endl;
    cout << "UDP check: " << pkt->udp_check << endl;
    cout << "--------------------------------------------------------------" << endl;
}

void init_ipv6_normal_requests(struct kernel_args* args, struct ipv6_prefix* prefix_arr, unsigned prefix_ind)
{
    unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * g_batch_size * sizeof(struct ipv6_pkt_hdr_normal));
    struct ipv6_pkt_hdr_normal* packet_buffer = NULL;
    CUdeviceptr gpu_packet_buffer = 0;
                                                                                                            
    cout << "== GPU Packet Swizzle Initialization ==" << endl;
    cout << "Number of batches = " << g_num_batches << endl;
    cout << "Batch size = " << g_batch_size << endl;
    cout << "Number of normal packets = " << total_num_requests << endl;
    cout << "Buffer size = " << buffer_alloc_size/1e6 << " MB" << endl;
                                                                                                        
    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_packet_buffer, buffer_alloc_size)); 
    packet_buffer = (ipv6_pkt_hdr_normal*)malloc(buffer_alloc_size);

    struct ipv6_pkt_hdr pkt;
    for (unsigned i=0; i<g_num_batches; ++i) {
        for (unsigned j=0; j<g_batch_size; ++j) {
            // Load in the actual packet
            ipv6_generate_dummy_packet(&pkt, &prefix_arr[prefix_ind]);
            prefix_ind = (prefix_ind+1) % IPV6_NUM_RAND_PREFIXES;
            unsigned ind = i*g_batch_size + j;
            ipv6_normal_packet(packet_buffer, &pkt, ind);
        }                                                                                                                                                   
    }

    assert(args);
    args->buffer_size = buffer_alloc_size;
    args->batch_size = g_batch_size;
    args->n_batches = g_num_batches;
    args->h_packet_buffer = (void*)packet_buffer;
    args->h_response_buffer = NULL;
    args->g_packet_buffer = gpu_packet_buffer;
    args->g_response_buffer = 0;
}


void init_ipv6_swizzle_requests(struct kernel_args* args, struct ipv6_prefix* prefix_arr, unsigned prefix_ind)
{
    unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * sizeof(struct ipv6_pkt_hdr_swizzle));

    struct ipv6_pkt_hdr_swizzle* packet_buffer = NULL;
    CUdeviceptr gpu_packet_buffer = 0;

    cout << "== GPU Packet Swizzle Initialization ==" << endl;
    cout << "Number of batches = " << g_num_batches << endl;
    cout << "Batch size = " << g_batch_size << endl;
    cout << "Number of swizzled packets = " << total_num_requests << endl;
    cout << "Buffer size = " << buffer_alloc_size/1e6 << " MB" << endl;

    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_packet_buffer, buffer_alloc_size)); 
    packet_buffer = (ipv6_pkt_hdr_swizzle*)malloc(buffer_alloc_size);
    

    struct ipv6_pkt_hdr pkt;
    for (unsigned i=0; i<g_num_batches; ++i) {
        for (unsigned j=0; j<g_batch_size; ++j) {
            // Load in the actual packet
            ipv6_generate_dummy_packet(&pkt, &prefix_arr[prefix_ind]);
            prefix_ind = (prefix_ind+1) % IPV6_NUM_RAND_PREFIXES;
            ipv6_swizzle_packet(&packet_buffer[i], &pkt, j);
        }
    }

    assert(args);
    args->buffer_size = buffer_alloc_size;
    args->batch_size = g_batch_size;
    args->n_batches = g_num_batches;
    args->h_packet_buffer = (void*)packet_buffer;
    args->h_response_buffer = NULL;
    args->g_packet_buffer = gpu_packet_buffer;
    args->g_response_buffer = 0;
}

void ipv6_swizzle_packet(struct ipv6_pkt_hdr_swizzle* pkt_hdr_batch_ptr, struct ipv6_pkt_hdr* pkt, unsigned pkt_ind)
{
    // ETH
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_dhost_1[pkt_ind], pkt->eh.ether_dhost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_dhost_2[pkt_ind], pkt->eh.ether_dhost + 4, 2); 
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_shost_1[pkt_ind], pkt->eh.ether_shost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_batch_ptr->ether_shost_2[pkt_ind], pkt->eh.ether_shost + 4, 2); 
    pkt_hdr_batch_ptr->ether_type[pkt_ind] = (u_int32_t)pkt->eh.ether_type; 

    // IPH
    pkt_hdr_batch_ptr->ip_vtc_flow[pkt_ind]   = (u_int32_t)pkt->iph.vtc_flow;
    pkt_hdr_batch_ptr->ip_payload_len[pkt_ind]   = (u_int32_t)pkt->iph.payload_len;
    pkt_hdr_batch_ptr->ip_proto[pkt_ind]   = (u_int32_t)pkt->iph.proto;
    pkt_hdr_batch_ptr->ip_hop_limits[pkt_ind]   = (u_int32_t)pkt->iph.hop_limits;
    memcpy(&pkt_hdr_batch_ptr->ip_saddr1[pkt_ind], &pkt->iph.src_addr[0], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_saddr2[pkt_ind], &pkt->iph.src_addr[4], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_saddr3[pkt_ind], &pkt->iph.src_addr[8], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_saddr4[pkt_ind], &pkt->iph.src_addr[12], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_daddr1[pkt_ind], &pkt->iph.dst_addr[0], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_daddr2[pkt_ind], &pkt->iph.dst_addr[4], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_daddr3[pkt_ind], &pkt->iph.dst_addr[8], 4);
    memcpy(&pkt_hdr_batch_ptr->ip_daddr4[pkt_ind], &pkt->iph.dst_addr[12], 4);

    // UDPH
    pkt_hdr_batch_ptr->udp_source[pkt_ind]   = (u_int32_t)pkt->uh.source;
    pkt_hdr_batch_ptr->udp_dest[pkt_ind]     = (u_int32_t)pkt->uh.dest; 
    pkt_hdr_batch_ptr->udp_len[pkt_ind]      = (u_int32_t)pkt->uh.len; 
    pkt_hdr_batch_ptr->udp_check[pkt_ind]    = (u_int32_t)pkt->uh.check; 
}

void ipv6_normal_packet(struct ipv6_pkt_hdr_normal* pkt_hdr_normal_ptr, struct ipv6_pkt_hdr* pkt, unsigned pkt_ind)
{
    // ETH
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_dhost_1, pkt->eh.ether_dhost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_dhost_2, pkt->eh.ether_dhost + 4, 2); 
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_shost_1, pkt->eh.ether_shost, 4); 
    memcpy((u_int8_t*)&pkt_hdr_normal_ptr[pkt_ind].ether_shost_2, pkt->eh.ether_shost + 4, 2); 
    pkt_hdr_normal_ptr[pkt_ind].ether_type = (u_int32_t)pkt->eh.ether_type; 
                                                                                                
    // IPH 
    pkt_hdr_normal_ptr[pkt_ind].ip_vtc_flow   = (u_int32_t)pkt->iph.vtc_flow;
    pkt_hdr_normal_ptr[pkt_ind].ip_payload_len   = (u_int32_t)pkt->iph.payload_len;
    pkt_hdr_normal_ptr[pkt_ind].ip_proto   = (u_int32_t)pkt->iph.proto;
    pkt_hdr_normal_ptr[pkt_ind].ip_hop_limits   = (u_int32_t)pkt->iph.hop_limits;
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_saddr1, &pkt->iph.src_addr[0], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_saddr2, &pkt->iph.src_addr[4], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_saddr3, &pkt->iph.src_addr[8], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_saddr4, &pkt->iph.src_addr[12], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_daddr1, &pkt->iph.dst_addr[0], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_daddr2, &pkt->iph.dst_addr[4], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_daddr3, &pkt->iph.dst_addr[8], 4);
    memcpy(&pkt_hdr_normal_ptr[pkt_ind].ip_daddr4, &pkt->iph.dst_addr[12], 4);
                                                                                                
    // UDPH
    pkt_hdr_normal_ptr[pkt_ind].udp_source   = (u_int32_t)pkt->uh.source;
    pkt_hdr_normal_ptr[pkt_ind].udp_dest     = (u_int32_t)pkt->uh.dest; 
    pkt_hdr_normal_ptr[pkt_ind].udp_len      = (u_int32_t)pkt->uh.len; 
    pkt_hdr_normal_ptr[pkt_ind].udp_check    = (u_int32_t)pkt->uh.check; 
}




void init_normal_requests(struct kernel_args* args, bool alloc_response)
{
    unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * g_batch_size * sizeof(struct pkt_hdr_normal));

    struct pkt_hdr_normal* packet_buffer = NULL;
    struct pkt_hdr_normal* response_buffer = NULL;
    CUdeviceptr gpu_packet_buffer = 0;
    CUdeviceptr gpu_response_buffer = 0;

    cout << "== GPU Packet Swizzle Initialization ==" << endl;
    cout << "Number of batches = " << g_num_batches << endl;
    cout << "Batch size = " << g_batch_size << endl;
    cout << "Number of normal packets = " << total_num_requests << endl;
    cout << "Buffer size = " << buffer_alloc_size/1e6 << " MB" << endl;
                                                                                                        
    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_packet_buffer, buffer_alloc_size)); 
    packet_buffer = (pkt_hdr_normal*)malloc(buffer_alloc_size);

    if (alloc_response) {
        checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_response_buffer, buffer_alloc_size)); 
        response_buffer = (pkt_hdr_normal*)malloc(buffer_alloc_size);
    }

    struct pkt_hdr pkt;
    for (unsigned i=0; i<g_num_batches; ++i) {
        for (unsigned j=0; j<g_batch_size; ++j) {
            // Load in the actual packet
            generate_dummy_packet(&pkt, 1);
            unsigned ind = i*g_batch_size + j;
            normal_packet(packet_buffer, &pkt, ind);
        }
    }
                                                                
    assert(args);
    args->buffer_size = buffer_alloc_size;
    args->batch_size = g_batch_size;
    args->n_batches = g_num_batches;
    args->h_packet_buffer = (void*)packet_buffer;
    args->h_response_buffer = (void*)response_buffer;
    args->g_packet_buffer = gpu_packet_buffer;
    args->g_response_buffer = gpu_response_buffer;
}

void init_swizzle_requests(struct kernel_args* args, bool alloc_response)
{
    CUresult res = CUDA_SUCCESS;
    unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * sizeof(struct pkt_hdr_batch));

    struct pkt_hdr_batch* packet_buffer = NULL;
    struct pkt_hdr_batch* response_buffer = NULL;
    CUdeviceptr gpu_packet_buffer = 0;
    CUdeviceptr gpu_response_buffer = 0;

    cout << "== GPU Packet Swizzle Initialization ==" << endl;
    cout << "Number of batches = " << g_num_batches << endl;
    cout << "Batch size = " << g_batch_size << endl;
    cout << "Number of swizzled packets = " << total_num_requests << endl;
    cout << "Buffer size = " << buffer_alloc_size/1e6 << " MB" << endl;

    checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_packet_buffer, buffer_alloc_size)); 
    packet_buffer = (pkt_hdr_batch*)malloc(buffer_alloc_size);

    if (alloc_response) {
        checkCudaErrors(cuMemAlloc_v2((long long unsigned int *)&gpu_response_buffer, buffer_alloc_size)); 
        response_buffer = (pkt_hdr_batch*)malloc(buffer_alloc_size);
    }

    cout << "SIZE OF IPHDR = " << sizeof(ip_header) << endl;
    cout << "SIZE OF MINE = " << sizeof(struct pkt_hdr) << endl;
    cout << "SIZE OF eh = " << sizeof(ether_header) << endl;
    cout << "SIZE OF iph = " << sizeof(ip_header) << endl;
    cout << "SIZE OF uh = " << sizeof(udp_header) << endl;
    
    unsigned pkt_to_print = 312;
    bool verbose = false;

    struct pkt_hdr pkt;
    for (unsigned i=0; i<g_num_batches; ++i) {
        for (unsigned j=0; j<g_batch_size; ++j) {
            // Load in the actual packet
            generate_dummy_packet(&pkt, 1);
            if (verbose && j == pkt_to_print) {
                print_pkt_hdr(&pkt);
            }
            swizzle_packet(&packet_buffer[i], &pkt, j);
        }
    }

    if (verbose)
        print_swizzled_packet(&packet_buffer[0], pkt_to_print);

    assert(args);
    args->buffer_size = buffer_alloc_size;
    args->batch_size = g_batch_size;
    args->n_batches = g_num_batches;
    args->h_packet_buffer = (void*)packet_buffer;
    args->h_response_buffer = (void*)response_buffer;
    args->g_packet_buffer = gpu_packet_buffer;
    args->g_response_buffer = gpu_response_buffer;
}

unsigned g_pkt_id=0; 
void ipv6_generate_dummy_packet(struct ipv6_pkt_hdr* pkt, struct ipv6_prefix* pfa)
{
    unsigned pkt_size = sizeof(struct ipv6_pkt_hdr);
    set_mac(&pkt->eh.ether_shost[0], src_mac_arr[0][0]);                                          
    set_mac(&pkt->eh.ether_dhost[0], dst_mac_arr[0][0]);
    pkt->eh.ether_type = htons(0x0800);

    pkt->iph.vtc_flow = 0;
    pkt->iph.payload_len = 2 + sizeof(int) + sizeof(LL);
    pkt->iph.proto = IPPROTO_IPV6;
    pkt->iph.hop_limits = 64;
    
    memcpy(pkt->iph.src_addr, pfa->bytes, IPV6_ADDR_LEN);
    memcpy(pkt->iph.dst_addr, pfa->bytes, IPV6_ADDR_LEN);

    // UDP
    pkt->uh.source = htons(9191);
    pkt->uh.dest = htons(9960);
    pkt->uh.len = htons(pkt_size - sizeof(ether_header) - sizeof(struct ipv6_hdr));
    pkt->uh.check = 0; /* It must be 0 to compute the checksum */
}

void generate_dummy_packet(struct pkt_hdr* pkt, unsigned gen_type)
{
    unsigned pkt_size = sizeof(struct pkt_hdr);
    if (gen_type == 0) {
        u_int32_t src_ip = 0xC0A80002 /* from 192.168.0.2 */;
        u_int32_t dst_ip =  0xC0A80104 /* 192.168.1.4 */;


        // Ethernet
        pkt->eh.ether_type = htons(0x0800);
        pkt->eh.ether_shost[0] = 0x68;
        pkt->eh.ether_shost[1] = 0x05;
        pkt->eh.ether_shost[2] = 0xCA;
        pkt->eh.ether_shost[3] = 0x13;
        pkt->eh.ether_shost[4] = 0xCE;
        pkt->eh.ether_shost[5] = 0x79;
        pkt->eh.ether_dhost[0] = 0x68;
        pkt->eh.ether_dhost[1] = 0x05;
        pkt->eh.ether_dhost[2] = 0xCA;
        pkt->eh.ether_dhost[3] = 0x1B;
        pkt->eh.ether_dhost[4] = 0x1E;
        pkt->eh.ether_dhost[5] = 0x66;

        // IP
        //pkt->iph.ihl = 5;
        pkt->iph.version = 4;
        pkt->iph.tos = 0;
        pkt->iph.tot_len = htons(pkt_size - sizeof(ether_header));
        pkt->iph.id = htons(g_pkt_id++);
        pkt->iph.ttl = 64;
        pkt->iph.frag_off = htons(0);
        pkt->iph.protocol = IPPROTO_UDP;
        pkt->iph.daddr = htonl(dst_ip);
        pkt->iph.saddr = htonl(src_ip);
        pkt->iph.check = wrapsum(in_cksum((unsigned char *)&pkt->iph, sizeof(struct ip_header), 0));

        // UDP
        pkt->uh.source = htons(9191);
        pkt->uh.dest = htons(9960);
        pkt->uh.len = htons(pkt_size - sizeof(ether_header) - sizeof(ip_header));
        pkt->uh.check = 0; /* It must be 0 to compute the checksum */

        //i = sizeof(struct ether_header) + sizeof(struct ip_header) + sizeof(struct udp_header);
        /*udp_header->check = wrapsum(in_cksum((unsigned char *)udp_header, sizeof(struct udp_header),
                                             in_cksum((unsigned char *)&buffer[i], send_len-i,
                              in_cksum((unsigned char *)&ip_header->saddr,
                                   2*sizeof(ip_header->saddr),
                                   IPPROTO_UDP + ntohs(udp_header->len)))));*/
    } else if (gen_type == 1) {

		set_mac(&pkt->eh.ether_shost[0], src_mac_arr[0][0]);
		set_mac(&pkt->eh.ether_dhost[0], dst_mac_arr[0][0]);
        pkt->eh.ether_type = htons(0x0800);
    
        pkt->iph.version = 0x40 | 0x05;
		pkt->iph.tos = 0;
        pkt->iph.tot_len = htons(pkt_size - sizeof(ether_header));
        pkt->iph.id = htons(g_pkt_id++);
        pkt->iph.ttl = 64;
        pkt->iph.frag_off = htons(0);
        pkt->iph.protocol = IPPROTO_UDP;
		pkt->iph.saddr = htonl(fastrand(&rss_seed));
        pkt->iph.daddr = htonl(fastrand(&rss_seed));

        pkt->iph.check = wrapsum(in_cksum((unsigned char *)&pkt->iph, sizeof(struct ip_header), 0));
                                                                                                     
        // UDP
        pkt->uh.source = htons(9191);
        pkt->uh.dest = htons(9960);
        pkt->uh.len = htons(pkt_size - sizeof(ether_header) - sizeof(ip_header));
        pkt->uh.check = 0; /* It must be 0 to compute the checksum */

		if (g_pkt_id < 4) {
			print_pkt_hdr(pkt);		
		}



    } else {
        cout << "Error: Unknown gen_type = " << gen_type << endl;
        abort();
    }
    return;
}




int in_cksum(unsigned char *buf, unsigned nbytes, int sum) 
{
    uint i;

    /* Checksum all the pairs of bytes first... */
    for (i = 0; i < (nbytes & ~1U); i += 2) {
        sum += (u_int16_t) ntohs(*((u_int16_t *)(buf + i)));
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

static u_int32_t wrapsum (u_int32_t sum) 
{
  sum = ~sum & 0xFFFF;
  return htons(sum);
}


void print_pkt_hdr(struct pkt_hdr* hdr)
{

    printf("Packet header contents: \n");

    /***** ETHERNET HEADER *****/
    printf("\t==Ethernet header==\n");
    printf("\t\tDest: ");
    print_mac_arr(hdr->eh.ether_dhost);
    printf("\t\tSource: ");
    print_mac_arr(hdr->eh.ether_shost);
    printf("\t\tType: %hx\n", hdr->eh.ether_type);
    /***** END ETHERNET HEADER *****/

    /***** IP HEADER *****/
    printf("\t==IP header==\n");
    printf("\t\tVersion+hdr_len: %hhu\n", hdr->iph.version);
    printf("\t\tTOS: %hhu\n", hdr->iph.tos);
    printf("\t\tTotal Length: %hu\n", ntohs(hdr->iph.tot_len));
    printf("\t\tID: %hu\n", ntohs(hdr->iph.id));
    printf("\t\tFrag_off: %hu\n", hdr->iph.frag_off);
    printf("\t\tTTL: %hhu\n", hdr->iph.ttl);
    printf("\t\tProtocol: %hhu\n", hdr->iph.protocol);
    printf("\t\tchecksum: %hu\n", ntohs(hdr->iph.check));
    printf("\t\tSource address: ");
    print_ip_addr(ntohl(hdr->iph.saddr));
    printf("\t\tDest address: ");
    print_ip_addr(ntohl(hdr->iph.daddr));
    /***** END IP HEADER *****/

    /***** UDP HEADER *****/
    printf("\t==UDP header==\n");
    printf("\t\tSource port: %hu\n", ntohs(hdr->uh.source));
    printf("\t\tDest port: %hu\n", ntohs(hdr->uh.dest));
    printf("\t\tLength: %hu\n", ntohs(hdr->uh.len));
    printf("\t\tChecksum: %hu\n", hdr->uh.check);
    /***** END UDP HEADER *****/
}


//////////////////////////

bool inline findModulePath(const char *module_file, string &module_path, string &ptx_source)
{
    char *actual_path = sdkFindFilePath(module_file, g_file_path);

    if (actual_path) {
        module_path = actual_path;
    } else {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }

    if (module_path.empty()) {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    } else {
        printf("> findModulePath <%s>\n", module_path.c_str());
        if (module_path.rfind(".ptx") != string::npos) {
            FILE *fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *buf = new char[file_size+1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }
        return true;
    }
}


static CUresult initCUDA(unsigned dev_num, CUfunction *kernel, const char* kernel_name)
{
    CUfunction cuFunction = 0;
    CUresult status;
    int major = 0, minor = 0;
    char deviceName[100];
    string module_path, ptx_source;

    checkCudaErrors(cuInit(0));    
    
    checkCudaErrors(cuDeviceGet(&cuDevice, dev_num));

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
    printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:             %s\n", (totalGlobalMem > (unsigned long long)4*1024*1024*1024L) ? "YES" : "NO");

    status = cuCtxCreate(&cuContext, 0, cuDevice);

    if (CUDA_SUCCESS != status)
        goto Error;

    // first search for the module path before we load the results
    if (!findModulePath(PTX_FILE, module_path, ptx_source)) {
        if (!findModulePath(CUBIN_FILE, module_path, ptx_source)) {
            printf("> findModulePath could not find ptx or cubin\n");
            status = CUDA_ERROR_NOT_FOUND;
            goto Error;
        }
    } else {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    if (module_path.rfind("ptx") != string::npos) {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;

        status = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
    } else {
        status = cuModuleLoad(&cuModule, module_path.c_str());
    }

    if (CUDA_SUCCESS != status)
        goto Error;

    status = cuModuleGetFunction(&cuFunction, cuModule, kernel_name);

    if (CUDA_SUCCESS != status)
        goto Error;

    *kernel = cuFunction;
    return CUDA_SUCCESS;

Error:
    cuCtxDestroy(cuContext);
    return status;
}


