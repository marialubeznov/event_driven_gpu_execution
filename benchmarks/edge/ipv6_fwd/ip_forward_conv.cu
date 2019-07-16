#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
//#include <matrix_mul_kernel.cu>
#include <edge_cuda.h>

#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#include <util.h>
#include <ipv4.h>
#include <ipv6.h>
#include <arpa/inet.h>
#include <common.h>

#define CEIL(x, y) ( (x)/(y) + ( (x)%(y) ? 1 : 0 ) )
#define MAX(x, y) ( (x)>(y) ? (x) : (y) )
#define IPV6_REG_NUM 16

/**< [xia-router0 - xge0,1,2,3], [xia-router1 - xge0,1,2,3] */
LL src_mac_arr[2][4] = {{0x36d3bd211b00, 0x37d3bd211b00, 0xa8d6a3211b00, 0xa9d6a3211b00},
                        {0x44d7a3211b00, 0x45d7a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00}};

/**< [xia-router2 - xge0,1,4,5], [xia-router2 - xge2,3,6,7] */
LL dst_mac_arr[2][4] = {{0x6c10bb211b00, 0x6d10bb211b00, 0xc8a610ca0568, 0xc9a610ca0568},
                        {0x64d2bd211b00, 0x65d2bd211b00, 0xa2a610ca0568, 0xa3a610ca0568}};

uint64_t rss_seed = 0xdeadbeef;

int nStreams = 32;
cudaStream_t* streams = NULL; 
unsigned gTimerEventPeriod = 1000;
unsigned gTimerEventPeriodBatch = 200;
bool swizzle = false;
int n_batches = 1;
int n_requests_per_batch = 32;
bool save_regs = true;

int schedule_batch_size = 64;

#define CEIL(x, y) ( (x)/(y) + ( (x)%(y) ? 1 : 0 ) )
#define BLOCK_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define DEFAULT_MATRIX_SIZE 128

struct kernel_args {
    unsigned buffer_size;
    unsigned batch_size;
    unsigned n_batches;
    void* h_packet_buffer; 
    void* h_response_buffer; 
    void* g_packet_buffer;
    void* g_response_buffer;
};

struct MemcParam
{
    struct pkt_hdr_normal* packet_buf;
    int* gpu_tbl24;
    int* gpu_tbl8;
    unsigned n;
    int* reg_buffer;
    bool save_regs;
};

struct MemcParamSwizzle
{
    struct pkt_hdr_batch* packet_buf;
    int* gpu_tbl24;
    int* gpu_tbl8;
    unsigned n;
};

enum RunConfig {
    BASE_IPV6=0,
    EVENT_IPV6, 
    TIMER_IPV6,
    EVENT_TIMER_BG_IPV6,
    EVENT_TIMER_BATCH_BG_IPV6,
    BG_TASK
};

enum ScheduleType {
    SINGLE=0,
    TIMER, 
    BATCH
};

enum BgTaskType {
    CONV=0,
    MATRIX_MUL,
    BACKPROP,
	BFS
};

int bg_task_type = MATRIX_MUL;

#define CU_CHECK_ERR(err)                                                       \
    if ( err != cudaSuccess ) {                                                 \
        printf("CUDA Error: %s\n", cudaGetErrorString(cudaGetLastError()));     \
        abort();                                                                \
    }

void randomInit(float* data, int size);
void configureParamMem(MemcParam* paramMem, size_t totalBufferSize, size_t batchSize, size_t maxBatches, struct kernel_args* args);
void configureParamMemSwizzle(MemcParamSwizzle* paramMem, size_t totalBufferSize, size_t batchSize, size_t maxBatches, struct kernel_args* args);
void PrintMatrices(float* h_A, float* h_B, float* h_C, int dimAx, int dimAy, int dimBx, int dimBy, int dimCx, int dimCy);
int run_conv_kernel(int argc, char **argv, bool block, bool warmup);
int IpForwardEDGE(int argc, char** argv, bool RunBackgroundTask, ScheduleType scheduleType, bool swizzle);
int IpForwardBase(int argc, char** argv, bool swizzle);
void generate_dummy_packet(struct pkt_hdr* pkt, unsigned gen_type);
unsigned init_normal_requests(struct kernel_args* args, bool alloc_response, int g_batch_size, int g_num_batches);
unsigned init_swizzle_requests(struct kernel_args* args, bool alloc_response, int g_batch_size, int g_num_batches);
void normal_packet(struct pkt_hdr_normal* pkt_hdr_normal_ptr, struct pkt_hdr* pkt, unsigned pkt_ind);
int in_cksum(unsigned char *buf, unsigned nbytes, int sum) ;
void swizzle_packet(struct pkt_hdr_batch* pkt_hdr_batch_ptr, struct pkt_hdr* pkt, unsigned pkt_ind);
static u_int32_t wrapsum (u_int32_t sum) ;
struct rte_lpm *ipv4_init();
void randomInit(float* data, int size);
int MatrixMulBase(int argc, char** argv, bool block);
unsigned init_ipv6_normal_requests(struct kernel_args* args, struct ipv6_prefix* prefix_arr, unsigned prefix_ind, int g_batch_size, int g_num_batches);
void ipv6_normal_packet(struct ipv6_pkt_hdr_normal* pkt_hdr_normal_ptr, struct ipv6_pkt_hdr* pkt, unsigned pkt_ind);
void ipv6_generate_dummy_packet(struct ipv6_pkt_hdr* pkt, struct ipv6_prefix* pfa);
static u_int32_t wrapsum (u_int32_t sum) ;
struct rte_lpm6 *ipv6_init();

//kernels
__global__ void ipv6_fwd_kernel(ipv6_pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts, int* reg_buffer, bool save_regs);
__global__ void ipv6_fwd_kernel_save_regs(ipv6_pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts, int* reg_buffer, bool save_regs);
extern "C" __global__ void matrixMul( float* C, float* A, float* B, int wA, int wB);

void _filterActs(float *images, int images_cols, int images_rows, float *filters, int filters_cols,
                int filters_rows,  float *targets, int targets_cols, int targets_rows,
                int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                int numImgColors, int numGroups, float scaleTargets, float scaleOutput, int conv, cudaStream_t stream, bool warmup);
void run_backprop(int argc, char **argv, bool block);
extern "C" int setup(int argc, char** argv);

int bfs_main(int argc, char** argv);

void run_bfs(int argc, char **argv, bool block) {
    printf("MARIA inside run_bfs\n");
    bfs_main(argc, argv);
}

int div_up(int n, int d) {
    return n / d + (((n < 0) ^ (d > 0)) && (n % d));
} 

int main(int argc, char** argv) {
    std::cout << "=== EDGE BEGIN ===" << std::endl;
    int ret = 0;
    RunConfig testToRun = EVENT_IPV6;
    int opt;
    opterr = 0;

    int count = 0;
    while( (opt = getopt(argc, argv, "t:s:g:q:z:p:c:b:n:g:i:") ) != -1 ) {
        switch(opt) {
            case 't': 
                count += 2;
                testToRun = (RunConfig)atoi(optarg);
                break;
            case 'p':
                count += 2;
                gTimerEventPeriod = atoi(optarg);
                break;
            case 'i':
                count += 2;
                gTimerEventPeriodBatch = atoi(optarg);
                break;
            case 's':
                count += 2;
                swizzle = atoi(optarg);
                break;
            case 'n':
                count += 2;
                n_batches = atoi(optarg);
                break;
            case 'b':
                count += 2;
                n_requests_per_batch = atoi(optarg);
                break;
            case 'g':
                count += 2;
                bg_task_type = atoi(optarg);
                break;
            default: 
                std::cout << "Error: Unknown parameter: " << opt << std::endl;
                abort();
        }
    }

    streams = new cudaStream_t[nStreams];
    for( unsigned i=0; i<nStreams; ++i ) {
        CU_CHECK_ERR( cudaStreamCreate(&streams[i]) ); 
    }

    char** modArgv = &argv[count];
    int modArgc = argc-count;
    
    switch( testToRun ) {
        case BASE_IPV6:
            ret = IpForwardBase(modArgc, modArgv, swizzle);
            break;
        case EVENT_IPV6:
            ret = IpForwardEDGE(modArgc, modArgv, false, SINGLE, swizzle);
            break;
        case TIMER_IPV6:
            ret = IpForwardEDGE(modArgc, modArgv, false, TIMER, swizzle);
            break;
        case EVENT_TIMER_BG_IPV6: 
            ret = IpForwardEDGE(modArgc, modArgv, true, TIMER, swizzle);
        	break;
        case EVENT_TIMER_BATCH_BG_IPV6: 
            ret = IpForwardEDGE(modArgc, modArgv, true, BATCH, swizzle);
            break;
        case BG_TASK: 
            printf("Running only background task: %d \n", bg_task_type);
            if (bg_task_type == CONV) {
                printf("Running background task: conv\n");
                run_conv_kernel(argc, argv, true, false);
            }
            if (bg_task_type == MATRIX_MUL) {
                MatrixMulBase(argc, argv, true);
            }
            if (bg_task_type == BACKPROP) {
                run_backprop(argc, argv, true);
            }
			if (bg_task_type == BFS) {
                run_bfs(argc, argv, true);
            }
            ret = 0;
            break;
        default:
            std::cout << "Error: Undefined test configuration # (" << testToRun << ")" << std::endl;
            break;
    }

    if( ret ) {
        std::cout << "Error running test " << testToRun << " - Error=" << ret << std::endl;
    }

    std::cout << "=== EDGE END ===" << std::endl;
    return ret;
}

int IpForwardBase(int argc, char** argv, bool swizzle) {
    printf("IpForward EDGE Base test. Swizzle = %d\n", swizzle);
    struct kernel_args k_args;
    struct rte_lpm6 *lpm;

    int num_prefixes = IPV6_NUM_RAND_PREFIXES;
    int prefix_mem_size = num_prefixes * sizeof(struct ipv6_prefix);
    struct ipv6_prefix *prefix_arr = (struct ipv6_prefix*)malloc(prefix_mem_size);

    int mem_size = sizeof(*lpm) + (sizeof(lpm->tbl8[0]) *RTE_LPM6_TBL8_GROUP_NUM_ENTRIES * IPV6_NUM_TBL8);
    int rules_size = sizeof(struct rte_lpm6_rule) * 100000;

    /* Allocate memory to store the LPM data structures. Zero out counters. */
    lpm = (struct rte_lpm6 *) lpm6_hrd_malloc_socket(RTE_LPM6_SHM_KEY,
            mem_size, 0);

    int prefix_arr_i = rand() % IPV6_NUM_RAND_PREFIXES;
    printf("Mem init trick - do ipv6_init in CPU\n");
    CU_CHECK_ERR( edgeExtraipv6(1, (void*)lpm, IPV6_XIA_R2_PORT_MASK, (void*)prefix_arr, 1,n_requests_per_batch, n_batches  ) );

    init_ipv6_normal_requests(&k_args, prefix_arr, prefix_arr_i,n_requests_per_batch, n_batches);

    /**< rte_lpm_tbl24_entry ~ rte_lpm_tbl8_entry ~ uint16_t */
    int entry_sz = sizeof(struct rte_lpm6_tbl_entry);
    int tbl24_bytes = RTE_LPM6_TBL24_NUM_ENTRIES * entry_sz;
    int tbl8_bytes = (IPV6_NUM_TBL8 * RTE_LPM6_TBL8_GROUP_NUM_ENTRIES) * entry_sz;
    
    int* gpu_tbl24 = 0;
    int* gpu_tbl8 = 0;
    
    /**< Alloc and copy tbl24 and tbl8 arrays to GPU memory */
    printf("\tGPU master: alloc tbl24 (size = %lf MB) on device\n", (float)tbl24_bytes / 1e6);
    CU_CHECK_ERR(cudaMalloc(&gpu_tbl24, tbl24_bytes)); 
    CU_CHECK_ERR(cudaMemcpy(gpu_tbl24, lpm->tbl24, tbl24_bytes, cudaMemcpyHostToDevice));

    printf("\tGPU master: alloc tbl8 (size = %lf MB) on device\n", (float)tbl8_bytes / 1e6);
    CU_CHECK_ERR(cudaMalloc(&gpu_tbl8, tbl8_bytes)); 
    CU_CHECK_ERR(cudaMemcpy(gpu_tbl8, lpm->tbl8, tbl8_bytes, cudaMemcpyHostToDevice));

    CU_CHECK_ERR(cudaMemcpy(k_args.g_packet_buffer, k_args.h_packet_buffer, k_args.buffer_size, cudaMemcpyHostToDevice));
               
    //launch kernel                                                                                                                   
    dim3 block(n_requests_per_batch, 1, 1);
    dim3 grid(n_batches, 1, 1);

    unsigned n_packets = k_args.n_batches * k_args.batch_size;
    ipv6_fwd_kernel<<<grid, block>>>((ipv6_pkt_hdr_normal*)k_args.g_packet_buffer, (uint16_t *)gpu_tbl24, (uint16_t *)gpu_tbl8, n_packets, NULL, false);

    cudaDeviceSynchronize();
    return 0;


    
}

int IpForwardEDGE(int argc, char** argv, bool RunBackgroundTask, ScheduleType scheduleType, bool swizzle) {
    printf("IpForward EDGE Test RunBackgroundTask: %d, ScheduleType: %d, Swizzle: %d\n", RunBackgroundTask, scheduleType, swizzle);

    int MaxEventsNum = n_batches;
    unsigned single_buffer_alloc_size = (n_requests_per_batch * sizeof(struct ipv6_pkt_hdr_normal));

    struct kernel_args *k_args = (struct kernel_args *)malloc(MaxEventsNum*sizeof(struct kernel_args));

    struct rte_lpm6 *lpm;
    //struct ipv6_prefix *prefix_arr;
    int num_prefixes = IPV6_NUM_RAND_PREFIXES;
    int prefix_mem_size = num_prefixes * sizeof(struct ipv6_prefix);
    struct ipv6_prefix *prefix_arr = (struct ipv6_prefix*)malloc(prefix_mem_size);

    int mem_size = sizeof(*lpm) + (sizeof(lpm->tbl8[0]) *RTE_LPM6_TBL8_GROUP_NUM_ENTRIES * IPV6_NUM_TBL8);
    int rules_size = sizeof(struct rte_lpm6_rule) * 100000;

    /* Allocate memory to store the LPM data structures. Zero out counters. */
    lpm = (struct rte_lpm6 *) lpm6_hrd_malloc_socket(RTE_LPM6_SHM_KEY,
            mem_size, 0);

    //lpm = ipv6_init(IPV6_XIA_R2_PORT_MASK, &prefix_arr, 1);
    int prefix_arr_i = rand() % IPV6_NUM_RAND_PREFIXES;
     printf("Mem init trick - do ipv6_init in CPU\n");
    CU_CHECK_ERR( edgeExtraipv6(1, (void*)lpm, IPV6_XIA_R2_PORT_MASK, (void*)prefix_arr, 1,n_requests_per_batch, n_batches  ) );

    // initialize host memory
    for( unsigned batch=0; batch<MaxEventsNum; ++batch ) {
        unsigned buffer_alloc_size = init_ipv6_normal_requests(&k_args[batch], prefix_arr, prefix_arr_i,n_requests_per_batch, n_batches);
        //printf("Generated input packets for batch %d. buffer_alloc_size = %lld\n", single_buffer_alloc_size);
    }

    int entry_sz = sizeof(struct rte_lpm6_tbl_entry);
    //int entry_sz = sizeof(struct rte_lpm_tbl24_entry);
    int tbl24_bytes = RTE_LPM6_TBL24_NUM_ENTRIES * entry_sz;
    int tbl8_bytes = (IPV6_NUM_TBL8 * RTE_LPM6_TBL8_GROUP_NUM_ENTRIES) * entry_sz;
    
    int* gpu_tbl24 = 0;
    int* gpu_tbl8 = 0;
    
    /**< Alloc and copy tbl24 and tbl8 arrays to GPU memory */
    printf("\tGPU master: alloc tbl24 (size = %lf MB) on device\n", (float)tbl24_bytes / 1e6);
    CU_CHECK_ERR(cudaMalloc(&gpu_tbl24, tbl24_bytes)); 
    CU_CHECK_ERR(cudaMemcpy(gpu_tbl24, lpm->tbl24, tbl24_bytes, cudaMemcpyHostToDevice));

    printf("\tGPU master: alloc tbl8 (size = %lf MB) on device\n", (float)tbl8_bytes / 1e6);
    CU_CHECK_ERR(cudaMalloc(&gpu_tbl8, tbl8_bytes)); 
    CU_CHECK_ERR(cudaMemcpy(gpu_tbl8, lpm->tbl8, tbl8_bytes, cudaMemcpyHostToDevice));

    //------------------------------------------//

    // setup execution parameters
    dim3 block(min(n_requests_per_batch, MAX_THREADS_PER_BLOCK), 1, 1); 
    dim3 grid(div_up(n_requests_per_batch, MAX_THREADS_PER_BLOCK), 1, 1);    

    // Register the event kernel
    int eventId;
        eventId = cudaRegisterEvent((void*)ipv6_fwd_kernel, (void*)ipv6_fwd_kernel_save_regs, grid, block, 0); 

        //Setup the arguments
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(struct ipv6_pkt_hdr_normal*), 0) ); //packet buffer
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(uint16_t*), 8) ); //gpu_tbl24
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(uint16_t*), 16) ); //gpu_tbl8
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 24) ); //n_packets
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(int*), 32) ); //reg_buffer
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(bool), 40) ); //save_regs

        // Configure the parameter memory
        unsigned paramSize = sizeof(MemcParam);
        MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, MaxEventsNum, false);
        printf("MARIA DEBUG allocated param mem = %lld \n", paramMem);
        configureParamMem(paramMem, single_buffer_alloc_size, n_requests_per_batch, MaxEventsNum, k_args);

        //copy from host to gpu
        MemcParam* curParam = paramMem;
        for( unsigned batch=0; batch<MaxEventsNum; ++batch ) {
            CU_CHECK_ERR(cudaMemcpy(curParam->packet_buf, k_args[batch].h_packet_buffer, single_buffer_alloc_size, cudaMemcpyHostToDevice)); 
            curParam->gpu_tbl24 = gpu_tbl24;
            curParam->gpu_tbl8 = gpu_tbl8;
            curParam++;
        }

        //////////////////////HACK
        paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, MaxEventsNum, true);
        ////////////////////////
      
     
    
    

    printf("Scheduling EDGE event\n");
    if (scheduleType == TIMER) {
        //Schedule the event kernel to run on a timer
        CU_CHECK_ERR( cudaScheduleTimerEvent(eventId, gTimerEventPeriod) );
    } else if (scheduleType==SINGLE) { //event
        for( unsigned batch=0; batch<MaxEventsNum; ++batch ) {
            CU_CHECK_ERR( cudaScheduleEvent(eventId) );
        };
    } else { //batch
        CU_CHECK_ERR( cudaScheduleEventTimerBatch(eventId, gTimerEventPeriod, schedule_batch_size, gTimerEventPeriodBatch) );
    };
    
    if (RunBackgroundTask) {
    	if (bg_task_type == CONV) {
        	printf("Running background task: conv\n");
        	run_conv_kernel(argc, argv, false, false);
    	}
    	if (bg_task_type == MATRIX_MUL) {
        	MatrixMulBase(argc, argv, false);
    	}
    	if (bg_task_type == BACKPROP) {
        	run_backprop(argc, argv, true);
    	}
        if (bg_task_type == BFS) {
            run_bfs(argc, argv, true);
        }
    }
        
    CU_CHECK_ERR( cudaDeviceSynchronize() );
    std::cout << "Success!" << std::endl;
    
    return 0;

}

void configureParamMem(MemcParam* paramMem, size_t totalBufferSize, size_t batchSize, size_t maxBatches, struct kernel_args* args)
{
    MemcParam* curParam = paramMem;

    for( unsigned batch=0; batch<maxBatches; ++batch ) {
        //CU_CHECK_ERR( cudaMalloc((void**)&curParam->packet_buf, totalBufferSize) );
        curParam->packet_buf = (struct pkt_hdr_normal *)args[batch].g_packet_buffer;
        int reg_buffer_size = 32 * IPV4_REG_NUM * 512;
        CU_CHECK_ERR(cudaMalloc(&curParam->reg_buffer, reg_buffer_size));
        curParam->n = batchSize;
        curParam->save_regs = true;
        curParam++;
    }
}

void configureParamMemSwizzle(MemcParamSwizzle* paramMem, size_t totalBufferSize, size_t batchSize, size_t maxBatches, struct kernel_args* args)
{
    MemcParamSwizzle* curParam = paramMem;

    for( unsigned batch=0; batch<maxBatches; ++batch ) {
        //CU_CHECK_ERR( cudaMalloc((void**)&curParam->packet_buf, totalBufferSize) );
        curParam->packet_buf = (struct pkt_hdr_batch*)args[batch].g_packet_buffer;
        curParam->n = batchSize;
        //curParam->save_regs = save_regs;
        curParam++;
    }
}

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

//////////////////////////////////////////////////////IP-FORWARDING related functions /////////////////////////////////////////////

unsigned init_normal_requests(struct kernel_args* args, bool alloc_response, int g_batch_size, int g_num_batches)
{
    unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * g_batch_size * sizeof(struct pkt_hdr_normal));

    struct pkt_hdr_normal* packet_buffer = NULL;
    struct pkt_hdr_normal* response_buffer = NULL;
    struct pkt_hdr_normal*  gpu_packet_buffer = NULL;
    struct pkt_hdr_normal*  gpu_response_buffer = NULL;
                                                                                                       
    CU_CHECK_ERR(cudaMalloc(&gpu_packet_buffer, buffer_alloc_size));
    packet_buffer = (pkt_hdr_normal*)malloc(buffer_alloc_size);

    if (alloc_response) {
        CU_CHECK_ERR(cudaMalloc(&gpu_response_buffer, buffer_alloc_size)); 
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
    return buffer_alloc_size;
}

unsigned init_swizzle_requests(struct kernel_args* args, bool alloc_response, int g_batch_size, int g_num_batches)
{
    int res = CUDA_SUCCESS;
    unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * sizeof(struct pkt_hdr_batch));

    struct pkt_hdr_batch* packet_buffer = NULL;
    struct pkt_hdr_batch* response_buffer = NULL;
    struct pkt_hdr_normal* gpu_packet_buffer = 0;
    struct pkt_hdr_normal* gpu_response_buffer = 0;

    CU_CHECK_ERR(cudaMalloc(&gpu_packet_buffer, buffer_alloc_size)); 
    packet_buffer = (pkt_hdr_batch*)malloc(buffer_alloc_size);

    if (alloc_response) {
        CU_CHECK_ERR(cudaMalloc(&gpu_response_buffer, buffer_alloc_size)); 
        response_buffer = (pkt_hdr_batch*)malloc(buffer_alloc_size);
    }
  
    unsigned pkt_to_print = 312;
    bool verbose = false;

    struct pkt_hdr pkt;
    for (unsigned i=0; i<g_num_batches; ++i) {
        for (unsigned j=0; j<g_batch_size; ++j) {
            // Load in the actual packet
            generate_dummy_packet(&pkt, 1);
            if (verbose && j == pkt_to_print) {
                //print_pkt_hdr(&pkt);
            }
            swizzle_packet(&packet_buffer[i], &pkt, j);
        }
    }

    if (verbose)
        //print_swizzled_packet(&packet_buffer[0], pkt_to_print);

    assert(args);
    args->buffer_size = buffer_alloc_size;
    args->batch_size = g_batch_size;
    args->n_batches = g_num_batches;
    args->h_packet_buffer = (void*)packet_buffer;
    args->h_response_buffer = (void*)response_buffer;
    args->g_packet_buffer = gpu_packet_buffer;
    args->g_response_buffer = gpu_response_buffer;
    return buffer_alloc_size;
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

unsigned init_ipv6_normal_requests(struct kernel_args* args, struct ipv6_prefix* prefix_arr, unsigned prefix_ind, int g_batch_size, int g_num_batches)
{
       unsigned total_num_requests = g_batch_size * g_num_batches;
    unsigned buffer_alloc_size = (g_num_batches * g_batch_size * sizeof(struct ipv6_pkt_hdr_normal));
    struct ipv6_pkt_hdr_normal* packet_buffer = NULL;
    struct ipv6_pkt_hdr_normal*  gpu_packet_buffer = NULL;
                                                                                                            

                                                                                                        
     cudaMalloc(&gpu_packet_buffer, buffer_alloc_size); 
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
    return buffer_alloc_size;
       

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

unsigned g_pkt_id=0; 
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

        //if (g_pkt_id < 4) {
        //    print_pkt_hdr(pkt);     
        //}



    } else {
        //cout << "Error: Unknown gen_type = " << gen_type << endl;
        abort();
    }
    return;
}
//unsigned g_pkt_id=0; 
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

/////////////////////////////////////BACKPROP//////////////////////////////////////////////
void run_backprop(int argc, char **argv, bool block) {
    printf("MARIA inside run_backprop\n");
    setup(argc, argv);
}

////////////////////////////////////////CONVOLUTION//////////////////////////////////////////////////

int run_conv_kernel(int argc, char** argv, bool block, bool warmup)
{
    float *h_images;
    float *h_filters;
    float *h_targets;

    float *d_images;
    float *d_filters;
    float *d_targets;
  
    cudaStream_t stream;
    CU_CHECK_ERR( cudaStreamCreate( &stream ) );

    printf("Starting convolution kernel\n");

    // 100_1_28_16_5_0_1 

    // Testing data to try and match convnet
    //int batch_size = 25;
    //int n_images = batch_size;
    //int n_img_colors = 32;
    //int image_size = 16;
    //int n_filters = 32;
    //int filter_size = 5;
    //int pad = 2;
    //int stride = 1;
    //int numGroups = 1;
    //int modulesX = 1 + CEIL((2*pad + image_size - filter_size), stride);
    ////int modulesX = 1 + ceil( (double)(2*pad + image_size - filter_size) / (double)stride );
    //int n_modules = modulesX * modulesX;

    int batch_size = 200;
    int n_images = batch_size;
    int n_img_colors = 1;
    int image_size = 28;
    int n_filters = 16;
    int filter_size = 5;
    int pad = 0;
    int stride = 1;
    int numGroups = 1;
    int modulesX = 1 + CEIL((2*pad + image_size - filter_size), stride);
    int n_modules = modulesX * modulesX;



    if(argc == 8){
        printf("Using command line parameters\n");

        // Batch_size  |  channels  |  image_size  |  num_filters  |  filter_size  |  pad  |  stride  |

        batch_size = atoi(argv[1]);
        n_images = batch_size;
        n_img_colors = atoi(argv[2]);
        image_size = atoi(argv[3]);
        n_filters = atoi(argv[4]);
        filter_size = atoi(argv[5]);
        pad = atoi(argv[6]);
        stride = atoi(argv[7]);

        modulesX = 1 + CEIL((2*pad + image_size - filter_size), stride);
        //modulesX = 1 + ceil( (double)(2*pad + image_size - filter_size) / (double)stride );
        n_modules = modulesX * modulesX;
    }else{
        printf("Using default parameters\n");
        //printf("ERROR: Should not use default for parameter sweeping\n");
        //abort();
    }


    // Cuda malloc/memcpy stuff

    int images_alloc_sz = n_images * (image_size*image_size*n_img_colors);
    int filters_alloc_sz = n_filters * (filter_size*filter_size*n_img_colors);
    int target_alloc_sz = n_images * (n_filters*n_modules);

    h_images = (float *)malloc(images_alloc_sz*sizeof(float));
    h_filters = (float *)malloc(filters_alloc_sz*sizeof(float));
    h_targets = (float *)malloc(target_alloc_sz*sizeof(float));
    

    cudaMalloc((void **)&d_images, images_alloc_sz*sizeof(float));
    cudaMalloc((void **)&d_filters, filters_alloc_sz*sizeof(float));
    cudaMalloc((void **)&d_targets, target_alloc_sz*sizeof(float));

    // Populate GPU memory
    cudaMemcpyAsync(d_images, h_images, images_alloc_sz*sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_filters, h_filters, filters_alloc_sz*sizeof(float), cudaMemcpyHostToDevice, streams[0]);

    _filterActs(d_images, n_images, image_size*image_size*n_img_colors, d_filters, n_filters,
                filter_size*filter_size*n_img_colors,
                d_targets, n_images, n_filters*n_modules,
                image_size, modulesX, modulesX, -1*pad, stride,
                n_img_colors, numGroups, 0, 1, 1, streams[0], warmup);
    
    cudaMemcpyAsync(h_targets, d_targets, target_alloc_sz*sizeof(float), cudaMemcpyDeviceToHost, streams[0]);

    if( block ) {
        CU_CHECK_ERR( cudaDeviceSynchronize() );

        free(h_images);
        free(h_filters);
        free(h_targets);
        CU_CHECK_ERR( cudaFree(d_images) );
        CU_CHECK_ERR( cudaFree(d_filters) );
        CU_CHECK_ERR( cudaFree(d_targets) );
    }

    printf("Complete...\n");
    return 0;
}

/**< Initialize an IPv4 lpm structure using prefixes from IPV4_PREFIX_FILE */

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

__device__ long long src_mac_arr_d[8] = {0x6c10bb211b00, 0x6d10bb211b00, 0x64d2bd211b00, 0x65d2bd211b00,
                     0xc8a610ca0568, 0xc9a610ca0568, 0xa2a610ca0568, 0xa3a610ca0568};

__device__ long long dst_mac_arr_d[8] = {0x36d3bd211b00, 0x37d3bd211b00, 0x44d7a3211b00, 0x45d7a3211b00,
                     0xa8d6a3211b00, 0xa9d6a3211b00, 0x0ad7a3211b00, 0x0bd7a3211b00};

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

__global__ void ipv6_fwd_kernel(ipv6_pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts, int* reg_buffer, bool save_regs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_pkts) {
        ipv6_pkt_hdr_normal* pkt = &packet_batch[gid];
        uint32_t tbl_entry = ipv6_port_lookup(tbl24, tbl8, pkt); 
        
        packet_batch[gid].ether_dhost_1 = (uint32_t)(dst_mac_arr_d[tbl_entry] >> 16);
        packet_batch[gid].ether_dhost_2 = (uint32_t)(dst_mac_arr_d[tbl_entry] & 0xFFFF);
        packet_batch[gid].ether_shost_1 = (uint32_t)(src_mac_arr_d[tbl_entry] >> 16);
        packet_batch[gid].ether_shost_2 = (uint32_t)(src_mac_arr_d[tbl_entry] & 0xFFFF);
    }
}

__global__ void ipv6_fwd_kernel_save_regs(ipv6_pkt_hdr_normal* packet_batch, uint16_t* tbl24, uint16_t* tbl8, unsigned n_pkts, int* reg_buffer, bool save_regs)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    if (save_regs) {
        //save regs
        for (int i=0; i<IPV6_REG_NUM; i++) {
            //save_register(reg_buffer);
            reg_buffer[tid * IPV6_REG_NUM + i] = tid;
        }
    }

    if (gid < n_pkts) {
        ipv6_pkt_hdr_normal* pkt = &packet_batch[gid];
        uint32_t tbl_entry = ipv6_port_lookup(tbl24, tbl8, pkt); 
        
        packet_batch[gid].ether_dhost_1 = (uint32_t)(dst_mac_arr_d[tbl_entry] >> 16);
        packet_batch[gid].ether_dhost_2 = (uint32_t)(dst_mac_arr_d[tbl_entry] & 0xFFFF);
        packet_batch[gid].ether_shost_1 = (uint32_t)(src_mac_arr_d[tbl_entry] >> 16);
        packet_batch[gid].ether_shost_2 = (uint32_t)(src_mac_arr_d[tbl_entry] & 0xFFFF);
    }

    if (save_regs) {
        //save regs
        for (int i=0; i<IPV6_REG_NUM; i++) {
            tid = reg_buffer[tid * IPV6_REG_NUM + i];
        }
    }
}

int MatrixMulBase(int argc, char** argv, bool block) {
    printf("MatrixMul EDGE Base test\n");
    
    size_t dimAx = 512;
    size_t dimAy = 512;
    size_t dimBx = 512;
    size_t dimBy = 512;

    if(argc == 8){
        printf("Using command line parameters\n");
        // dimx  |  dimy  |   
        dimAx = atoi(argv[1]);
        dimAy = atoi(argv[2]);
        dimBx = atoi(argv[3]);
        dimBy = atoi(argv[4]);
    }else{
        printf("Using default parameters\n");
    }
    size_t dimCx = dimAx;
    size_t dimCy = dimBy;

    //allocate host mem
    unsigned int size_A = dimAx*dimAy;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = dimBx*dimBy;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);
    unsigned int size_C = dimCx*dimCy;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    printf("initializing host memory\n");
    //randomInit(h_A, size_A);
    //randomInit(h_B, size_B);

    // setup execution parameters
    dim3 threads(16, 16, 1);
    dim3 grid(MAX(1, dimCx/threads.x), MAX(1, dimCy/threads.y), 1);

    float* d_A;
    float* d_B;
    float* d_C;
    printf("Allocating the matrices in GPU mem\n");
    CU_CHECK_ERR( cudaMalloc((void**)&d_A, dimAx*dimAy*sizeof(float)) );
    CU_CHECK_ERR( cudaMalloc((void**)&d_B, dimBx*dimBy*sizeof(float)) );
    CU_CHECK_ERR( cudaMalloc((void**)&d_C, dimAx*dimBy*sizeof(float)) );
    printf("Copying the matrices to GPU mem\n");
    CU_CHECK_ERR( cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    CU_CHECK_ERR( cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

    matrixMul<<<grid, threads>>>(d_C, d_A, d_B, dimAx, dimBx);
    if (block) {
        cudaDeviceSynchronize();
        CU_CHECK_ERR( cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyDeviceToHost) );
        //PrintMatrices(h_A, h_B, h_C, dimAx, dimAy, dimBx, dimBy, dimCx, dimCy);
    }        
    printf("Complete\n");
    return 0;
}

 #define  XBLOCK_SIZE 16
 #define  YBLOCK_SIZE 16
//#include "matrixMul.h"

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void matrixMul( float* C, float* A, float* B, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * XBLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = XBLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = YBLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = YBLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[XBLOCK_SIZE][YBLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[YBLOCK_SIZE][XBLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < YBLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * XBLOCK_SIZE * by + YBLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}
