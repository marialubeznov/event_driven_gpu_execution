// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>

#include "memc.h"
#include "memc_shared.h"
#include <edge_cuda.h>

#include <netinet/in_systm.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
//#include <net/ethernet.h>     /* the L2 protocols */
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/udp.h>


#include <pthread.h>
#include "memc_kernel.h"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

////// PREFERS SHARED in config (cudaFuncCachePreferShared)


#define CEIL(x, y) ( (x)/(y) + ( (x)%(y) ? 1 : 0 ) )


#define CU_CHECK_ERR(err)                                                       \
    if ( err != cudaSuccess ) {                                                 \
        printf("CUDA Error: %s\n", cudaGetErrorString(cudaGetLastError()));     \
        abort();                                                                \
    }

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
void m5_reset_stats(uint64_t ns_delay, uint64_t ns_period);
void m5_dump_stats(uint64_t ns_delay, uint64_t ns_period);
void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
}
#endif

volatile int x=0;
void 
delay(int val)
{
    while( val > 0 ) {
        val--;
        x++;
    }
}

#define DEBUG_SIZE 1024*32

enum RunConfig {
    BASE = 0,
    PERSISTENT_THREAD,
    EVENT, 
    TIMER_ONLY,
    CONV_ONLY, 
    MEMC_TIMER_CONV,
    MEMC_BURST_CONV,
    CONV_NULL_EVENT,
    CONV_NULL_WARMUP_EVENT,
    MEMC_TIMER_CONV2,
    //PTHREAD,
    MEMC_CONV_BASE,
    CPU_SLEEP_TEST
};

enum ScheduleType {
    SINGLE=0,
    TIMER, 
    BURST
};

enum BgTaskType {
    CONV=0,
    MATRIX_MUL,
	BACKPROP,
	BFS
};

int bg_task_type = MATRIX_MUL;



//////////////////////////////////////////////////////////////////////////////
/////////////////////////// Memcached stuff //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
gpuPrimaryHashtable* hPrimaryHashtable = NULL;
int* hLocks = NULL; 
MemcValue* hValueHeap = NULL;
int* hDebugPtr = NULL; 
int* tDebugPtr = NULL;

__device__ gpuPrimaryHashtable* gPrimaryHashtable;
__device__ int* gLocks;
__device__ MemcValue* gValueHeap;
__device__ int* gDebugPtr;


__global__ void emptyKernel();

__global__ void memcGetKernel(char* rxBuffers, char* txBuffers, unsigned nPkts);
__global__ void simpleMemcGetKernel(char* packets, unsigned nPkts, int* reg_buffer);
__global__ void simpleMemcGetKernel_save_regs(char* packets, unsigned nPkts, int* reg_buffer);
__global__ void initHashTable(SetRequest* inputs, unsigned N, unsigned timestamp);
__global__ void initDataStructures();

// Dummy kernels
__global__ void gMemcGetKernel(unsigned char* rxBuffers, unsigned char* txBuffers, unsigned nPkts);
__global__ void dummyKernel(float* A, float* B, float* C, unsigned stride, unsigned N);

void printPktHdr(void *hdr, bool staticPayloadLen = false);


int memcBase();
int memcEDGE();
int memcTimerEDGE();
int memcTimerEdgeConv(int argc, char** argv, ScheduleType scheduleType);
int convMemcEventNoRun(int argc, char** argv);
int warmupConvMemcEventNoRun(int argc, char** argv);
int memcTimerEdgeConv2(int argc, char** argv);
int memcConvBase();
void initMemc();
void populateHashTable();


int testMemcEventBaseline();

//////////////// Globals ////////////////
int gNumSets = 512;
int gNumGets = 512;
char gGetFilename[256];
char gSetFilename[256];

unsigned gTimerEventPeriod = 1000;
unsigned gTimerEventPeriodBatch = 200;
bool swizzle = false;
int n_batches = 1;
int n_requests_per_batch = 32;
bool save_regs = true;
int schedule_batch_size = 64;

unsigned gMemcCpuSleepCycles = 90000;
/////////////////////////////////////////

int nStreams = 32;
cudaStream_t* streams = NULL; 

struct MemcParam
{
    unsigned char* rx;
    unsigned n;
    int* reg_buffer;
};

void configureParamMem(MemcParam* paramMem, size_t bufferSize, size_t batchSize, size_t maxBatches)
{
	MemcParam* curParam = paramMem;
	size_t totalBufferSize = bufferSize * batchSize;

	for( unsigned batch=0; batch<maxBatches; ++batch ) {
        CU_CHECK_ERR( cudaMalloc((void**)&curParam->rx, totalBufferSize) );
        int reg_buffer_size = 32*MEMC_REG_NUM*512;
        CU_CHECK_ERR(cudaMalloc(&curParam->reg_buffer, reg_buffer_size));
        curParam->n = batchSize;
        curParam++;
	}
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////



/////////////////////
// Convolution Begin
/////////////////////
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

void _filterActs(float *images, int images_cols, int images_rows, float *filters, int filters_cols,
                int filters_rows,  float *targets, int targets_cols, int targets_rows,
                int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                int numImgColors, int numGroups, float scaleTargets, float scaleOutput, int conv, cudaStream_t stream, bool warmup);

int run_conv_kernel(int argc, char **argv, bool block, bool warmup);
int MatrixMulBase(int argc, char** argv, bool block);
extern "C" __global__ void matrixMul( float* C, float* A, float* B, int wA, int wB);
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
    RunConfig testToRun = EVENT;
    int opt;
    opterr = 0;

    int count = 0;

    const char* g = "get_file_16B.txt";
    const char* s = "set_file_16B.txt";

    strcpy(gGetFilename, g);
    strcpy(gSetFilename, s);


    while( (opt = getopt(argc, argv, "t:s:g:q:z:p:c:p:i:n:m:") ) != -1 ) {
        switch(opt) {
            case 't': 
                count += 2;
                testToRun = (RunConfig)atoi(optarg);
                break;
            case 'i':
                count += 2;
                gTimerEventPeriodBatch = atoi(optarg);
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
            case 's':
                count += 2;
                gNumSets = atoi(optarg);
                break;
            case 'm':
                count += 2;
                gNumGets = atoi(optarg);
                break;
            case 'q':
                count += 2;
                strcpy(gGetFilename, optarg);
                break;
            case 'z': 
                count += 2;
                strcpy(gSetFilename, optarg);
                break;

            case 'p':
                count += 2;
                gTimerEventPeriod = atoi(optarg);
                break;

            case 'c':
                count += 2;
                gMemcCpuSleepCycles = atoi(optarg);
                break;

            default: 
                std::cout << "Error: Unknown parameter: " << opt << std::endl;
                abort();
        }
    }

    printf("GET filename = %s\n", gGetFilename);
    printf("SET filename = %s\n", gSetFilename);

    streams = new cudaStream_t[nStreams];
    for( unsigned i=0; i<nStreams; ++i ) {
        CU_CHECK_ERR( cudaStreamCreate(&streams[i]) ); 
    }

    char** modArgv = &argv[count];
    int modArgc = argc-count;
    
    switch( testToRun ) {
        case BASE:
            ret = memcBase();
            break;

        case PERSISTENT_THREAD:
            abort();
            break;

        case EVENT:
            ret = memcEDGE();
            break;

        case TIMER_ONLY:
            ret = memcTimerEDGE();
            break;

        case CONV_ONLY:
            ret = run_conv_kernel(modArgc, modArgv, true, false);
            break;

        case MEMC_TIMER_CONV:
            ret = memcTimerEdgeConv(modArgc, modArgv, TIMER);
            break;

        case MEMC_BURST_CONV:
            ret = memcTimerEdgeConv(modArgc, modArgv, BURST);
            break;

        case CONV_NULL_EVENT:
            ret = convMemcEventNoRun(modArgc, modArgv);
            break;

        case CONV_NULL_WARMUP_EVENT:
            ret = warmupConvMemcEventNoRun(modArgc, modArgv);
            break;

        case MEMC_TIMER_CONV2:
            ret = memcTimerEdgeConv2(modArgc, modArgv);
            break;

        //case PTHREAD:
        //    ret = pthread_test();
        //    break;

        case MEMC_CONV_BASE:
            printf("Launching Memcached, sleeping %d cycles between packet batches\n", gMemcCpuSleepCycles);
            ret = memcConvBase();
            break;

        case CPU_SLEEP_TEST:
            ret = 0;
            //cudaSleepThread(1000000);
            //ret = testMemcEventBaseline();

            m5_dumpreset_stats(0, 0);
            //m5_work_begin(10, 0);
            cudaSleepThread(10000);
            m5_dumpreset_stats(0, 0);
            //m5_work_end(10, 0);


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

int memcTimerEdgeConv(int argc, char** argv, ScheduleType scheduleType)
{
    printf("Memcached Timer EDGE + Convolution kernel Test: n_requests_per_batch: %d maxBatches: %d\n", n_requests_per_batch, n_batches);
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = n_requests_per_batch;
    size_t maxBatches = n_batches;
    size_t MaxEventsNum = maxBatches;
    size_t sharedMem = 0;

    // setup execution parameters
    dim3 block(min(n_requests_per_batch, MAX_THREADS_PER_BLOCK), 1, 1); 
    dim3 grid(div_up(n_requests_per_batch, MAX_THREADS_PER_BLOCK), 1, 1); 

    // Register the event kernel
    //int eventId = cudaRegisterEvent((void*)memcGetKernel, grid, block, sharedMem);
    int eventId = cudaRegisterEvent((void*)simpleMemcGetKernel, (void*)simpleMemcGetKernel_save_regs, grid, block, sharedMem);

    // Setup the arguments.
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 0) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 8) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(int*), 16) ); //reg_buffer

    // Configure the parameter memory.
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, maxBatches, false);

	configureParamMem(paramMem, bufferSize, batchSize, maxBatches);

    int op = 1;
    int numGetReqs = gNumGets;
    GpuGetPkt* getPktTrace = NULL;
    
    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input 
    // within the simulator...
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );


    // Now copy all packets to GPU buffers 
    unsigned numBatches = numGetReqs / batchSize;
    MemcParam* curParam = paramMem;

    char* hrx = (char*)getPktTrace;
    for( unsigned i=0; i<numBatches; ++i ) {
        char* grx = (char*)curParam->rx;
        for( unsigned j=0; j<batchSize; ++j ) {
            CU_CHECK_ERR( cudaMemcpy(grx, hrx, RX_BUFFER_SIZE, cudaMemcpyHostToDevice) );
            grx += RX_BUFFER_SIZE;
            hrx += RX_BUFFER_SIZE;
        }
        curParam++;
    }

    //HACK for non save_regs kernel
    paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, MaxEventsNum, true);

 
    //if (RunBackgroundTask) {
        
    //}

    printf("Scheduling EDGE event as %s \n", scheduleType==TIMER? "TIMER" : scheduleType==SINGLE? "SINGLE" : "BURST");
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
        
    CU_CHECK_ERR( cudaDeviceSynchronize() );
   
    cudaMemcpy(hDebugPtr, tDebugPtr, 256*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Hit rate = %d/%d = %.3lf  \n", hDebugPtr[1], numGetReqs, 100.0*(double)hDebugPtr[1]/(double)numGetReqs);

    std::cout << "Success!" << std::endl;

    return 0;

}

void baseCudaLaunchGets()
{
    /////////////////// GET //////////////////
    int op = 1;
    int numGetReqs = gNumGets;    
   
    /*
    int nStreams = 32;
    cudaStream_t* streams = new cudaStream_t[nStreams];
    for( unsigned i=0; i<nStreams; ++i ) {
        CU_CHECK_ERR( cudaStreamCreate(&streams[i]) ); 
    }
    */

    GpuGetPkt* getPktTrace = NULL;
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));

    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input 
    // within the simulator...
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );

#if 0
    for(unsigned i=0; i<1; ++i) {
        char* x = (char*)&getPktTrace[i];
        printPktHdr(x);
    }

    GpuGetPkt* test = NULL;
    //numGetReqs = initGetReqTrace(NUM_REQUESTS_PER_BATCH, &test, "./get_file_16B.txt", gNumGets);
    numGetReqs = initGetReqTrace(32, &test, "./get_file_16B.txt", 32);
    for( unsigned i=0; i<1; ++i ) {
        printPktHdr(test);
    }
 
    return;
#endif

    char* rx;
    char* tx;
    int numBatches = numGetReqs / NUM_REQUESTS_PER_BATCH;

    cudaMalloc(&rx, RX_BUFFER_SIZE*numGetReqs);
    cudaMalloc(&tx, TX_BUFFER_SIZE*numGetReqs);

    printf("rx = %p, getPktTrace = %p, numrequests = %d, size = %d\n", 
            rx, getPktTrace, numGetReqs, RX_BUFFER_SIZE*numGetReqs);            

    cudaMemcpy(rx, getPktTrace, RX_BUFFER_SIZE*numGetReqs, cudaMemcpyHostToDevice);
    
    dim3 getGrid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 getBlock(MAX_THREADS_PER_BLOCK, 1, 1);

    char* trx = rx;
    char* ttx = tx;
    int streamIdx = 0;
    for( unsigned i=0; i<numBatches; ++i ) {
        simpleMemcGetKernel<<<getGrid, getBlock, 0, streams[streamIdx]>>>(trx, numGetReqs, NULL);
        trx += (RX_BUFFER_SIZE*NUM_REQUESTS_PER_BATCH);
        ttx += (RX_BUFFER_SIZE*NUM_REQUESTS_PER_BATCH);
        streamIdx = streamIdx + 1 / nStreams;
    }

    cudaDeviceSynchronize();

    printf("hDebugPtr = %p, gDebugPtr = %p\n", hDebugPtr, gDebugPtr);
    //cudaMemcpy(hDebugPtr, tDebugPtr, DEBUG_SIZE, cudaMemcpyDeviceToHost);

    //printf("Hit rate = %d/%d = %.3lf \n", hDebugPtr[1], numGetReqs, 100.0*((double)hDebugPtr[1]/(double)numGetReqs) );

    /*
    char* htx = (char*)malloc(TX_BUFFER_SIZE*numGetReqs);
    cudaMemcpy(htx, tx, TX_BUFFER_SIZE*numGetReqs, cudaMemcpyDeviceToHost);

    printPktHdr((void*)htx);
    printPktHdr((void*)(htx+TX_BUFFER_SIZE));
    printPktHdr((void*)(htx+255*TX_BUFFER_SIZE));
    printPktHdr((void*)(htx+256*TX_BUFFER_SIZE));
    printPktHdr((void*)(htx+257*TX_BUFFER_SIZE));
    */
}


void initMemc()
{
    initGpuShmemPtr(gpuPrimaryHashtable, hPrimaryHashtable, gPrimaryHashtable, hashsize(HASH_POWER));
    initGpuGlobals(int, hLocks, gLocks, numsets());
    initGpuGlobals(MemcValue, hValueHeap, gValueHeap, hashsize(HASH_POWER));
        
    // DEBUG
    hDebugPtr = (int*)malloc(DEBUG_SIZE*sizeof(int));
    initGpuGlobals(int, tDebugPtr, gDebugPtr, DEBUG_SIZE);

    int nThreads = 1024;
    dim3 block(nThreads, 1, 1);
    dim3 grid(hashsize(HASH_POWER) / nThreads, 1, 1);

    initDataStructures<<<grid, block>>>();

    cudaDeviceSynchronize();
}


///////////////////////// Initialize GPU Hashtable ////////////////////////////
void populateHashTable()
{
    SetRequest* setRequests = NULL;
    size_t numSetReqs = gNumSets; 

    setRequests = (SetRequest*)malloc(numSetReqs*sizeof(SetRequest));
    
    int op = 0;
    // Ask the real CPU to populate SET request buffers for us. Way faster than reading input 
    // within the simulator...
    CU_CHECK_ERR( edgeExtra(op, (void*)setRequests, numSetReqs, 0, gSetFilename) );

    printf("Setting up GPU set request buffer\n");
    SetRequest* gpuSetRequests;
    cudaMalloc(&gpuSetRequests, numSetReqs*sizeof(SetRequest));
    printf("copying\n");
    cudaMemcpy(gpuSetRequests, setRequests, numSetReqs*sizeof(SetRequest), cudaMemcpyHostToDevice);
    printf("complete\n");

    // Each warp processes a single SET request. 8 warps per block. 
    dim3 setGrid(numSetReqs / 8, 1, 1);
    dim3 setBlock(256, 1, 1);

    initHashTable<<<setGrid, setBlock>>>(gpuSetRequests, numSetReqs, 0);
   
    CU_CHECK_ERR( cudaDeviceSynchronize() );
 
    cudaMemcpy(hDebugPtr, tDebugPtr, 4*sizeof(int), cudaMemcpyDeviceToHost);
    printf("# of evictions = %d\n", hDebugPtr[0]);

    CU_CHECK_ERR( cudaFree(gpuSetRequests) );
    free( setRequests );
}




//////////////////////////////////////////////////////////rarely used//////////////////////////////////////////////////////////////////
int memcBase()
{
    std::cout << "Memc Base Test" << std::endl;

    // Initialize the Memcached data structures
    initMemc();
     
    // Populate the GPU hashtable with data
    populateHashTable();

    // Run some GET kernels
    baseCudaLaunchGets();

    return 0;
}


int memcTimerEDGE()
{
    std::cout << "Memcached Timer EDGE Test" << std::endl;
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = NUM_REQUESTS_PER_BATCH;
    size_t maxBatches = 32;
    size_t sharedMem = 0;

    dim3 grid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 block(MAX_THREADS_PER_BLOCK, 2, 1);

    // Register the event kernel
    //int eventId = cudaRegisterEvent((void*)memcGetKernel, grid, block, sharedMem);
    int eventId = cudaRegisterEvent((void*)memcGetKernel, (void*)memcGetKernel, grid, block, sharedMem);

    // AUTOMATICALLY GENERATED 
    // Should be:
    //      mod_cudaConfigureEventParam<<<unsigned char*, unsigned char*, unsigned>>>(eventId, maxBatches);
    // And have compiler generate the following

    // Setup the arguments.
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 0) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 8) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 16) );

    // Configure the parameter memory.
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, maxBatches, false);

    configureParamMem(paramMem, bufferSize, batchSize, maxBatches);

    int op = 1;
    int numGetReqs = gNumGets;
    GpuGetPkt* getPktTrace = NULL;
    
    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input 
    // within the simulator...
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );


    // Now copy all packets to GPU buffers 
    unsigned numBatches = numGetReqs / batchSize;
    MemcParam* curParam = paramMem;

    char* hrx = (char*)getPktTrace;
    for( unsigned i=0; i<numBatches; ++i ) {
        char* grx = (char*)curParam->rx;
        for( unsigned j=0; j<batchSize; ++j ) {
            CU_CHECK_ERR( cudaMemcpy(grx, hrx, RX_BUFFER_SIZE, cudaMemcpyHostToDevice) );
            grx += RX_BUFFER_SIZE;
            hrx += RX_BUFFER_SIZE;
        }
        curParam++;
    }

    // Schedule the event kernel to run on a timer
    CU_CHECK_ERR( cudaScheduleTimerEvent(eventId, gTimerEventPeriod) );

    CU_CHECK_ERR( cudaDeviceSynchronize() );
    
    cudaMemcpy(hDebugPtr, tDebugPtr, 256*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Hit rate = %d/%d = %.3lf \n", hDebugPtr[1], numGetReqs, 100.0*(double)hDebugPtr[1]/(double)numGetReqs);

    std::cout << "Success!" << std::endl;

    return 0;
}

int memcEDGE()
{
    printf("Memcached EDGE Test: n_requests_per_batch: %d maxBatches: %d\n", n_requests_per_batch, n_batches);
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = n_requests_per_batch;
    size_t maxBatches = n_batches;
    size_t MaxEventsNum = maxBatches;
    size_t sharedMem = 0;

    dim3 block(min(n_requests_per_batch, MAX_THREADS_PER_BLOCK), 1, 1); 
    dim3 grid(div_up(n_requests_per_batch, MAX_THREADS_PER_BLOCK), 1, 1); 

    // Register the event kernel
    //int eventId = cudaRegisterEvent((void*)memcGetKernel, grid, block, sharedMem);
    int eventId = cudaRegisterEvent((void*)simpleMemcGetKernel, (void*)simpleMemcGetKernel, grid, block, sharedMem);

    // AUTOMATICALLY GENERATED 
    // Should be:
    //      mod_cudaConfigureEventParam<<<unsigned char*, unsigned char*, unsigned>>>(eventId, maxBatches);
    // And have compiler generate the following

    // Setup the arguments.
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 0) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 8) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 16) );

    // Configure the parameter memory.
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, MaxEventsNum, false);

    configureParamMem(paramMem, bufferSize, batchSize, maxBatches);

    int op = 1;
    int numGetReqs = gNumGets;
    GpuGetPkt* getPktTrace = NULL;
    
    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input 
    // within the simulator...
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );


    // Now copy all packets to GPU buffers 
    unsigned numBatches = numGetReqs / batchSize;
    MemcParam* curParam = paramMem;

    char* hrx = (char*)getPktTrace;
    for( unsigned i=0; i<numBatches; ++i ) {
        char* grx = (char*)curParam->rx;
        for( unsigned j=0; j<batchSize; ++j ) {
            CU_CHECK_ERR( cudaMemcpy(grx, hrx, RX_BUFFER_SIZE, cudaMemcpyHostToDevice) );
            grx += RX_BUFFER_SIZE;
            hrx += RX_BUFFER_SIZE;
        }
        curParam++;
    }

    printf("finished mem copy operations, schediling an event \n");
    
    for( unsigned i=0; i<MaxEventsNum; ++i ) {
        CU_CHECK_ERR( cudaScheduleEvent(eventId) );
    }

    CU_CHECK_ERR( cudaDeviceSynchronize() );
    

    cudaMemcpy(hDebugPtr, tDebugPtr, 256*sizeof(int), cudaMemcpyDeviceToHost);
//    for( unsigned i=0; i<4; ++i ) {
//        printf("%d => %x\n", i, hDebugPtr[i]);
//    }

    printf("Hit rate = %d/%d = %.3lf \n", hDebugPtr[1], numGetReqs, 100.0*(double)hDebugPtr[1]/(double)numGetReqs);

/*
    unsigned totalBufferSize = bufferSize*batchSize;
    char* tmp = (char*)malloc(totalBufferSize);

    CU_CHECK_ERR( cudaMemcpy(tmp, paramMem->tx, totalBufferSize, cudaMemcpyDeviceToHost) ); 
    printPktHdr(tmp);
    printPktHdr(tmp+4*RX_BUFFER_SIZE);

    CU_CHECK_ERR( cudaMemcpy(tmp, (paramMem+1)->tx, totalBufferSize, cudaMemcpyDeviceToHost) ); 
    printPktHdr(tmp);
    printPktHdr(tmp + 256*RX_BUFFER_SIZE);
*/

    std::cout << "Success!" << std::endl;

    return 0;

}

int memcTimerEdgeConv2(int argc, char** argv)
{
    std::cout << "Memcached Timer EDGE + Convolution kernel Test 2" << std::endl;
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = NUM_REQUESTS_PER_BATCH;
    size_t maxBatches = 16;
    size_t sharedMem = 0;

    dim3 grid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 block(MAX_THREADS_PER_BLOCK, 2, 1);

    // Register the event kernel
    //int eventId = cudaRegisterEvent((void*)memcGetKernel, grid, block, sharedMem);
    int eventId = cudaRegisterEvent((void*)memcGetKernel, (void*)memcGetKernel, grid, block, sharedMem);

    // Setup the arguments.
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 0) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 8) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 16) );

    // Configure the parameter memory.
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, maxBatches, false);

    configureParamMem(paramMem, bufferSize, batchSize, maxBatches);

    int op = 1;
    int numGetReqs = gNumGets;
    GpuGetPkt* getPktTrace = NULL;
    
    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input 
    // within the simulator...
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );


    // Now copy all packets to GPU buffers 
    unsigned numBatches = numGetReqs / batchSize;
    MemcParam* curParam = paramMem;

    char* hrx = (char*)getPktTrace;
    for( unsigned i=0; i<numBatches; ++i ) {
        char* grx = (char*)curParam->rx;
        for( unsigned j=0; j<batchSize; ++j ) {
            CU_CHECK_ERR( cudaMemcpy(grx, hrx, RX_BUFFER_SIZE, cudaMemcpyHostToDevice) );
            grx += RX_BUFFER_SIZE;
            hrx += RX_BUFFER_SIZE;
        }
        curParam++;
    }

 
    // First, setup and launch the convolution kernel
    run_conv_kernel(argc, argv, false, false);

    // Then, schedule the event kernel to run on a timer
    CU_CHECK_ERR( cudaScheduleTimerEvent(eventId, gTimerEventPeriod) );

    // Finally, synchronize the device
    CU_CHECK_ERR( cudaDeviceSynchronize() );
    
    cudaMemcpy(hDebugPtr, tDebugPtr, 256*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Hit rate = %d/%d = %.3lf  \n", hDebugPtr[1], numGetReqs, 100.0*(double)hDebugPtr[1]/(double)numGetReqs);

    std::cout << "Success!" << std::endl;

    return 0;

}

int warmupConvMemcEventNoRun(int argc, char** argv)
{
    std::cout << "Convolution with Memcached event resource allocation only test" << std::endl;
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    //populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = NUM_REQUESTS_PER_BATCH;
    size_t maxBatches = 32;
    size_t sharedMem = 0;

    dim3 grid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 block(MAX_THREADS_PER_BLOCK, 2, 1);

    // Register the event kernel
    int eventId = cudaRegisterEvent((void*)memcGetKernel, (void*)memcGetKernel, grid, block, sharedMem);

    // Setup the arguments.
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 0) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 8) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 16) );

    // Configure the parameter memory.
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, maxBatches, false);

    // First, launch a warmup kernel. Then launch the real one. 
    run_conv_kernel(argc, argv, true, true);  // warmup
    
    run_conv_kernel(argc, argv, false, false); // real

    // Finally, synchronize the device
    CU_CHECK_ERR( cudaDeviceSynchronize() );
    
    return 0;
}

int convMemcEventNoRun(int argc, char** argv)
{
    std::cout << "Convolution with Memcached event resource allocation only test" << std::endl;
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    //populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = NUM_REQUESTS_PER_BATCH;
    size_t maxBatches = 32;
    size_t sharedMem = 0;

    dim3 grid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 block(MAX_THREADS_PER_BLOCK, 2, 1);

    // Register the event kernel
    int eventId = cudaRegisterEvent((void*)memcGetKernel,(void*)memcGetKernel, grid, block, sharedMem);

    // Setup the arguments.
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 0) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned char*), 8) );
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(unsigned), 16) );

    // Configure the parameter memory.
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, maxBatches, false);

 
    // First, setup and launch the convolution kernel
    run_conv_kernel(argc, argv, false, false);

    // Finally, synchronize the device
    CU_CHECK_ERR( cudaDeviceSynchronize() );
    
    return 0;

}

int testMemcEventBaseline()
{
    std::cout << "Memcached Event Baseline" << std::endl;
   
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = NUM_REQUESTS_PER_BATCH;
    size_t maxBatches = 32;
    size_t sharedMem = 0;

    dim3 getGrid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 getBlock(MAX_THREADS_PER_BLOCK, 2, 1);

    // Register the event kernel
    int eventId = cudaRegisterEvent((void*)memcGetKernel, (void*)memcGetKernel, getGrid, getBlock, sharedMem);

    /////////////////// GET //////////////////
    int op = 1;
    int numGetReqs = gNumGets;    
   
    GpuGetPkt* getPktTrace = NULL;
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));

    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input within the simulator... 
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );

    char* rx;
    char* tx;
    int numBatches = numGetReqs / NUM_REQUESTS_PER_BATCH;

    cudaMalloc(&rx, RX_BUFFER_SIZE*numGetReqs);
    cudaMalloc(&tx, TX_BUFFER_SIZE*numGetReqs);

    printf("rx = %p, getPktTrace = %p, numrequests = %d, size = %d\n", 
			rx, getPktTrace, numGetReqs, RX_BUFFER_SIZE*numGetReqs);            

    cudaMemcpy(rx, getPktTrace, RX_BUFFER_SIZE*numGetReqs, cudaMemcpyHostToDevice);
    
    char* trx = rx;
    char* ttx = tx;
    int streamIdx = 0;
    for( unsigned i=0; i<numBatches; ++i ) {
        printf("Launching batch %d\n", i);    
        memcGetKernel<<<getGrid, getBlock, 0, streams[streamIdx]>>>(trx, ttx, numGetReqs);
        trx += (RX_BUFFER_SIZE*NUM_REQUESTS_PER_BATCH);
        ttx += (RX_BUFFER_SIZE*NUM_REQUESTS_PER_BATCH);
        streamIdx = streamIdx + 1 / nStreams;

        printf("Sleeping 90k cycles\n");
        cudaSleepThread(gMemcCpuSleepCycles);
        printf("Done\n");
    }

    CU_CHECK_ERR( cudaDeviceSynchronize() );
    
    cudaMemcpy(hDebugPtr, tDebugPtr, 256*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Hit rate = %d/%d = %.3lf \n", hDebugPtr[1], numGetReqs, 100.0*(double)hDebugPtr[1]/(double)numGetReqs);

    std::cout << "Success!" << std::endl;

    return 0;
}




bool isConvFinished()
{
    int op = 2;
    char* getPktTrace = NULL; 
    int nGets = 0;
    cudaError_t streamZeroComplete = edgeExtra(op, getPktTrace, nGets, NUM_REQUESTS_PER_BATCH, gGetFilename); 
    
    if( streamZeroComplete == cudaSuccess ) {
        // CONV kernel has completed, time to stop launching
        printf("Convolution kernel completed, stoping Memcached kernel launches\n");
        return true; 
    } else {
        assert( streamZeroComplete == cudaErrorNotReady );
        return false;
    }
}





int memcConvBase()
{
    std::cout << "Memcached + Conv all CUDA API" << std::endl;
    
    //streams = new cudaStream_t[nStreams];
    //for( unsigned i=0; i<nStreams; ++i ) {
    //    CU_CHECK_ERR( cudaStreamCreate(&streams[i]) ); 
    //}

    ////////////////////// MEMCACHED Setup  /////////////////////////////
    // Initialize the Memcached data structures
    initMemc();

    // Populate the GPU hashtable with data
    populateHashTable();

    size_t bufferSize = BUFFER_SIZE;
    size_t batchSize = NUM_REQUESTS_PER_BATCH;
    size_t maxBatches = 16;
    size_t sharedMem = 0;

    int op = 1;
    int numGetReqs = gNumGets;    
                                                                                                         
    GpuGetPkt* getPktTrace = NULL;
    getPktTrace = (GpuGetPkt*)malloc(numGetReqs*sizeof(GpuGetPkt));
   
    dim3 getGrid(NUM_REQUESTS_PER_BATCH/MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 getBlock(MAX_THREADS_PER_BLOCK, 2, 1);

    // Register the event kernel
    int eventId = cudaRegisterEvent((void*)memcGetKernel, (void*)memcGetKernel,getGrid, getBlock, sharedMem);

    // Ask the real CPU to populate GSET request buffers for us. Way faster than reading input 
    // within the simulator...
    CU_CHECK_ERR( edgeExtra(op, (void*)getPktTrace, numGetReqs, NUM_REQUESTS_PER_BATCH, gGetFilename) );
                                                                                                         
    char* rx;
    char* tx;
    int numBatches = numGetReqs / NUM_REQUESTS_PER_BATCH;
                                                                                                         
    cudaMalloc(&rx, RX_BUFFER_SIZE*numGetReqs);
    cudaMalloc(&tx, TX_BUFFER_SIZE*numGetReqs);
                                                                                                         
    printf("rx = %p, getPktTrace = %p, numrequests = %d, size = %d\n", 
            rx, getPktTrace, numGetReqs, RX_BUFFER_SIZE*numGetReqs);            
                                                                                                         
    cudaMemcpy(rx, getPktTrace, RX_BUFFER_SIZE*numGetReqs, cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////////////////
    ////////////////////// Convolution Setup  /////////////////////////////

    // First, setup and launch the convolution kernel
    float *h_images;
    float *h_filters;
    float *h_targets;

    float *d_images;
    float *d_filters;
    float *d_targets;
  
    printf("Starting convolution kernel\n");

    // 100_1_28_16_5_0_1 
    int batch_size = 100;
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
                n_img_colors, numGroups, 0, 1, 1, streams[0], false);

    cudaMemcpyAsync(h_targets, d_targets, target_alloc_sz*sizeof(float), cudaMemcpyDeviceToHost, streams[0]);

    printf("Completed launching Convolution, now launching Memcached!\n");

    char* trx = rx;
    char* ttx = tx;
    unsigned batchCount = 0;
    int streamIdx = 0;
    //for( unsigned i=0; i<numBatches; ++i ) {
    while( true ) {
        // Reset stats for this next batch
        m5_reset_stats(0, 0);

        if( streamIdx == 0 ) // Don't use stream zero
            streamIdx = 1;
        memcGetKernel<<<getGrid, getBlock, 0, streams[streamIdx]>>>(trx, ttx, numGetReqs);

        if( ++batchCount >= numBatches ) {
            trx = rx;
            ttx = tx;
            batchCount = 0;
        } else {
            trx += (RX_BUFFER_SIZE*NUM_REQUESTS_PER_BATCH);
            ttx += (RX_BUFFER_SIZE*NUM_REQUESTS_PER_BATCH);
        }

        streamIdx = (streamIdx + 1) % nStreams;
       
        // Dump out stats
        m5_dump_stats(0, 0);

        cudaSleepThread(gMemcCpuSleepCycles);

        if( isConvFinished() )
            break;
    }

    // Finally, synchronize the device
    CU_CHECK_ERR( cudaDeviceSynchronize() );
    
    //cudaMemcpy(hDebugPtr, tDebugPtr, 256*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Success!" << std::endl;
                                                 
    free(h_images);
    free(h_filters);
    free(h_targets);
    CU_CHECK_ERR( cudaFree(d_images) );
    CU_CHECK_ERR( cudaFree(d_filters) );
    CU_CHECK_ERR( cudaFree(d_targets) );
    
    printf("Complete...\n");
    
    return 0;
}


void* pmain(void* arg)
{
    int tid = *(int*)arg;
    printf("I'm thread (%d)!\n", tid);
    
    if( tid == 0 ) {
        // Run some GET kernels
        //baseCudaLaunchGets();       
        //memcBase();
    } else {
        run_conv_kernel(0, NULL, true, false);
    }
   
    return NULL;
}


// int pthread_test()
// {

//     // Initialize the Memcached data structures
//     initMemc();
     
//     // Populate the GPU hashtable with data
//     populateHashTable();

//     pthread_t p[2];
//     int arg[2] = {0, 1};
//     for( unsigned i=0; i<2; ++i ) {
//         if( pthread_create(&p[i], NULL, pmain, &arg[i]) ) {
//             printf("Error creating thread\n");
//             return 1;
//         }
//     }

//     for( unsigned i=0; i<2; ++i ) {
//         if( pthread_join(p[i], NULL) ) {
//             printf("Error joining thread\n");
//             return 1;
//         }
//     }

//     return 0;
// }


//////////////////////////////////////////////////////////////////
//////////////////////// Helpers /////////////////////////////////
//////////////////////////////////////////////////////////////////
// void printPktHdr(void *hdr, bool staticPayloadLen)
// {
//     struct ether_header *eh = (struct ether_header *)hdr;
//     struct iphdr *iph = (struct iphdr *)((size_t)hdr + sizeof(struct ether_header));
//     struct udphdr *uh = (struct udphdr *)((size_t)hdr + sizeof(struct ether_header) + sizeof(struct iphdr));
//     uint8_t *payload = (uint8_t *)((size_t)hdr + sizeof(struct ether_header) + sizeof(struct iphdr) + sizeof(struct udphdr));

//     unsigned payloadLen;
//     if( !staticPayloadLen )
//         payloadLen = ntohs(uh->len) - sizeof(struct udphdr);
//     else
//         payloadLen = 14;
    

//     printf("Packet header contents: \n");

//     /***** ETHERNET HEADER *****/
//     printf("\t==Ethernet header==\n");
//     printf("\t\tDest: ");
//     for(unsigned i=0; i<ETH_ALEN; ++i)
//         printf("%hhx ", eh->ether_dhost[i]);
//     printf("\n\t\tSource: ");
//     for(unsigned i=0; i<ETH_ALEN; ++i)
//         printf("%hhx ", eh->ether_shost[i]);
//     printf("\n\t\tType: %hx\n", eh->ether_type);
//     /***** END ETHERNET HEADER *****/

//     **** IP HEADER ****
//     printf("\t==IP header==\n");
//     printf("\t\tVersion+hdr_len: %hhu\n", iph->version);
//     printf("\t\tTOS: %hhu\n", iph->tos);
//     printf("\t\tTotal Length: %hu\n", ntohs(iph->tot_len));
//     printf("\t\tID: %hu\n", ntohs(iph->id));
//     printf("\t\tFrag_off: %hu\n", iph->frag_off);
//     printf("\t\tTTL: %hhu\n", iph->ttl);
//     printf("\t\tProtocol: %hhu\n", iph->protocol);
//     printf("\t\tchecksum: %hu\n", ntohs(iph->check));
//     printf("\t\tSource address: %x\n", ntohl(iph->saddr));
//     printf("\t\tDest address: %x\n", ntohl(iph->daddr));
//     /***** END IP HEADER *****/

//     /***** UDP HEADER *****/
//     printf("\t==UDP header==\n");
//     printf("\t\tSource port: %hu\n", ntohs(uh->source));
//     printf("\t\tDest port: %hu\n", ntohs(uh->dest));
//     printf("\t\tLength: %hu\n", ntohs(uh->len));
//     printf("\t\tChecksum: %hu\n", uh->check);
//     /***** END UDP HEADER *****/

//     printf("\nPayload: ");
//     for(unsigned i=0; i<payloadLen; ++i){
//         printf("%02x", (payload[i]));
//     }

//     printf("\n\n");

// }

/////////////////////////////////////////////////////////////////////////////
//////////////////////// Convolution Kernel Stuff ///////////////////////////
/////////////////////////////////////////////////////////////////////////////












/////////////////////////////////////BACKPROP//////////////////////////////////////////////
void run_backprop(int argc, char **argv, bool block) {
    printf("MARIA inside run_backprop\n");
    setup(argc, argv);
}

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

#define MAX(x, y) ( (x)>(y) ? (x) : (y) )

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

  

