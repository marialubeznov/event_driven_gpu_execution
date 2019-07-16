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
#include "headers.h"
#include "packet.h"

#define IPSEC_REG_NUM 20

#define CEIL(x, y) ( (x)/(y) + ( (x)%(y) ? 1 : 0 ) )
#define MAX(x, y) ( (x)>(y) ? (x) : (y) )

int nStreams = 32;
cudaStream_t* streams = NULL; 
unsigned gTimerEventPeriod = 1000;
unsigned gTimerEventPeriodBatch = 200;
bool swizzle = false;
int n_batches = 1;
int n_requests_per_batch = 32;
bool save_regs = true;

int schedule_batch_size = 32;

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



struct MemcParam {
	uint32* esk;
	uint32* dsk;
    uint8* input_o;
    uint8* output_o;
    size_t len;
    size_t thread;
    int* reg_buffer;
    bool save_regs;
};

enum RunConfig {
    BASE=0,
    EVENT_TEST, 
    TIMER_TEST,
    EVENT_TIMER_BG,
    EVENT_TIMER_BATCH_BG,
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
void configureParamMem(MemcParam* paramMem, size_t esk_mem_size,size_t mem_size,size_t len,size_t thread,size_t maxBatches);
void PrintMatrices(float* h_A, float* h_B, float* h_C, int dimAx, int dimAy, int dimBx, int dimBy, int dimCx, int dimCy);
int run_conv_kernel(int argc, char **argv, bool block, bool warmup);
int BaseTest(int argc, char** argv);
int EDGETest(int argc, char** argv, bool RunBackgroundTask, ScheduleType scheduleType);
void randomInit(float* data, int size);
int MatrixMulBase(int argc, char** argv, bool block);

//kernels
extern "C" __global__ void matrixMul( float* C, float* A, float* B, int wA, int wB);
//__global__ void matrixMul_save_regs( float* C, float* A, float* B, int wA, int wB, int* reg_buffer, bool save_regs);
unsigned int packet_size (unsigned int packet_number);
void des_encrypt( uint32 *esk, uint32 *dsk, uint8 *input, uint8 *output, int len);
__global__ void des_encrypt_dev( uint32 *esk, uint32 *dsk, uint8 *input_o, uint8 *output_o, int len, int thread, int* reg_buffer, bool save_regs);
__global__ void des_encrypt_dev_save_regs( uint32 *esk, uint32 *dsk, uint8 *input_o, uint8 *output_o, int len, int thread, int* reg_buffer, bool save_regs);
__device__ void des_crypt_dev( uint32 *SK, uint8 *input, uint8 *output, int len, int thread);
void des_crypt( uint32 *SK, uint8 *input, uint8 *output, int len);
int des_set_key( uint32 *esk, uint32 *dsk, uint8 key1[8],uint8 key2[8], uint8 key3[8]);
int des_main_ks( uint32 *SK, uint8 *key );
__device__ void DES_ROUND_dev(uint32 *SK, uint32 X, uint32 Y);
void DES_ROUND(uint32 *SK, uint32 X, uint32 Y);
double my_timer();

extern "C" int setup(int argc, char** argv);
void run_backprop(int argc, char **argv, bool block);

void _filterActs(float *images, int images_cols, int images_rows, float *filters, int filters_cols,
                int filters_rows,  float *targets, int targets_cols, int targets_rows,
                int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                int numImgColors, int numGroups, float scaleTargets, float scaleOutput, int conv, cudaStream_t stream, bool warmup);

int div_up(int n, int d) {
    return n / d + (((n < 0) ^ (d > 0)) && (n % d));
} 
int bfs_main(int argc, char** argv);

void run_bfs(int argc, char **argv, bool block) {
    printf("MARIA inside run_bfs\n");
    bfs_main(argc, argv);
}

int main(int argc, char** argv) {
    std::cout << "=== EDGE BEGIN ===" << std::endl;
    int ret = 0;
    RunConfig testToRun = EVENT_TEST;
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
        case BASE:
            ret = BaseTest(modArgc, modArgv);
            break;
        case EVENT_TEST:
            ret = EDGETest(modArgc, modArgv, false, SINGLE);
            break;
        case TIMER_TEST:
            ret = EDGETest(modArgc, modArgv, false, TIMER);
            break;
        case EVENT_TIMER_BG: 
            ret = EDGETest(modArgc, modArgv, true, TIMER);
        	break;
        case EVENT_TIMER_BATCH_BG: 
            ret = EDGETest(modArgc, modArgv, true, BATCH);
            break;
        case BG_TASK: 
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

int BaseTest(int argc, char** argv) {
    printf("DES EDGE Base test\n");
    int numpackets=n_batches;
    int i, j;
    unsigned char **packet_in, **packet_in_dev, 
    **packet_out, **packet_out_dev, **packet_open;
    int num_thread;
    int num_size;
    uint32 *des_esk;
    uint32 *des_dsk;

    uint32 *des_esk_dev;
    uint32 *des_dsk_dev;
    //allocate host mem
    packet_in = (unsigned char**)malloc(numpackets*sizeof(unsigned char*)); 
    packet_in_dev = (unsigned char**)malloc(numpackets*sizeof(unsigned char*));
    packet_out = (unsigned char**)malloc(numpackets*sizeof(unsigned char*)); 
    packet_out_dev = (unsigned char**)malloc(numpackets*sizeof(unsigned char*));
    packet_open = (unsigned char**)malloc(numpackets*sizeof(unsigned char*));
    des_esk = (uint32*)malloc(96*sizeof(uint32));
    //CU_CHECK_ERR(cudaHostAlloc(&des_esk, 96*sizeof(uint32), cudaHostAllocDefault));
    CU_CHECK_ERR(cudaMalloc(&des_esk_dev, 96*sizeof(uint32)));
    des_dsk = (uint32*)malloc(96*sizeof(uint32));
    //CU_CHECK_ERR(cudaHostAlloc(&des_dsk, 96*sizeof(uint32), cudaHostAllocDefault));
    CU_CHECK_ERR(cudaMalloc(&des_dsk_dev, 96*sizeof(uint32)));
    num_thread=1;
    num_size = num_thread*16;

    /*Generate encryption key*/
    des_set_key(des_esk, des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);

    // setup execution parameters
    //memory allocation for packet
    for(i = 0; i < numpackets; i++){
                packet_in[i] = (unsigned char*)malloc(num_size*num_size*sizeof(unsigned char)); 
                //CU_CHECK_ERR(cudaHostAlloc(&packet_in[i], num_size[i]*num_size[i]*sizeof(unsigned char), cudaHostAllocDefault));
                CU_CHECK_ERR(cudaMalloc(&packet_in_dev[i], num_size*num_size*sizeof(unsigned char)));
                packet_out[i] = (unsigned char*)malloc(num_size*num_size*sizeof(unsigned char));
                //CU_CHECK_ERR(cudaHostAlloc(&packet_out[i], num_size[i]*num_size[i]*sizeof(unsigned char), cudaHostAllocDefault));
                CU_CHECK_ERR(cudaMalloc(&packet_out_dev[i], num_size*num_size*sizeof(unsigned char)));
                packet_open[i] =  (unsigned char *) malloc (num_size*num_size* sizeof(unsigned char));

            
    }

    printf("DES CUDA baseline inputs are generating\n");
    //generate packet
    for(i = 0; i < numpackets; i++){
            printf("%d PACKET size : %d \n", i, num_size);
            
            for(j = 0; j < num_size*num_size; j++){
                    if(j < HEADER_SIZE ){
                            packet_in[i][j] = headers[i % MAX_PACKETS][j];
                    }else{
                            packet_in[i][j] = DES3_init[j%8];
                    }
                }
            
    }

    // copy data to GPU
    for(i = 0; i < numpackets; i++){
        
        
                CU_CHECK_ERR(cudaMemcpyAsync(packet_in_dev[i], packet_in[i], 
                num_size*num_size*sizeof(unsigned char), cudaMemcpyHostToDevice));//, des_stream[i]));
    }
    CU_CHECK_ERR(cudaMemcpyAsync(des_esk_dev, des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice));//, des_stream[0]));
    CU_CHECK_ERR(cudaMemcpyAsync(des_dsk_dev, des_dsk, 96*sizeof(uint32), cudaMemcpyHostToDevice));//, des_stream[0]));
    CU_CHECK_ERR(cudaDeviceSynchronize());

    printf("DES CUDA baseline program is running\n");
    
    // run des
    for(i = 0; i < numpackets; i++){
            printf("Launching kernel deval %d with %d threads \n",i, num_thread*32 );
            
                des_encrypt_dev<<<1, num_thread*32, 0>>>( des_esk_dev, des_esk_dev, packet_in_dev[i],
                                        packet_out_dev[i],num_size*num_size/8, num_thread*32,NULL, false);
            

    }
    CU_CHECK_ERR(cudaDeviceSynchronize());
    

    for(i = 0; i < numpackets; i++){
            
                CU_CHECK_ERR(cudaMemcpyAsync(packet_out[i], packet_out_dev[i], 
                num_size*num_size*sizeof(unsigned char), cudaMemcpyDeviceToHost));//, des_stream[i]));
        
    }
    CU_CHECK_ERR(cudaDeviceSynchronize());

    printf("CPU program running\n");
    // run des
    for(i = 0; i < numpackets; i++){
            printf("%d packet processed\n", i );
            
                des_encrypt(des_esk, des_dsk, packet_in[i], packet_open[i], num_size*num_size/8);

    }


    /*Verification*/
    printf("verifying\n");
    int flag = 0;
    for(i = 0; i < numpackets; i++){
                for(j = 0; j < num_size*num_size; j++){
                    if(packet_out[i][j] != packet_open[i][j]){
                        printf("Error:%u, %u, %d, %d\n", packet_out[i][j], packet_open[i][j], i, j);
                    flag = 1;
                        break;
                    }
                }
            
    }

    if(!flag) printf("verify successfully\n");
 
    for(i = 0; i < numpackets; i++){
            //CU_CHECK_ERR(cudaStreamDestroy(des_stream[i]));
            CU_CHECK_ERR(cudaFreeHost(packet_in[i]));
            CU_CHECK_ERR(cudaFree(packet_in_dev[i]));
            CU_CHECK_ERR(cudaFreeHost(packet_out[i]));
            CU_CHECK_ERR(cudaFree(packet_out_dev[i]));
            free(packet_open[i]);
    }

    CU_CHECK_ERR(cudaFreeHost(des_esk));
    CU_CHECK_ERR(cudaFree(des_esk_dev));
    CU_CHECK_ERR(cudaFreeHost(des_dsk));
    CU_CHECK_ERR(cudaFree(des_dsk_dev));


    free(packet_in);
    free(packet_in_dev);
    free(packet_out);
    free(packet_out_dev);
    free(packet_open);

    cudaDeviceSynchronize();
    return 0;
}

int EDGETest(int argc, char** argv, bool RunBackgroundTask, ScheduleType scheduleType) {
    printf("DES EDGE Test RunBackgroundTask: %d, Timer: %d\n", RunBackgroundTask, scheduleType);


    int i, j;
    unsigned char **packet_in, 
    **packet_out, **packet_open;
    int num_thread;
    int num_size;
    uint32 *des_esk;
    uint32 *des_dsk;



    //allocate host mem
    int numpackets=n_batches;
    int MaxEventsNum=n_batches;
 
    packet_in = (unsigned char**)malloc(numpackets*sizeof(unsigned char*)); 
    packet_out = (unsigned char**)malloc(numpackets*sizeof(unsigned char*)); 
    packet_open = (unsigned char**)malloc(numpackets*sizeof(unsigned char*));

    des_esk = (uint32*)malloc(96*sizeof(uint32));    
    des_dsk = (uint32*)malloc(96*sizeof(uint32));
    
    num_thread=1;
    num_size = num_thread*16;
    des_set_key(des_esk, des_dsk, DES3_keys[0], DES3_keys[1], DES3_keys[2]);

//memory allocation for packet
    for(i = 0; i < numpackets; i++){
            
                packet_in[i] = (unsigned char*)malloc(num_size*num_size*sizeof(unsigned char)); 
                packet_out[i] = (unsigned char*)malloc(num_size*num_size*sizeof(unsigned char));
                packet_open[i] =  (unsigned char *) malloc (num_size*num_size* sizeof(unsigned char));      
    }
    
    //generate packet
    for(i = 0; i < numpackets; i++){            
            for(j = 0; j < num_size*num_size; j++){
                    if(j < HEADER_SIZE ){
                            packet_in[i][j] = headers[i % MAX_PACKETS][j];
                    }else{
                            packet_in[i][j] = DES3_init[j%8];
                    }
                }
            
    }
    
    // setup execution parameters
    dim3 threads(num_thread*32, 1,1);
    dim3 grid(1, 1,1);

    // Register the event kernel
    int eventId = cudaRegisterEvent((void*)des_encrypt_dev, (void*)des_encrypt_dev_save_regs, grid, threads, 0);
    // Setup the arguments
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(uint32*), 0) ); //des_esk_dev
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(uint32*), 8) ); //des_esk_dev
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(uint8*), 16) ); //input
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(uint8*), 24) ); //output
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(int), 32) ); //len
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(int), 40) ); //thread
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(int*), 48) ); //reg_buffer
    CU_CHECK_ERR( cudaSetupEventArgument(sizeof(bool), 56) ); //save_regs

    // Configure the parameter memory
    unsigned paramSize = sizeof(MemcParam);
    MemcParam* paramMem = (MemcParam*)cudaConfigureEventParam(eventId, paramSize, MaxEventsNum, false);
    configureParamMem(paramMem, 96*sizeof(uint32), num_size*num_size* sizeof(unsigned char),num_size*num_size/8, num_thread*32, MaxEventsNum);

    //copy from host to gpu
    MemcParam* curParam = paramMem;
    for( unsigned batch=0; batch<MaxEventsNum; batch++ ) {
        CU_CHECK_ERR( cudaMemcpy(curParam->esk, des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice) );
        CU_CHECK_ERR( cudaMemcpy(curParam->dsk, des_esk, 96*sizeof(uint32), cudaMemcpyHostToDevice) ); 
        CU_CHECK_ERR( cudaMemcpy(curParam->input_o, packet_in[batch], num_size*num_size*sizeof(unsigned char), cudaMemcpyHostToDevice) );
        //CU_CHECK_ERR( cudaMemcpy(curParam->output, h_B, matrix_mem_size, cudaMemcpyHostToDevice) );    
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
            run_bfs(argc, argv, false);
        }
    }
        
    CU_CHECK_ERR( cudaDeviceSynchronize() );
    std::cout << "Success!" << std::endl;
    
 
    for(i = 0; i < numpackets; i++){
            //CU_CHECK_ERR(cudaStreamDestroy(des_stream[i]));
            free(packet_in[i]);
            free(packet_out[i]);
            free(packet_open[i]);
    }

    free(des_esk);
    free(des_dsk);


    free(packet_in);
    free(packet_out);
    free(packet_open);


    return 0;
}

void configureParamMem(MemcParam* paramMem, size_t esk_mem_size,size_t mem_size,size_t len,size_t thread,size_t maxBatches)
{
    MemcParam* curParam = paramMem;

    for( unsigned batch=0; batch<maxBatches; ++batch ) {
        CU_CHECK_ERR( cudaMalloc((void**)&curParam->esk, esk_mem_size) );
        CU_CHECK_ERR( cudaMalloc((void**)&curParam->dsk, esk_mem_size) );
        CU_CHECK_ERR( cudaMalloc((void**)&curParam->input_o, mem_size) );
        CU_CHECK_ERR( cudaMalloc((void**)&curParam->output_o, mem_size) );
        curParam->len = len;
        curParam->thread = thread;
        int reg_buffer_size = 32 * IPSEC_REG_NUM * 512;
        CU_CHECK_ERR(cudaMalloc(&curParam->reg_buffer, reg_buffer_size));
        curParam->save_regs = true;
        curParam++;
    }
}

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
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
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

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