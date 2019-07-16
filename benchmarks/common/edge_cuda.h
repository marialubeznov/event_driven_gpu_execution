#ifndef __EDGE_CUDA_H__
#define __EDGE_CUDA_H__

struct EventScheduleParams {
	int eventId;
	unsigned long long N;
	unsigned long long maxEventsNum;
};


// EDGE API: TODO: Move to separate header
extern "C"  __host__ cudaError_t cudaScheduleEvent(int eventID);
extern "C"  __host__ cudaError_t cudaDeviceSynchronize();
extern "C" __host__ int cudaRegisterEvent(void* kernel, void* kernel2, dim3 grid, dim3 block, size_t sharedmem);
extern "C" __host__ cudaError_t cudaSetupEventArgument(size_t size, size_t offset);
extern "C" __host__ void* cudaConfigureEventParam(int eventId, size_t paramSize, size_t maxInFlightKernels, bool child_kernel);
extern "C" __host__ cudaError_t edgeExtra(int op, void* sr, size_t nReqs, size_t batchSize, const char* filename); 
extern "C" __host__ cudaError_t edgeExtraipv6(int op, void* sr, int portmask, void* prefix_arr, int add_prefixes, int n_requests_per_batch,int n_batches); 

extern "C" __host__ cudaError_t cudaScheduleTimerEvent(int eventId, unsigned long long N);
extern "C" __host__ cudaError_t cudaScheduleEventTimerBatch(int eventID, unsigned long long Nouter, unsigned long long batch, unsigned long long Ninner);
extern "C" __host__ int cudaRegisterBarrier();
extern "C" __host__ cudaError_t cudaSleepThread(size_t nGpuCycles);

#endif /* __EDGE_CUDA_H__ */
