#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

extern __device__ int* gLocks;
extern __device__ int* gDebugPtr;


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

  

