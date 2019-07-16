#ifndef __DNN_CONFIG_H__
#define __DNN_CONFIG_H__

#include "option_parser.h"

class CudaConfig {
    
public:
    CudaConfig(){};
    ~CudaConfig(){};
    
    void reg_options(option_parser_t opp);
    
    // Config to run
    unsigned config_to_run;

    unsigned dev_num;
};

#endif

