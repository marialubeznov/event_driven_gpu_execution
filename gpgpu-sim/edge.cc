#include "edge.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpu/gpgpu-sim/cuda_gpu.hh"
/**
 * Debug 
 */
const char* EdgeDebugLevelStr[] = {
    "EdgeDebug",
    "EdgeInfo",
    "EdgeNotice",
    "EdgeWarning",
    "EdgeErr",
    "EdgeCrit",
    "EdgeAlert",
    "EdgeEmerg"
};


/**
 * GPUEvent
 */
GPUEvent::GPUEvent(int eid, GPUEventType type, kernel_info_t* k, new_addr_type paramBaseAddr, size_t paramSize, int maxEvents)
    : _eid(eid), _type(type), _eventKernel(k), _paramBaseAddr(paramBaseAddr), _nextParamAddr(paramBaseAddr), 
        _paramSize(paramSize), _nEventInProgress(0), _maxEvents(maxEvents)
{
    
}

GPUEvent::GPUEvent(int eid, GPUEventType type, kernel_info_t* k) 
    : _eid(eid), _type(type), _eventKernel(k), _nEventInProgress(0)
{
    
}

GPUEvent::GPUEvent(int eid, GPUEventType type, kernel_info_t* k1, kernel_info_t* k2) 
    : _eid(eid), _type(type), _eventKernel(k1), _eventKernel2(k2), _nEventInProgress(0)
{
    
}

GPUEvent::~GPUEvent()
{
    
}

bool GPUEvent::beginEvent()
{
    if( _nEventInProgress+1 > _maxEvents ) 
        return false;
    _nEventInProgress++;
    return true;
}

void GPUEvent::completeEvent()
{
    _nEventInProgress--;
    assert( _nEventInProgress >= 0 );
}

new_addr_type GPUEvent::getNextParamAddr()
{
    new_addr_type addr = _nextParamAddr;
    _nextParamAddr = _nextParamAddr + _paramSize;
    if( _nextParamAddr >= (_paramBaseAddr + (_maxEvents*_paramSize)) ) {
        _nextParamAddr = _paramBaseAddr;
    }
    return addr;
}

/**
 * GPUEventManager
 */
GPUEventManager::GPUEventManager(class gpgpu_sim* gpu, int nCores)
    : _gpu(gpu), _nCores(nCores), _coreToSchedule(_gpu->get_shader(0)->get_config()->_edgeSingleSmForIsr==1? _gpu->get_shader(0)->get_config()->_edgeSingleSmForIsrIdx:0), _dynamicEventID(0), _edgeCycle(0)
{
    _coreEvents.resize(nCores);
    _coreWbQueues.resize(nCores);
    _eventsInProgress = new GPUEvent*[nCores];

    // Register a NULL event for testing and launching the interrupt warp without an Event to schedule.
    assert( registerGPUEvent(EDGE_NULL, NULL, 0, 0, 0) == NULL_EVENT_ID ); 

    // Register a warmup kernel to warmup the instruction cache with interrupt instructions. 
    assert( registerGPUEvent(EDGE_WARMUP, NULL, 0, 0, 0) == WARMUP_EVENT_ID ); 
}

GPUEventManager::~GPUEventManager()
{
    assert( _eventsInProgress );
    delete[] _eventsInProgress;
}

bool GPUEventManager::warmup()
{
    
    // if (_gpu->get_shader(0)->get_config()->_edgeSingleSmForIsr==1) {
    //     EDGE_DPRINT(EdgeDebug, "Warming up interrupt i-cache! Launching the warmup i-warp on cores %d\n", _gpu->get_shader(0)->get_config()->_edgeSingleSmForIsrIdx);
    //     if( !scheduleCoreEvent(_gpu->get_shader(0)->get_config()->_edgeSingleSmForIsrIdx, WARMUP_EVENT_ID) )
    //             return false;
    // }
    // else {
        EDGE_DPRINT(EdgeDebug, "Warming up interrupt i-cache! Launching the warmup i-warp on all cores\n");
        for( unsigned core=0; core<_nCores; ++core ) {
            if( !scheduleCoreEvent(core, WARMUP_EVENT_ID) )
                return false;
        }
    //}       

    scheduleGem5Cycle();
    return true;
}

bool GPUEventManager::scheduleGem5Cycle()
{
    return _gpu->gem5CudaGPU->ScheduleGem5Cycle(0);
}

void GPUEventManager::cycle()
{
    // See if we can schedule any events. Then service a potential register read from each core.
     
    for( unsigned i=0; i<_nCores; ++i ) {
        if( pendingCoreEvents(i) ) {
            //printf("MARIA: edge cycle, trying to send an interrupt for core %d\n", i);
            // There is an event pending for Core i. Send the interrupt signal to core i.
            _gpu->setIntSignal(i);
        }

        if( !_coreWbQueues[i].empty() ) { 
            wbReqState& wbs = _coreWbQueues[i].front();
            warp_inst_t& inst = wbs._inst;
            if( !_gpu->get_shader(i)->ldst_unit_wb_inst(inst) ) {
                // Writeback is blocked, stall writeback
                 
            } else {
                uint8_t data[16];
                *(int*)data = wbs._writeVal;
                _gpu->get_shader(i)->writeRegister(inst, 32, 0, (char*)data);
                _coreWbQueues[i].pop();
            }
        }
    }
   
    _edgeCycle++;
}

int GPUEventManager::registerGPUEvent(GPUEventType type, kernel_info_t* k, new_addr_type paramBaseAddr, size_t paramSize, int maxEvents)
{
    if( _events.size() >= MAX_NUM_EVENTS ) {
        EDGE_DPRINT(EdgeDebug, "Too many events added to the EventManager\n");
        return -1;
    }
    int eventId = _dynamicEventID++;
    GPUEvent* e = new GPUEvent(eventId, type, k, paramBaseAddr, paramSize, maxEvents);    
    _events[eventId] = e;

    // Conditionally reserve resources in each SM
    if( k )
        _gpu->reserveEventResources(k); 
    
    return eventId;
}

int GPUEventManager::registerGPUEvent(GPUEventType type, kernel_info_t* k1, kernel_info_t* k2)
{
    if( _events.size() >= MAX_NUM_EVENTS ) {
        EDGE_DPRINT(EdgeDebug, "Too many events added to the EventManager\n");
        return -1;
    }
    int eventId = _dynamicEventID++;
    GPUEvent* e = new GPUEvent(eventId, type, k1, k2);    
    _events[eventId] = e;
    
    // Conditionally reserve resources in each SM
    if( k1 )
        _gpu->reserveEventResources(k1); 

    return eventId;
}

int GPUEventManager::registerGPUEvent(GPUEventType type, kernel_info_t* k)
{
    if( _events.size() >= MAX_NUM_EVENTS ) {
        EDGE_DPRINT(EdgeDebug, "Too many events added to the EventManager\n");
        return -1;
    }
    int eventId = _dynamicEventID++;
    GPUEvent* e = new GPUEvent(eventId, type, k);    
    _events[eventId] = e;
    
    // Conditionally reserve resources in each SM
    if( k )
        _gpu->reserveEventResources(k); 

    return eventId;
}

int GPUEventManager::registerGPUEvent(GPUEventType type)
{    
    if( _events.size() >= MAX_NUM_EVENTS ) {
        EDGE_DPRINT(EdgeDebug, "Too many events added to the EventManager\n");
        return -1;
    }
    
    assert( type == EDGE_RELEASE_BAR );

    int eventId = _dynamicEventID++;
    GPUEvent* e = new GPUEvent(eventId, type, NULL);    
    _events[eventId] = e;
    
    return eventId;
}


bool GPUEventManager::configureEvent(int eventId, new_addr_type paramBaseAddr, size_t paramSize, int maxEvents)
{
    if( !validEvent(eventId) ) {  
        return false;
    }
        
    GPUEvent* e = _events[eventId];
    e->setParamMem(paramBaseAddr);
    e->setParamSize(paramSize);
    e->setMaxEvents(maxEvents);

    return true;
}


bool GPUEventManager::removeGPUEvent(int eventId)
{
    if( !validEvent(eventId) ) {  
        return false;
    }
    _events.erase(eventId);
    return true;
}

/**
 * External CPU process or hardware event can trigger this
 * scheduling. The eventId is the interrupt ID specifying what to do 
 * on this interrupt. 
 */
bool GPUEventManager::scheduleGPUEvent(int eventId)
{
    if( !validEvent(eventId) )
        return false;

    GPUEventType type = _events[eventId]->getType();

    if( type == EDGE_USER_EVENT || type == EDGE_NULL ) {
        if( !scheduleCoreEvent(_coreToSchedule, eventId) ) 
            return false;

        nextCoreToSchedule(); 
    } else if ( type == EDGE_RELEASE_BAR ) {
        for( unsigned i=0; i<_nCores; ++i ) {
            if( !scheduleCoreEvent(i, eventId) )
                return false;
        }
    }

    scheduleGem5Cycle();
    return true;
}



bool GPUEventManager::scheduleGPUTimerEvent(int eventId, unsigned long long N)
{
    if( !validEvent(eventId) ) 
        return false;

    bool ret = _gpu->scheduleTimerEvent(eventId, N);

    if( ret )
        scheduleGem5Cycle();

    return ret;
}

bool GPUEventManager::scheduleGPUTimerBatchEvent(int eventId, unsigned long long Nouter, unsigned long long batch, unsigned long long Ninner) {
    if( !validEvent(eventId) ) 
        return false;

    bool ret = _gpu->scheduleTimerBatchEvent(eventId, Nouter, batch, Ninner);

    if( ret )
        scheduleGem5Cycle();

    return ret;
}


bool GPUEventManager::scheduleCoreEvent(unsigned smID, int eventId)
{
    if( !validEvent(eventId) ) 
        return false;

    if( smID >= _nCores )  {
        EDGE_DPRINT(EdgeErr, "ScheduleCoreEvent - Invalid smID %d\n", smID);
        return false;
    }

    GPUEvent* event = _events[eventId];
    _coreEvents[smID].push(event);
    if (eventId != WARMUP_EVENT_ID) {
    	_gpu->get_shader(smID)->_edgeInterruptAssertCycle.push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
        _gpu->get_shader(smID)->_edgeInterruptAssertCycleForISR.push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
    	//printf("MARIA EDGE: new incoming interrupt on core %d at time %lld\n", smID, gpu_sim_cycle + gpu_tot_sim_cycle);
    };
    return true;
}

bool GPUEventManager::pendingEvents() const
{
    for( unsigned i=0; i<_nCores; ++i ) {
        if( pendingCoreEvents(i) )
            return true;
    }
    return false;
}

int GPUEventManager::pendingCoreEvents(unsigned smID) const
{
    if( smID >= _nCores ) 
        return -1;
    
    return _coreEvents[smID].size();
}

bool GPUEventManager::nextCorePendingEvent() const
{
   return pendingCoreEvents(_coreToSchedule);
}


void GPUEventManager::beginInt(unsigned smID)
{
    if( smID >= _nCores || _coreEvents[smID].empty() )
        return;

    //EDGE_DPRINT(EdgeDebug, "Starting interrupt for event %p on core %d\n", _coreEvents[smID].front(), smID);
    _eventsInProgress[smID] = _coreEvents[smID].front();
    _coreEvents[smID].pop(); // Remove from the queue for this core
}

void GPUEventManager::completeInt(unsigned smID)
{
    if( smID >= _nCores ) 
        return;

    //EDGE_DPRINT(EdgeDebug, "Completing interrupt for event %p on core %d\n", _eventsInProgress[smID], smID);
    _eventsInProgress[smID] = NULL; 
}

GPUEvent* GPUEventManager::eventInProgress(unsigned smID)
{
    if( !_eventsInProgress[smID] )
        return NULL;
    else
        return _eventsInProgress[smID];
}


bool GPUEventManager::validEvent(int eventId)
{
    if ( _events.find(eventId) == _events.end() ) {
       EDGE_DPRINT(EdgeErr, "Invalid eventId: %d\n", eventId);
       return false;
    } else {
        return true;
    }
}

void GPUEventManager::nextCoreToSchedule()
{
    if (_gpu->get_shader(0)->get_config()->_edgeSingleSmForIsr==1)
        _coreToSchedule = _gpu->get_shader(0)->get_config()->_edgeSingleSmForIsrIdx;
    else 
        _coreToSchedule = (_coreToSchedule + 1) % _nCores;
}

void GPUEventManager::pushWBQueue(int smID, const warp_inst_t& inst, int val)
{
    _coreWbQueues[smID].push( wbReqState(inst, val) );
}

kernel_info_t* GPUEventManager::getKernel(int eventId)
{
    if( !validEvent(eventId) ) 
        return NULL;

    return _events[eventId]->getKernel();
}

kernel_info_t* GPUEventManager::getKernel2(int eventId)
{
    if( !validEvent(eventId) ) 
        return NULL;

    return _events[eventId]->getKernel2();
}

GPUEvent* GPUEventManager::getEvent(int eventId)
{
    if( !validEvent(eventId) ) 
        return NULL;

    return _events[eventId];
}
