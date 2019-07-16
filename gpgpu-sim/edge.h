#ifndef __EDGE_H__
#define __EDGE_H__

#include <stdio.h>
#include <vector>
#include <queue>
#include <map>

#include "abstract_hardware_model.h"
#include "edge_helper.h"

enum EdgeDebugLevel {
    EdgeDebug,
    EdgeInfo,
    EdgeNotice,
    EdgeWarning,
    EdgeErr,
    EdgeCrit,
    EdgeAlert,
    EdgeEmerg,
    EdgeNumLevels
};

extern const char* EdgeDebugLevelStr[];

#define EDGE_DBG
#ifdef EDGE_DBG
    #define EDGE_DBG_LVL EdgeDebug
    #define EDGE_DPRINT(x, ...)                         \
        do {                                            \
            if( x >= EDGE_DBG_LVL ) {                   \
                printf("%s: ", EdgeDebugLevelStr[x]);   \
                printf(__VA_ARGS__);                    \
            }                                           \
        } while(0)
#else
    #define EDGE_DBG_LVL emerg
    #define EDGE_DPRINT(x, ...)
#endif
#endif



enum GPUIntWarpSelection {
    EDGE_DEDICATED,
    EDGE_OLDEST,
    EDGE_NEWEST, 
    EDGE_RANDOM,
    EDGE_BEST,
    EDGE_PSEUDO_DEDICATED,
    EDGE_NUM_WARP_SELECTION
};


class gpgpu_sim;

// TODO: 
class GPUEvent {
public:
    GPUEvent() : _type(EDGE_NULL), _eventKernel(NULL) {};
    
    GPUEvent(int eid, GPUEventType type, kernel_info_t* k, new_addr_type paramBaseAddr, size_t paramSize, int maxEvents);
    GPUEvent(int eid, GPUEventType type, kernel_info_t* k);
    GPUEvent(int eid, GPUEventType type, kernel_info_t* k1, kernel_info_t* k2);

    GPUEvent(GPUEvent& rhs) {
        _type               = rhs._type;
        _eid                = rhs._eid;
        _eventKernel        = rhs._eventKernel;
        _paramBaseAddr      = rhs._paramBaseAddr;
        _paramSize          = rhs._paramSize;
        _nEventInProgress   = rhs._nEventInProgress;
        _maxEvents          = rhs._maxEvents;
    }
    ~GPUEvent();

    int getEventId() const { return _eid; }
    GPUEventType getType() const { return _type; }
    kernel_info_t* getKernel() { return _eventKernel; }
    kernel_info_t* getKernel2() { return _eventKernel2; }
    bool isNull() const { return (_type == EDGE_NULL); }
    
    new_addr_type getParamBaseAddr() const { return _paramBaseAddr; }
    new_addr_type getNextParamAddr();

    void setParamMem(new_addr_type paramBaseAddr) { _paramBaseAddr = paramBaseAddr; _nextParamAddr = paramBaseAddr; }
    void setParamSize(size_t paramSize) { _paramSize = paramSize; }
    void setMaxEvents(int maxEvents) { _maxEvents = maxEvents; }

    bool beginEvent();
    void completeEvent();

private:
    int             _eid;
    GPUEventType    _type;
    kernel_info_t*  _eventKernel;
    kernel_info_t*  _eventKernel2;

    new_addr_type   _paramBaseAddr;
    new_addr_type   _nextParamAddr;
    size_t          _paramSize; 

    int             _nEventInProgress;
    int             _maxEvents;
};

#define MAX_NUM_EVENTS  64

struct wbReqState {
    wbReqState() : _writeVal(-1) {}; 
    wbReqState(warp_inst_t inst, int val) : _inst(inst), _writeVal(val) {};
    
    wbReqState(const wbReqState& rhs) {
        _inst       = rhs._inst;
        _writeVal   = rhs._writeVal;
    }
    
    warp_inst_t     _inst;
    int             _writeVal;
};

class EventList             : public std::queue<GPUEvent*> {};
class CoreEvents            : public std::vector<EventList> {};
class GPUEventMap           : public std::map<unsigned,GPUEvent*> {};

class WriteBackQueue        : public std::queue<wbReqState> {}; 
class CoreWriteBackQueues   : public std::vector<WriteBackQueue> {};

/**
 *  This is an event Manager
 */
class GPUEventManager {

public:
    GPUEventManager(class gpgpu_sim* gpu, int nCores);
    ~GPUEventManager();

    /// Event manager cycle operation for timing simulation 
    void cycle();

    /// Add the new event to the Event Manager
    int registerGPUEvent(GPUEventType type, kernel_info_t* k, new_addr_type paramBaseAddr, size_t paramSize, int maxEvents);
    int registerGPUEvent(GPUEventType type, kernel_info_t* k);
    int registerGPUEvent(GPUEventType type, kernel_info_t* k1, kernel_info_t* k2);
    int registerGPUEvent(GPUEventType type);

    bool configureEvent(int eventId, new_addr_type paramBaseAddr, size_t paramSize, int maxEvents);
    bool warmup();


    /// Remove a registered GPU event from the event manager. 
    bool removeGPUEvent(int eventId);

    /// Schedule a GPU event, eventId, to run on the GPU. Invokes the interrupt 
    /// handler on a selected core.
    bool scheduleGPUEvent(int eventId);
    
    bool scheduleGPUTimerEvent(int eventId, unsigned long long N);

    bool scheduleGPUTimerBatchEvent(int eventId, unsigned long long Nouter, unsigned long long batch, unsigned long long Ninner);

    /// Returns true if there are any pending GPU events 
    bool pendingEvents() const; 

    /// Returns the number of pending events on core, smID
    int pendingCoreEvents(unsigned smID) const;

    /// Checks if the next core to schedule already has a pending event or not
    bool nextCorePendingEvent() const;

    /// Signals to the Event Manager that core, smID, is starting to process
    /// the event at the head of the queue for core, smID. 
    void beginInt(unsigned smID);

    /// Signals to the Event Manager that core, smID, is completing to process
    /// the interrupt to schedule the event at the head of the queue for core, smID.   
    void completeInt(unsigned smID);

    /// Returns the event currently being processed by core, smID. This will return
    /// NULL if not between valid BeginEvent and EndEvent calls.
    GPUEvent* eventInProgress(unsigned smID);
    
    /// For warp instruction, inst, running on core, smID, set val as the value to writeback
    /// to a pending load instruction.
    void pushWBQueue(int smID, const warp_inst_t& inst, int val);

    kernel_info_t* getKernel(int eventId);
    kernel_info_t* getKernel2(int eventId);
    GPUEvent* getEvent(int eventId);

private:
    /// Schedule a GPU event, eventId, on core, smID. Called from ScheduleGPUEvent. 
    bool scheduleCoreEvent(unsigned smID, int eventId); 

    /// Schedule a Gem5 cycle event if not already scheduled
    bool scheduleGem5Cycle();

    /// Checks if eventId is a registered event
    bool validEvent(int eventId);

    /// Cycles the Event Manager for the next core to schedule an event
    void nextCoreToSchedule();

    class gpgpu_sim*    _gpu;               //< Back pointer to GPGPU-Sim for handling events
    int                 _nCores;            //< # of cores in GPGPU-Sim
    GPUEventMap         _events;            //< Map of event IDs to registered events
    CoreEvents          _coreEvents;        //< Queue of scheduled/in progress events for a given core
    GPUEvent**          _eventsInProgress;  //< Event in progress by a given core
    int                 _coreToSchedule;    //< Index of the next core to schedule an event to
    int                 _dynamicEventID;    //< Dynamic ID of the next event to register

    CoreWriteBackQueues _coreWbQueues;      //< A list of writeback queues for each core

    size_t              _edgeCycle;
};
