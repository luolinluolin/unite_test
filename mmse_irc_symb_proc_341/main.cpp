#include <stdio.h>

#include <sys/syscall.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sched.h>
#include <termios.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <time.h>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>
#include <ucontext.h>
#include <dlfcn.h>
#include <stdint.h>
#include <pthread.h>
#include <semaphore.h>

#include <map>
#include <tuple>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include "bblib_common.hpp"
#include "simd_insts.hpp"
#include "phy_rx_mimo_mmse.h"
#include "phy_pusch_symbol_processing_5gnr.h"
#include "phy_pusch_irc_symbol_processing_5gnr.h"

#define BIND_CORE   (5)
#define BIND_PRIO   (98)
#define BIND_POLICY (SCHED_FIFO)

using namespace W_SDK;

#define OUT_LOOP 200
#define IN_LOOP 50
#if 0
#define RING_BUFFER
#define MEM_NUM (IN_LOOP)
#else
#define MEM_NUM (1)
#endif
#define N_USED_SC 3280 // 3276 -> 3280
#define _aligned_malloc(x,y) memalign(y,x)

#define iAssert(p) if(!(p)){fprintf(stderr,\
    "Assertion failed: %s, file %s, line %d\n",\
    #p, __FILE__, __LINE__);exit(-1);}


int8_t* pllrs[16];
uint64_t MLogTick(void)
{
    uint32_t hi, lo;
    uint64_t time;

    __asm volatile ("rdtsc" : "=a"(lo), "=d"(hi));

    time = ( (uint64_t)lo)|( ((uint64_t)hi)<<32 );

    return time;
}

uint64_t timer_get_ticks_diff(uint64_t CurrTick, uint64_t LastTick)
{
    if (CurrTick >= LastTick)
        return (uint64_t)(CurrTick - LastTick);
    else
        return (uint64_t)(0xFFFFFFFFFFFFFFFF - LastTick + CurrTick);
}

static inline double get_rdtsc_freq_mhz(void)
{
    uint64_t start, end;
    double freq;
    struct timeval tv;

    tv.tv_sec = 1;
    tv.tv_usec = 0;

    start = MLogTick();
    while(select(0, NULL, NULL, NULL, &tv));
    end = MLogTick();
    freq = (double)timer_get_ticks_diff(end, start);
    freq /= 1000000.0;

    //printf("RDTSC => %0.1lf GHz\n", freq / 1000);

    return freq;
}


int sys_affinity_bind(int coreNum)
{
    cpu_set_t cpuset;
    int rc;

    /* set main thread affinity mask to CPU1 */

    CPU_ZERO(&cpuset);
    CPU_SET(coreNum, &cpuset);

    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc)
    {
        printf("pthread_setaffinity_np failed: %d", rc);
        return -1;
    }

    /* check the actual affinity mask assigned to the thread */

    CPU_ZERO(&cpuset);

    rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (rc)
    {
        printf("pthread_getaffinity_np failed: %d", rc);
        return -1;
    }

    if (!CPU_ISSET(coreNum, &cpuset))
    {
        printf("affinity failed\n");
        return -1;
    }

    return 0;
}


int sys_pthread_set_prio(int prio, int policy)
{
    int rc;

    if (prio)
    {
        struct sched_param sched_param;

        sched_param.sched_priority = prio;

        if ((rc = pthread_setschedparam(pthread_self(), policy, &sched_param))) {
            printf("priority is not changed: %d\n", rc);
            return -1;
        }

    }

    return 0;
}

void alloc_buffer (bblib_pusch_symbol_processing_request &request, bblib_pusch_symbol_processing_response &response) {
    int32_t i;
    int32_t nRBBlock = 70;
    int16_t nSymb;

    int32_t sizeCE = (N_USED_SC * 2) * BBLIB_MAX_RX_ANT_NUM * BBLIB_MAX_TX_LAYER_NUM * BBLIB_N_SYMB_PER_SF;
    int32_t sizeRnn = nRBBlock * BBLIB_MAX_RX_ANT_NUM * BBLIB_MAX_RX_ANT_NUM * BBLIB_N_SYMB_PER_SF;
    int32_t sizeIn = (N_USED_SC * 2) * BBLIB_N_SYMB_PER_SF * BBLIB_MAX_RX_ANT_NUM;
    int32_t sizeOut = (N_USED_SC * 2) * BBLIB_N_SYMB_PER_SF * BBLIB_MAX_TX_LAYER_NUM;
    int32_t sizeGain = (N_USED_SC) * BBLIB_MAX_TX_LAYER_NUM * BBLIB_N_SYMB_PER_SF;

    int16_t* pCEtp = (int16_t *)_aligned_malloc(sizeCE*2*MEM_NUM, 64);
    for (i = 0; i < sizeCE*MEM_NUM; i++){
        *(pCEtp + i) = rand() % 32767;
    }
    int16_t* pMIMOintp = (int16_t *)_aligned_malloc(sizeIn*2*MEM_NUM, 64);
    for (i = 0; i < sizeIn*MEM_NUM; i++){
        *(pMIMOintp + i) = rand() % 32767;
    }

    int32_t* pRnnRetp = (int32_t *)_aligned_malloc(sizeRnn*4*MEM_NUM, 64);
    for (i = 0; i < sizeRnn*MEM_NUM; i++){
        *(pRnnRetp + i) = rand() % 32767;
    }        
    int32_t* pRnnImtp = (int32_t *)_aligned_malloc(sizeRnn*4*MEM_NUM, 64);
    for (i = 0; i < sizeRnn*MEM_NUM; i++){
        *(pRnnImtp + i) = rand() % 32767;
    }

    int16_t* pMIMOouttp = (int16_t *)_aligned_malloc(sizeOut*2*MEM_NUM, 64);
    memset(pMIMOouttp, 0, sizeof(int16_t) * sizeOut*MEM_NUM);
    float* pSNRouttp = (float *)_aligned_malloc(sizeGain*4*MEM_NUM, 64);
    memset(pSNRouttp, 0, sizeof(float) * sizeGain*MEM_NUM);
    float* pGainouttp = (float *)_aligned_malloc(sizeGain*4*MEM_NUM, 64);
    memset(pGainouttp, 0, sizeof(float) * sizeGain*MEM_NUM);   


    for (int i = 0; i < request.nRxAnt; i++) {
        for (int j = 0; j < request.nSymb; j++) {
            nSymb = request.pSymbIndex[j];
            request.pRxSignal[i][nSymb] = (int16_t*)(pMIMOintp+(((request.nSymb*i)+j)*request.nTotalAlignedSubCarrier*2));
        }
    }
    
    for (int i = 0; i < request.nLayerInGroup; i++) {
        for (int j = 0; j < request.nSymb; j++) {
            nSymb = request.pSymbIndex[j];
            response.pEstTxSignal[i][nSymb] = (void*)(pMIMOouttp+(((request.nSymb*i)+j)*request.nTotalAlignedSubCarrier*2));
        }
    }
    
    for (int i = 0; i < request.nLayerInGroup; i++) {
        response.pMmseGain[i] = (float*)(pGainouttp + request.nSymb*request.nTotalAlignedSubCarrier*i);
    }
    
    for (int i = 0; i < request.nLayerInGroup; i++) {
        response.pPostSINR[i] = (float*)(pSNRouttp + request.nSymb*request.nTotalAlignedSubCarrier*i);
    }
    
    for(int iUe=0; iUe<request.nUeInGroup; iUe++)
    {
        sizeOut = request.nTotalSubCarrier * request.nSymb * request.nLayerInGroup*request.eModOrder[iUe];
        pllrs[iUe] = (int8_t *)_aligned_malloc(sizeOut, 64);
        memset(pllrs[iUe], 0, sizeof(int8_t) * sizeOut);

    }

    for (int i = 0; i < request.nRxAnt; i++) {
        for (int j = 0; j < request.nLayerInGroup; j++) {
            request.pChState[i][j] = (int16_t*)(pCEtp+(((request.nLayerInGroup*i)+j)*request.nChSymb*request.nTotalAlignedSubCarrier*2));
        }
    }

    for(int isym = 0; isym < request.nChSymb; isym ++) {
        for (int i = 0; i < request.nRxAnt; i++) {
            for (int j = 0; j < request.nRxAnt; j++) {
                request.pRnn_Re[isym][i][j] = (void*)(pRnnRetp + isym*request.nRxAnt*request.nRxAnt*nRBBlock
                    + i*request.nRxAnt*nRBBlock + j*nRBBlock);
            }
        }
    }

    for(int isym = 0; isym < request.nChSymb; isym ++) {
        for (int i = 0; i < request.nRxAnt; i++) {
            for (int j = 0; j < request.nRxAnt; j++) {
                request.pRnn_Im[isym][i][j] = (void*)(pRnnImtp + isym*request.nRxAnt*request.nRxAnt*nRBBlock
                    + i*request.nRxAnt*nRBBlock + j*nRBBlock);
            }
        }
    }

}


int main()
{
    //printf("TODO\n");
    printf("\n\n");

    sys_affinity_bind(BIND_CORE);
    //sys_pthread_set_prio(BIND_PRIO, BIND_POLICY);

    uint16_t nDataSymbNumA[BBLIB_N_SYMB_PER_SF] = { 0,1,3,4,5,6,7,8,9,10,12,13,0,0 };
    uint16_t nDmrsSymbNumA[BBLIB_N_SYMB_PER_SF] = { 2,11 };
    //uint16_t nDataSymbNumA_2p2[BBLIB_N_SYMB_PER_SF] = { 0,1,4,5,6,7,8, 9,12,13,0,0,0,0 };
    uint16_t nLayerPerUe[8] = {0};

    bblib_pusch_symbol_processing_request request{};
    bblib_pusch_symbol_processing_response response{};
    volatile uint64_t t1, t2, diff;
    volatile uint64_t nAvg = 0;
    uint32_t nInLoop, nOutLoop;
    int16_t nRB = 272;

    request.nStartSC = 0;
    request.nSubCarrier = nRB*12; 
    request.nTotalSubCarrier = nRB*12;
    request.nLinInterpEnable = 1;
    request.nTotalAlignedSubCarrier = nRB*12;
    request.nChSymb = 2;
    request.nSymb = 12;
    request.nRxAnt = 8;
    request.nLayerInGroup = 4;
    request.nUeInGroup = 2;
    request.nLlrFxpPoints = 5;
    request.nLlrSaturatedBits = 8;
    request.pSymbIndex = &nDataSymbNumA[0];
    request.pDmrsSymbolIdx = &nDmrsSymbNumA[0];
    request.nAgcGain = 0;
    request.nDisableRnnInv = 1;
    request.nEnable2ScProcess = 2; // 2SC processing
    request.nDMRSType = 1;
    request.nNrOfCDMs = 2;
    request.nNrOfDMRSSymbols = 1;

    request.nEnableFoComp = 0;
    if (request.nEnableFoComp == 0)
    {
        request.nFftSize = 0;
        request.nNumerology = 0;
        for (int16_t i = 0; i < request.nLayerInGroup; i++)
            request.fEstCfo[i] = 0.0;
    }

    for (int16_t iUe = 0; iUe < request.nUeInGroup; iUe++)
    {
        nLayerPerUe[iUe] = 2;
        request.eModOrder[iUe] = (enum bblib_modulation_order)(6); // 256QAM        
    }
    request.pLayerNumPerUE = nLayerPerUe;

    request.nTpFlag = 0;

    if (request.nLinInterpEnable == 1)
    {
        request.nDmrsChSymb = 2;
        request.nMappingType = 0; // typeA
        request.nGranularity = 2; // Linear3
        request.nSymb = BBLIB_N_SYMB_PER_SF - request.nDmrsChSymb;
    }

    alloc_buffer (request, response);

    double freq = get_rdtsc_freq_mhz();

    ////////////////////////////////////////////////////////////////////////////////////////
    // Calling functions
#ifdef SUBMODULE_PERF_UT
    bblib_pusch_symbol_processing_response *pResponse = &response;
#endif
    LOG_PERF_RESET(pResponse, PUSCH_SP_SUBMODULE_MAX);

    for (nOutLoop = 0; nOutLoop < OUT_LOOP; nOutLoop++){
        for (nInLoop = 0; nInLoop < IN_LOOP; nInLoop++){

	    int32_t nLlrOffset = 0;
            for (int16_t iUe = 0; iUe < request.nUeInGroup; iUe++)
            {
                for (int j = 0; j < request.nSymb; j++) {
                    int16_t nSymb = request.pSymbIndex[j];
                    response.pLlr[nSymb][iUe] = pllrs[iUe] + nLlrOffset;
	            nLlrOffset += request.nTotalSubCarrier * request.nLayerInGroup * request.eModOrder[iUe];
                }
            }
            t1 = MLogTick();
#ifdef _BBLIB_SPR_
            bblib_pusch_irc_symbol_processing_avx512_5gisa(&request, &response);
#else
            bblib_pusch_irc_symbol_processing_avx512(&request, &response);
#endif
            t2 = MLogTick();

            diff = timer_get_ticks_diff(t2, t1);
            nAvg += diff;
        }
    }

    LOG_PERF_PRINT(pResponse, PUSCH_SP_SUBMODULE_MAX);
    uint64_t nCycles = nAvg / IN_LOOP / OUT_LOOP;
    printf("IRC Symbol Process: nRxAnt:%2d, nLayer:%d, nRB:%4d, cycles: %8ld, delay: %10.3lf us (%0.1lf GHz)\n",
        request.nRxAnt, request.nLayerInGroup, request.nSubCarrier/12, nCycles, nCycles/freq, freq/1000);
    printf("\n\n");

    return 0;
}
