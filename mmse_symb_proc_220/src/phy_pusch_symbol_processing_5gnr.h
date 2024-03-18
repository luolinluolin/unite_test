/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_pusch_symbol_processing_5gnr.h
    \brief  External API for 5GNR PUSCH symbol processing.

   __Overview:__
   This module implements MMSE MIMO equalization, layer demapping,and LLR dempping.
   MMSE MIMO equalization supports 1x2, 1x4, 2x2 and 2x4; and 4x4 in 2UE MU case
   LLR demapping supports pi/2 BPSK, QPSK, 16QAM and 64QAM

   __Algorithm Guidance:__
   1. MMSE MIMO equalization refers to lib_equalization
   3. Layer demapping refers to lib_layerdemapping_5gnr
   4. LLR demapping refers to the inline comments
 */

#ifndef _PHY_PUSCH_SYMBOL_PROCESSING_H_
#define _PHY_PUSCH_SYMBOL_PROCESSING_H_

#include "common_typedef_sdk.h"
#include "bblib_common_const.h"
//#include "simd_insts.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

//using namespace W_SDK;

#define PUSCH_MAX_DMRS_SYMBOL       (4)
#define PUSCH_MAX_DMRS_PORT_NUM     (4)


/*!
    \enum SP_SUBMODULE_TYPE
    \brief SP submodules
*/
enum SP_SUBMODULE_TYPE {
    PUSCH_SP_H_INTERP = 0,
    PUSCH_SP_LoadH_P1,
    PUSCH_SP_LoadH_P2,
    PUSCH_SP_HxH,
    PUSCH_SP_INV,
    PUSCH_SP_GAIN,
    PUSCH_SP_EQU,
    PUSCH_SP_DEMOD,
    PUSCH_SP_SUBMODULE_MAX
};



/*!
    \struct bblib_pusch_symbol_processing_request
    \brief Request struct of PUSCH symbol processing
*/
typedef struct {
    uint16_t nLayerInGroup;                     /*!< Number of Layers in One MU Group - valid values 1, 2, 4*/
    uint16_t *pLayerNumPerUE;                   /*!< Number of Layers per UE, support different number of layers - valid values 1, 2 */
    uint16_t nUeInGroup;                        /*!< Number of UEs in This MU Group - valid values 1,2 */
    uint16_t nRxAnt;                            /*!< Number of Rx Virtual Antennas  - valid values 1, 2, 4 */
    uint16_t nStartSC;                          /*!< Start Subcarrier Index*/
    uint16_t nSubCarrier;                       /*!< Number of Granted Subcarriers For This Function Call - valid range [1-3276] */
    uint16_t nTotalSubCarrier;                  /*!< Number of Total Granted Subcarriers, Used In RB Level Split Case, Not Less Than nSubCarrier - valid range [1-3276] */
    uint16_t nTotalAlignedSubCarrier;           /*!< Number of Aligned Total Granted Subcarriers, Decide the Buffer Offset, Need to Be Mulitple of 16. */
    uint16_t nChSymb;                           /*!< Number of Channel Symbols - valid range [1-(N_SYMB_PER_SF-1)] */
    uint16_t nSymb;                             /*!< Number of Total Granted Symbols - valid range [1-(N_SYMB_PER_SF-1)]. Dummy, not used right now */
    uint16_t nSymbPerDmrs[BBLIB_N_SYMB_PER_SF]; /*!< Number of Granted Symbols Per Dmrs  - valid range [1-(N_SYMB_PER_SF-1)] */
    //used for data/dmrs interleaved
    uint8_t  nDMRSType;                         /*!< DMRS Type for PUSCH */
    uint8_t  nNrOfCDMs;                         /*!< Number of CDM DMRS Groups without data */
    uint8_t  nNrOfDMRSSymbols;                  /*!< Number of front loaded DMRS symbols */
    uint8_t  nTotalDmrsSymbol;                  /*!< Number of DMRS Symbols in slot */
    uint16_t *pDmrsSymbolIdx;                   /*!< Pointer for DMRS Symbol indexes */
    uint8_t  *pDmrsPortIdx[BBLIB_MAX_MU];       /*!< Pointer for DMRS Port Indexes for each UE in this group */
    uint8_t  nLlrFxpPoints;                     /*!< Indicate the Decimal Digits of Llr Output Fixed Point Value. Right now need to be 0~7 */
    uint8_t  nLlrSaturatedBits;                    /*!< Indicate the Total digits of Llr Output. Right now need to be 6~8 */
    uint8_t  nTpFlag;                           /*!< Indicate Transform Precoding is Enabled or Not, 0 disable, other enable */
    float    nSigma2;                           /*!< Noise power */
    enum bblib_modulation_order eModOrder[BBLIB_MAX_MU]; /*!< Supported Modulation Values are: 1 (pi/2 BPSK), 2 (QPSK), 4 (16QAM), 6 (64QAM) */
    uint16_t *pSymbIndex;                       /*!< Pointer for Data Symbol index */
    void * pChState[BBLIB_MAX_RX_ANT_NUM][BBLIB_MAX_TX_LAYER_NUM]; /*!< Data Pointer Points to nRxAnt*nTxLayer Channel, The Layers Need to Be Mapped UE by UE to Get Correct Layer Demapping Output, format 16S13 */
    void * pRxSignal[BBLIB_MAX_RX_ANT_NUM][BBLIB_N_SYMB_PER_SF];   /*!< Data Pointer Points to nRxAnt*nSymbol Received Data, format 16S13 */
    float nAgcGain;  /*!< Digital AGC Gain in num of shift bits  or  convert scaling for fp16*/

    // For linear interpolation support
    int16_t nLinInterpEnable; /*!< 0 - stored DMRS CE is used directly (nearest neighbor), 1 - time linear interpolation is used */
    int16_t nDmrsChSymb; /*!< Number of Dmrs Symbols - valid range [1-4]. Defines how many CEs stored on input at this function call */
    int16_t nMappingType; /*!< Dmrs Type - A is 0 and B is 1 */
    int16_t nGranularity; /*!< Defines the number of adjacent symbols sharing the same CE inside the slot */
    int16_t *pIntrpweights[BBLIB_MAX_TX_LAYER_NUM]; /*!< Pointer to array of 14x4 interpolation weights for each layer */

    // FOC support
    float    fEstCfo[BBLIB_MAX_TX_LAYER_NUM]; /*!< Angle offset estimates for each layer */
    int16_t nFftSize; /*!< FFT Size */
    int16_t nNumerology; /*!< Numerology */
    int16_t nEnableFoComp; /*!< Flag to enable frequency offset compensation */
    int16_t *pFoCompScCp; /*!< FO table */

    //PTRS Interleave Support
    uint16_t  nPtrsPresent[BBLIB_MAX_MU]; /*!< PTRS Enable/Disable flag for each UE */
    uint16_t  nPtrsPortIndex[BBLIB_MAX_MU]; /*!< PTRS port index for each UE */
    uint16_t  nPtrsTimeDensity[BBLIB_MAX_MU]; /*!< PTRS time density for each UE */
    uint16_t  nPtrsFreqDensity[BBLIB_MAX_MU]; /*!< PTRS freq density for each UE  */
    uint16_t  nPtrsReOffset[BBLIB_MAX_MU]; /*!< PTRS RE Offset for each UE */
    uint16_t  nRnti[BBLIB_MAX_MU]; /*!< nRnti for each UE */
    uint16_t  nPtrsSymbolIdx[BBLIB_MAX_MU][BBLIB_N_SYMB_PER_SF]; /*!< nPtrsSymbolIdx for each UE */
    uint8_t   nTotalPtrsSymbol[BBLIB_MAX_MU]; /*!< nTotalPtrsSymbol for each UE */

    // For MMSE IRC support
    uint16_t nIrcRbgStart; /*!< Start IRC RBG Index*/
    void * pRnn_Re[BBLIB_N_SYMB_PER_SF][BBLIB_MAX_RX_ANT_NUM][BBLIB_MAX_RX_ANT_NUM]; /*!< Data pointer points to nRxAnt*nRxAnt*nSymbol Rnn data */
    void * pRnn_Im[BBLIB_N_SYMB_PER_SF][BBLIB_MAX_RX_ANT_NUM][BBLIB_MAX_RX_ANT_NUM]; /*!< Data pointer points to nRxAnt*nRxAnt*nSymbol Rnn data */
    int16_t nDisableRnnInv; /*!< 0: calulate Rnn inversion in MMSE-IRC SDK; 1: disable the calculation */
    int16_t nEnable2ScProcess; /*!< 0: default, process for every subcarrier; 1:, process for every 2SC subcarrier*/
    int16_t nBufferShuffle; /*!< 0: default, process with 16Rx/8Rx buffer shuffle; 1:, process without buffer shuffle*/

    bblib_pusch_xran_decomp *pPuschDecomp;  /*!< XRAN decompress pointer */

} bblib_pusch_symbol_processing_request;

/*!
    \struct bblib_pusch_symbol_processing_response
    \brief Response struct of PUSCH symbol processing
*/
typedef struct {
    int8_t *pLlr[BBLIB_N_SYMB_PER_SF][BBLIB_MAX_MU];  /*!< Pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints) */
    float  *pMmseGain[BBLIB_MAX_TX_LAYER_NUM];   /*!< Pointer Points to nTx*1 Estimated MMSE Gain, floating number */
    float  *pPostSINR[BBLIB_MAX_TX_LAYER_NUM];   /*!< Pointer Points to nTx*1 pPostSINR, floating number */
    float  *pMmseOutReal[BBLIB_MAX_TX_LAYER_NUM][BBLIB_N_SYMB_PER_SF]; /*!< Pointer Points to nTx*1 MMSE output real part, debug interface, floating number */
    float  *pMmseOutImag[BBLIB_MAX_TX_LAYER_NUM][BBLIB_N_SYMB_PER_SF]; /*!< Pointer Points to nTx*1 MMSE output real part, debug interface, floating number */
    uint32_t nAgcOverflow; /*!< Total number of overflow IQ data after digital AGC */
    float  fPostSINR[BBLIB_MAX_TX_LAYER_NUM];    /*!< Reported Post_SNR per Layer, estimated by this PUSCH Symbol Process */
    // For IRC
    void   *pEstTxSignal[BBLIB_MAX_TX_LAYER_NUM][BBLIB_N_SYMB_PER_SF]; /*!< Data pointer points to nTx*nSymbol estimated TX signal */
#ifdef SUBMODULE_TICK
    uint32_t n_cnt; /*!< counter of each submodule, average tick number = n_tick / n_cnt */
    uint64_t n_tick[PUSCH_SP_SUBMODULE_MAX]; /*!< total cycles of each submodules with peroid of n_cnt*/
#endif
} bblib_pusch_symbol_processing_response;


/*! \brief Report the version number for the bblib_pusch_symbol_processing library
    \param [in] version Pointer to a char buffer where the version string should be copied.
    \param [in] buffer_size The length of the string buffer, must be at least
           BBLIB_SDK_VERSION_STRING_MAX_LEN characters.
    \return 0 if the version string was populated, otherwise -1.
*/
int16_t
bblib_pusch_symbol_processing_version(char *version, int buffer_size);

//! @{
/*! \brief 5GNR PUSCH symbol processing: MMSE MIMO+TA Compensation+Layer demapping+LLR demapping
    \param [in] request Input request structure for PUSCH symbol processing
    \param [out] response Output response structure for PUSCH symbol processing
    \return 0 for success, and -1 for error
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
// int32_t
// bblib_pusch_symbol_processing(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response *response);

// int32_t
// bblib_pusch_symbol_processing_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response);


// //! @{
// /*! \brief MMSE MIMO detection, with post SNR calculation.
//     \param [in] request Input request structure for MMSE MIMO.
//     \param [out] response Output response structure for MMSE MIMO..
//     \return 0 for success, and -1 for error
//     \warning
//     \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
// */
int32_t
bblib_pusch_symbol_processing(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response *response);

int32_t
bblib_pusch_symbol_processing_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response);

int32_t
bblib_pusch_symbol_processing_avx512_5gisa(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response);
//! @}

/*!
    \struct bblib_layer_llr_demap_request
    \brief Request struct of layer demapping and LLR demapping processing
*/
typedef struct {
    uint16_t nLayer;                             /*!< Number of Layers per UE, Only Support Equal Layers Number Now - valid values 1, 2*/
    uint16_t nLen;                               /*!< Number of Continguous Subcarrier Number */
    uint8_t nLlrFxpPoints;                       /*!< Indicate the Decimal Digits of Llr Output Fixed Point Value. Right now need to be 0~7 */
    uint8_t  nLlrSaturatedBits;                    /*!< Indicate the Total digits of Llr Output. Right now need to be 6~8 */
    enum bblib_modulation_order eModOrder;       /*!< Supported Modulation Values are: 2 (QPSK), 4 (16QAM), 6 (64QAM) */
    int16_t * pEqualOut[BBLIB_MAX_TX_LAYER_NUM]; /*!< Data Pointer Points to Equalization Output Data, Requires Memory Continguous, format 16S13 */
    float  *pMmseGain[BBLIB_MAX_TX_LAYER_NUM];   /*!< Pointer Points to nTx*1 Estimated MMSE Gain, floating number */
    float  *pPostSINR[BBLIB_MAX_TX_LAYER_NUM];   /*!< Pointer Points to nTx*1 pPostSINR, floating number */
} bblib_layer_llr_demap_request;

/*!
    \struct bblib_layer_llr_demap_response
    \brief Response struct of layer demapping and LR demapping processing
*/
typedef struct {
    int8_t *pLlr;                  /*!< Pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints) */
} bblib_layer_llr_demap_response;

//! @{
/*! \brief 5GNR PUSCH layer demap and LLR demap processing: Layer demapping+LLR demapping
    \param [in] request Input request structure for Layer demapping and LLR demapping
    \param [out] response Output response structure for Layer demapping and LLR demapping
    \return 0 for success, and -1 for error
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
int32_t
bblib_layer_llr_demap_processing(bblib_layer_llr_demap_request *request, bblib_layer_llr_demap_response *response);

int32_t
bblib_layer_llr_demap_processing_avx512(bblib_layer_llr_demap_request *request, bblib_layer_llr_demap_response* response);
//! @}


#ifdef __cplusplus
}
#endif

#endif
