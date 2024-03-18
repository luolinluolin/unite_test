/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_rx_mimo_mmse.h
    \brief  External API for MMSE MIMO detection, with post SINR calculation.

   __Overview:__
   This module implements MMSE MIMO detection in 5GNR, with post SINR calculation.
   It can support 1TX1R, 1TX2R, 1TX4R, 1TX8R, 2TX2R, 2TX4R, 2TX8R, 4TX4R, 4TX8R, 4TX16RX
   8TX8R, 8TX16RX, 16TX16RX antennas.

   __Algorithm Guidance:__
   1. Calculate weighting matrix W = inv(H' * H + Sigma2*I) * H',
      where H is channel transfer function in frequenchy domain among Tx and Rx antennas
      Sigma2 is noise power
   2. Multiply weighting matrix with input signal from Rx antennas: X = W * Y
      get estimated Tx signal.
   3. For post SINR, let gain = real(diag(inv(H'H+sigma2)*H'*H))
      post SINR = gain ./ (1-gain)
 */

#ifndef _PHY_RX_MIMO_MMSE_H_
#define _PHY_RX_MIMO_MMSE_H_

// #include "common_typedef_sdk.h"
#include "bblib_common_const.h"
#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


#define N_TX_2 2
#define N_TX_4 4
#define N_TX_8 8
#define N_TX_16 16

#define N_RX_2 2
#define N_RX_4 4
#define N_RX_8 8
#define N_RX_16 16

#define MAX_SC_TIME 205 //3276/16=205
#define MAX_SC 3276 
#define MMSE_MAX_DMRS_SYM 4 

#define BBLIB_INTERP_GRANS (5)  // Number of possible granularities for 2 DMRS CE symbols

/*!
    \struct bblib_mmse_mimo_request
    \brief Request struct of MMSE MIMO
*/
typedef struct {
    int32_t nLayer; /*!< Number of Layers - valid values 1, 2, 4, 8, 16 */
    int32_t nRxAnt;    /*!< Number of Rx antennas  - valid values 1, 2, 4, 8, 16 */
    void * pChState[BBLIB_MAX_RX_ANT_NUM][BBLIB_MAX_TX_LAYER_NUM]; /*!< Data pointer points to nRxAnt*nTxLayer channel, format 16S13*/
    void * pRxSignal[BBLIB_MAX_RX_ANT_NUM][BBLIB_N_SYMB_PER_SF]; /*!< Data pointer points to nRxAnt*nSymbol received data, format 16S13 */
    float  nSigma2; /*!< Noise power */
    int16_t nStartSC; /*!< Start Subcarrier */
    int16_t nSubCarrier; /*!< Number of granted subcarriers - valid range [1-3276] */
    int16_t nTotalAlignedSubCarrier; /*!< Number of Aligned Total Granted Subcarriers, Decide the Buffer Offset, Need to Be Mulitple of 16. */
    int16_t nChSymb; /*!< Number of Channel Symbols - valid range [1-(N_SYMB_PER_SF-1)] */
    int16_t nSymb; /*!< Number of granted Symbols - valid range [1-(N_SYMB_PER_SF-1)] */
    int16_t nSymbPerDmrs[BBLIB_N_SYMB_PER_SF]; /*!< Number of granted Symbols Per Dmrs  - valid range [1-(N_SYMB_PER_SF-1)] */
    int16_t *pSymbIndex; /*!< Pointer for Data Symbol index */
    int16_t *pFoCompScCp;/*!< FO table */

// For linear interpolation support
    int16_t nLinInterpEnable; /*!< 0 - stored DMRS CE is used directly (nearest neighbor), 1 - time linear interpolation is used */
    int16_t nDmrsChSymb; /*!< Number of Dmrs Symbols - valid range [1-4]. Defines how many CEs stored on input at this function call */
    int16_t nMappingType; /*!< Dmrs Type - A is 0 and B is 1 */
    int16_t nGranularity; /*!< Defines the number of adjacent symbols sharing the same CE inside the slot */

    // FOC support
    float    fEstCfo[BBLIB_MAX_TX_LAYER_NUM]; /*!< Normalized frequency offset estimates for each layer */
    int16_t nFftSize; /*!< FFT Size */
    int16_t nNumerology; /*!< Numerology */
    int16_t nEnableFoComp; /*!< Flag to enable frequency offset compensation */
} bblib_mmse_mimo_request;

/*!
    \struct bblib_mmse_mimo_response
    \brief Response struct of MMSE MIMO
*/
typedef struct {
    void * pEstTxSignal[BBLIB_MAX_TX_LAYER_NUM][BBLIB_N_SYMB_PER_SF]; /*!< Data pointer points to nTx*nSymbol estimated TX signal, format 16S13 */
    void * pPostSINR[BBLIB_MAX_TX_LAYER_NUM]; /*!< Pointer points to nTx*1 estimated post SINR, floating number */
} bblib_mmse_mimo_response;

/*! \brief Report the version number for the bblib_mimo_mmse_detection library
    \param [in] version Pointer to a char buffer where the version string should be copied.
    \param [in] buffer_size The length of the string buffer, must be at least
           BBLIB_SDK_VERSION_STRING_MAX_LEN characters.
    \return 0 if the version string was populated, otherwise -1.
*/
int16_t
bblib_mimo_mmse_detection_version(char *version, int buffer_size);

//! @{
/*! \brief MMSE MIMO detection, with post SNR calculation.
    \param [in] request Input request structure for MMSE MIMO.
    \param [out] response Output response structure for MMSE MIMO..
    \return 0 for success, and -1 for error
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
int32_t
bblib_mimo_mmse_detection(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response *response);

int32_t
bblib_mimo_mmse_detection_avx512(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response);

int32_t
bblib_mimo_mmse_detection_avx512_5gisa(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response* response);
//! @}

//! @{
/*! \brief matrix inverse for 4x4, using lemma method
    \param [in] ftempARe is the real part of the input matrix
    \param [in] ftempAIm is the imaginary part of the input matrix
    \param [in] nFixedBitsSquare is the square value of the decimal digits number for the fixed point input data
    \param [out] ftempInvARe is the real part of the inversed matrix
    \param [out] ftempInvAIm is the imaginary part of the inversed matrix
    \return 0 for success, and -1 for error
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
int32_t matrix_inv_lemma_4x4(__m512 ftempARe[4][4], __m512 ftempAIm[4][4],
    __m512 ftempInvARe[4][4], __m512 ftempInvAIm[4][4], int16_t nFixedBitsSquare);

//! @}

#ifdef __cplusplus
}
#endif

#endif
