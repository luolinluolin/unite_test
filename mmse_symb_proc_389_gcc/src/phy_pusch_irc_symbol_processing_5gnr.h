/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_pusch_irc_symbol_processing_5gnr.h
    \brief  External API for 5GNR PUSCH symbol processing.

   __Overview:__
   This module implements MMSE IRC MIMO equalization, layer demapping,and LLR dempping.
   MMSE IRC MIMO equalization supports 1x2, 2x2, 2x4 1x16 2x16 in SU case; 4x16 in 2UE MU case;
   8x16 in 4UE MU case.
   LLR demapping supports QPSK, 16QAM, 64QAM and 256QAM

   __Algorithm Guidance:__
   1. MMSE IRC MIMO equalization refers to lib_equalization
   3. Layer demapping refers to lib_layerdemapping_5gnr
   4. LLR demapping refers to the inline comments
 */

#ifndef _PHY_PUSCH_IRC_SYMBOL_PROCESSING_H_
#define _PHY_PUSCH_IRC_SYMBOL_PROCESSING_H_

#include "common_typedef_sdk.h"
#include "bblib_common_const.h"
#include "phy_pusch_symbol_processing_5gnr.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PUSCH_MAX_DMRS_SYMBOL       (4)
#define PUSCH_MAX_DMRS_PORT_NUM     (4)

/*!
    \enum MMSE_IRC_SUBMODULE_TYPE
    \brief MMSE_IRC submodules
*/
enum MMSE_IRC_SUBMODULE_TYPE {
    PUSCH_MMSE_IRC_LOAD_RNN = 0,
    PUSCH_MMSE_IRC_LOAD_H,
    PUSCH_MMSE_IRC_CALC_H,
    PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN,
    PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I,
    PUSCH_MMSE_IRC_MATRIX_INVERSE,
    PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN,
    PUSCH_MMSE_IRC_GAIN_SINR,
    PUSCH_MMSE_IRC_TX_CALC,
    PUSCH_MMSE_IRC_LLR_DEMAPER,
    PUSCH_MMSE_IRC_SUBMODULE_MAX
};

/*! \brief Report the version number for the bblib_pusch_irc_symbol_processing library
    \param [in] version Pointer to a char buffer where the version string should be copied.
    \param [in] buffer_size The length of the string buffer, must be at least
           BBLIB_SDK_VERSION_STRING_MAX_LEN characters.
    \return 0 if the version string was populated, otherwise -1.
*/
int16_t
bblib_pusch_irc_symbol_processing_version(char *version, int buffer_size);

//! @{
/*! \brief 5GNR PUSCH IRC symbol processing: MMSE IRC MIMO+Layer demapping+LLR demapping
    \param [in] request Input request structure for PUSCH symbol processing
    \param [out] response Output response structure for PUSCH symbol processing
    \return 0 for success, and -1 for error
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
int32_t
bblib_pusch_irc_symbol_processing(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response *response);

int32_t
bblib_pusch_irc_symbol_processing_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response);

int32_t
bblib_pusch_irc_symbol_processing_avx512_5gisa(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response);

void rnn_inverse_all(bblib_pusch_symbol_processing_request *request, size_t mode);

void rnn_inverse_all_5gisa(bblib_pusch_symbol_processing_request *request, size_t mode);
//! @}



#ifdef __cplusplus
}
#endif

#endif
