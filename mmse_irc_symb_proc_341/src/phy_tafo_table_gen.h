/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/
/*! \file   phy_tafo_table_gen.h
    \brief  This file will generate ta/fo tables.
*/
#ifndef _TAFO_TABLE_GEN
#define _TAFO_TABLE_GEN


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef FO_LUT_SIZE
#define FO_LUT_SIZE (4096)
#endif

/*!
    \struct bblib_ta_request.
    \brief Request structure for ta table
*/
struct bblib_ta_request
{
    int16_t n_fft_size; /*!< FFT size */
    int32_t n_fullband_sc; /*!< Number of subcarriers infull band */
    int32_t n_cp; /*!< Number of CPs */
};

/*!
    \struct bblib_ta_response.
    \brief Response structure for ta table
*/
struct bblib_ta_response
{
    int16_t *pCeTaFftShiftScCp;/*!< TA table */
};


/*!
    \struct bblib_fo_request.
    \brief Request structure for fo table
*/
struct bblib_fo_request
{
    int16_t n_fft_size; /*!< FFT size */
};


/*!
    \struct bblib_fo_response.
    \brief Response structure for fo table
*/
struct bblib_fo_response
{
    int16_t *pFoCompScCp; /*!< FO table */
};

//! @{
/*! \brief ta/fo table generate procedures.
    \param [in] request Structure containing the input data which need to be 64 bytes alignment.
    \param [out] response Structure containing the compensated output data which need 64 byte alignment.
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
void bblib_init_common_time_offset_tables(const struct bblib_ta_request *request, struct bblib_ta_response *response);
void bblib_init_common_frequency_compensation_tables(const struct bblib_fo_request *request, struct bblib_fo_response *response);
//! @}

//! @{
/*! \brief system parameters generate procedures.
*/
int16_t bblib_get_sys_params(const int16_t nMu, const int16_t nFftSize, int16_t *nMaxCp, int16_t *nMinCp, float *fSampleRate);
//! @}

#ifdef __cplusplus
}
#endif

#endif