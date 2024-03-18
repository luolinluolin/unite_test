/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/*
 * @file   phy_tafo_table_gen.cpp
 * @brief  This file will generate ta/fo tables.
*/
#include <string.h>
#include "bblib_common.hpp"
#include "phy_tafo_table_gen.h"
#include "bblib_common_const.h"

#ifndef PI
#define PI ((float) 3.14159265358979323846)
#endif

void bblib_init_common_time_offset_tables(const struct bblib_ta_request *request, struct bblib_ta_response *response)
{
    uint32_t n, a, b;
    double tempTO, pi = PI;
    int16_t *pRealN, *pTemp;
    int16_t nTimeOffset;
    uint32_t nSeq;
    uint16_t nFft = request->n_fft_size;
    int32_t nCP = request->n_cp;
    uint32_t nSC = request->n_fullband_sc;

    pRealN = (int16_t *)malloc(nSC * sizeof(int16_t));
    pTemp = (int16_t *)malloc(nFft * 2 * sizeof(int16_t));

    if (NULL == pRealN || NULL == pTemp)
    {
        printf("\ninit_common_time_offset_tables: \n");
        printf("Not able to allocate pRealN or pTemp, size = %ld\n\n", nSC * sizeof(int16_t));
        exit(-1);
    }

    memset((void *)pRealN, 0, nSC * sizeof(int16_t));
    memset((void *)pTemp, 0, nFft * 2 * sizeof(int16_t));

    // calculate phase shift real and imagine part in 16S15 FXP
    for (n = 0; n < nFft; n++)
    {
        tempTO = (2 * pi * n) / nFft;
        pTemp[2 * n]     = (int16_t)(round(cos(tempTO) * 32767.0));
        pTemp[2 * n + 1] = (int16_t)(round(sin(tempTO) * 32767.0));
    }

    for (n = 0; n < nSC; n++)
    {
        if (n < (nSC / 2))
        {
            pRealN[n] = n + (nFft - nSC / 2);
        }
        else if ((n >= (nSC / 2)))
        {
            pRealN[n] = n - nSC / 2;
        }
    }

    nSeq = 0;
    for (nTimeOffset = -nCP; nTimeOffset <= nCP; nTimeOffset++)
    {
        for (n = 0; n < nSC; n++)
        {
            a = pRealN[n];
            b = (a*nTimeOffset) % nFft;
            response->pCeTaFftShiftScCp[2 * nSeq]     = pTemp[2 * b];
            response->pCeTaFftShiftScCp[2 * nSeq + 1] = pTemp[2 * b + 1];
            nSeq = nSeq + 1;
        }
    }

    free(pRealN);
    free(pTemp);

}

void bblib_init_common_frequency_compensation_tables(const struct bblib_fo_request *request, struct bblib_fo_response *response)
{

    int16_t nFft = request->n_fft_size;
    int16_t n;
    double tempTO = 0, pi = PI;

    int16_t cosine, sine;
    double factor = 2 * pi / nFft;

    // calculate phase shift real and imagine part in 16S15 FXP
    for (n = 0; n < nFft; n++)
    {
        cosine = (int16_t)(round(cos(tempTO) * 32767.0));
        sine = (int16_t)(round(sin(tempTO) * 32767.0));
        response->pFoCompScCp[2 * n] = cosine;
        response->pFoCompScCp[2 * n + 1] = sine;
        tempTO += factor;
    }
}

int16_t bblib_get_sys_params(const int16_t nMu, const int16_t nFftSize, int16_t *nMaxCp, int16_t *nMinCp, float *fSampleRate)
{
    int16_t nMaxCpTemp= 0, nMinCpTemp= 0;
    float fSampleRateTemp = 0;
    if (0 == nMu) // 0: 15khz
    {
        if ((N_FFT_SIZE_MU0_40MHZ_AND_50MHZ == nFftSize) || (0 == nFftSize))
        {
            nMaxCpTemp = N_MAX_CP_MU0_40MHZ_AND_50MHZ;
            nMinCpTemp = N_MIN_CP_MU0_40MHZ_AND_50MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU0_40MHZ_AND_50MHZ;
        }
        else if (N_FFT_SIZE_MU0_30MHZ_AND_35MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU0_30MHZ_AND_35MHZ;
            nMinCpTemp = N_MIN_CP_MU0_30MHZ_AND_35MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU0_30MHZ_AND_35MHZ;
        }
        else if (N_FFT_SIZE_MU0_20MHZ_AND_25MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU0_20MHZ_AND_25MHZ;
            nMinCpTemp = N_MIN_CP_MU0_20MHZ_AND_25MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU0_20MHZ_AND_25MHZ;
        }
        else if (N_FFT_SIZE_MU0_15MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU0_15MHZ;
            nMinCpTemp = N_MIN_CP_MU0_15MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU0_15MHZ;
        }
        else if (N_FFT_SIZE_MU0_10MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU0_10MHZ;
            nMinCpTemp = N_MIN_CP_MU0_10MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU0_10MHZ;

        }
        else if (N_FFT_SIZE_MU0_5MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU0_5MHZ;
            nMinCpTemp = N_MIN_CP_MU0_5MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU0_5MHZ;
        }
        else
        {
            //printf("Error! Currently not support this case\n");
            return -1;
        }
    }
    else if (1 == nMu) // 1: 30KHz
    {
        if ((N_FFT_SIZE_MU1_80MHZ_90MHZ_AND_100MHZ == nFftSize) || (0 == nFftSize))
        {
            nMaxCpTemp = N_MAX_CP_MU1_80MHZ_90MHZ_AND_100MHZ;
            nMinCpTemp = N_MIN_CP_MU1_80MHZ_90MHZ_AND_100MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_80MHZ_90MHZ_AND_100MHZ;
        }
        else if (N_FFT_SIZE_MU1_60MHZ_AND_70MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_60MHZ_AND_70MHZ;
            nMinCpTemp = N_MIN_CP_MU1_60MHZ_AND_70MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_60MHZ_AND_70MHZ;
        }
        else if (N_FFT_SIZE_MU1_40MHZ_45MHZ_AND_50MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_40MHZ_45MHZ_AND_50MHZ;
            nMinCpTemp = N_MIN_CP_MU1_40MHZ_45MHZ_AND_50MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_40MHZ_45MHZ_AND_50MHZ;
        }
        else if (N_FFT_SIZE_MU1_30MHZ_AND_35MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_30MHZ_AND_35MHZ;
            nMinCpTemp = N_MIN_CP_MU1_30MHZ_AND_35MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_30MHZ_AND_35MHZ;
        }
        else if (N_FFT_SIZE_MU1_20MHZ_AND_25MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_20MHZ_AND_25MHZ;
            nMinCpTemp = N_MIN_CP_MU1_20MHZ_AND_25MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_20MHZ_AND_25MHZ;
        }
        else if (N_FFT_SIZE_MU1_15MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_15MHZ;
            nMinCpTemp = N_MIN_CP_MU1_15MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_15MHZ;
        }
        else if (N_FFT_SIZE_MU1_10MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_10MHZ;
            nMinCpTemp = N_MIN_CP_MU1_10MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_10MHZ;
        }
        else if (N_FFT_SIZE_MU1_5MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU1_5MHZ;
            nMinCpTemp = N_MIN_CP_MU1_5MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU1_5MHZ;
        }
        else
        {
            return -1;
        }
    }
    else if (2 == nMu) // 2: 60KHz
    {
        if ((N_FFT_SIZE_MU2_80MHZ_90MHZ_AND_100MHZ == nFftSize) || (0 == nFftSize))
        {
            nMaxCpTemp = N_MAX_CP_MU2_80MHZ_90MHZ_AND_100MHZ;
            nMinCpTemp = N_MIN_CP_MU2_80MHZ_90MHZ_AND_100MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_80MHZ_90MHZ_AND_100MHZ;
        }
        else if (N_FFT_SIZE_MU2_60MHZ_AND_70MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU2_60MHZ_AND_70MHZ;
            nMinCpTemp = N_MIN_CP_MU2_60MHZ_AND_70MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_60MHZ_AND_70MHZ;
        }
        else if (N_FFT_SIZE_MU2_40MHZ_45MHZ_AND_50MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU2_40MHZ_45MHZ_AND_50MHZ;
            nMinCpTemp = N_MIN_CP_MU2_40MHZ_45MHZ_AND_50MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_40MHZ_45MHZ_AND_50MHZ;
        }
        else if (N_FFT_SIZE_MU2_30MHZ_AND_35MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU2_30MHZ_AND_35MHZ;
            nMinCpTemp = N_MIN_CP_MU2_30MHZ_AND_35MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_30MHZ_AND_35MHZ;
        }
        else if (N_FFT_SIZE_MU2_20MHZ_AND_25MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU2_20MHZ_AND_25MHZ;
            nMinCpTemp = N_MIN_CP_MU2_20MHZ_AND_25MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_20MHZ_AND_25MHZ;
        }
        else if (N_FFT_SIZE_MU2_15MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU2_15MHZ;
            nMinCpTemp = N_MIN_CP_MU2_15MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_15MHZ;
        }
        else if (N_FFT_SIZE_MU2_10MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU2_10MHZ;
            nMinCpTemp = N_MIN_CP_MU2_10MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU2_10MHZ;
        }
        else
        {
            return -1;
        }
    }
    else if (3 == nMu) // 3: 120KHz
    {
        if (N_FFT_SIZE_MU3_400MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU3_400MHZ;
            nMinCpTemp = N_MIN_CP_MU3_400MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU3_400MHZ;
        }
        else if (N_FFT_SIZE_MU3_200MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU3_200MHZ;
            nMinCpTemp = N_MIN_CP_MU3_200MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU3_200MHZ;
        }
        else if ((N_FFT_SIZE_MU3_100MHZ == nFftSize) || (0 == nFftSize))
        {
            nMaxCpTemp = N_MAX_CP_MU3_100MHZ;
            nMinCpTemp = N_MIN_CP_MU3_100MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU3_100MHZ;
        }
        else if (N_FFT_SIZE_MU3_50MHZ == nFftSize)
        {
            nMaxCpTemp = N_MAX_CP_MU3_50MHZ;
            nMinCpTemp = N_MIN_CP_MU3_50MHZ;
            fSampleRateTemp = N_SAMPLE_RATE_MU3_50MHZ;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        return -1;
    }

    if (nMaxCp != NULL)
        *nMaxCp = nMaxCpTemp;
    if (nMinCp != NULL)
        *nMinCp = nMinCpTemp;
    if (fSampleRate != NULL)
        *fSampleRate = fSampleRateTemp;

    return 0;
}

