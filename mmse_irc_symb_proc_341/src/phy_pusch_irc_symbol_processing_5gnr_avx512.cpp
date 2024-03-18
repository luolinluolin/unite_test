/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*******************************************************************************
* @file phy_pusch_symbol_processing_5gnr_avx512.cpp
* @brief 5GNR PUSCH symbol processing.
*******************************************************************************/

/*******************************************************************************
* Include private header files
*******************************************************************************/
#include <map>
#include <tuple>
#include "bblib_common.hpp"
#include "mimo.hpp"
#include "phy_pusch_irc_symbol_processing_5gnr.h"
#include "phy_pusch_irc_symbol_processing_5gnr_internal.h"
#include "phy_tafo_table_gen.h"
//#include "phy_pusch_symbol_processing_5gnr_avx512.h"
#ifdef _BBLIB_AVX512_
// #define SAVE_GAIN (0)
using namespace W_SDK;

//using namespace W_SDK;

#ifndef PI
#define PI          (3.14159265358979323846)
#endif

// Original tables are in phy_rx_mimo_mmse_avx512.cpp
extern int16_t numgroups[BBLIB_INTERP_GRANS];
extern int16_t numingroup[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t symnumsA[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t symnumsB[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t g_flag_symH_upd[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t wType[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t wType_A[6][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t wType_B[4][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t numgroups_2p2_A[BBLIB_INTERP_GRANS];
extern int16_t numgroups_2p2_B[BBLIB_INTERP_GRANS];
extern int16_t numingroup_2p2_A[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t numingroup_2p2_B[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t symnums_2p2_A[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t symnums_2p2_B[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t flag_symH_upd_2p2[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t wType_2p2[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t g_flag_symH_upd_optA[6][BBLIB_INTERP_GRANS];
extern int16_t g_flag_symH_upd_optB[4][BBLIB_INTERP_GRANS];
extern int16_t flag_symH_upd_2p2_opt[2][BBLIB_INTERP_GRANS];
extern float FocPhaseFixTable[2][BBLIB_N_SYMB_PER_SF];
int16_t nDmrsIndex_AB[2][BBLIB_N_SYMB_PER_SF] = {{ 1,0,0,0,0,0,0,2,0,0,0,0,0,0 }, { 1,0,0,0,0,0,2,0,0,0,0,0,0,0 }};

//    15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
/*! \brief MMSE MIMO detection for 16TX16R, with post SINR calculation.
    \param [in] request Input request structure for MMSE MIMO.
    \param [out] response Output response structure for MMSE MIMO..
*/

#define ptr_cast reinterpret_cast

#ifdef _BBLIB_SPR_
template<typename T, size_t N_RX>
static inline void cal_inv_rnn(bblib_pusch_symbol_processing_request *request,
                        size_t nDmrsChSymb, T fRnnC[N_RX][N_RX], size_t rnnSc, size_t nRestLen, size_t avg2sym) {
    size_t nStartRBGIdx = request->nIrcRbgStart;
    size_t nBufferShuffle = request->nBufferShuffle;
    if (0 == avg2sym) // process each DMRS symbol
    {
        for (size_t iDmrsChSymb = 0; iDmrsChSymb < nDmrsChSymb; iDmrsChSymb ++) {
            for (size_t nSCIdx = nStartRBGIdx; nSCIdx < (rnnSc + nStartRBGIdx); nSCIdx += 16) {
                Mask16 scFlag = (nSCIdx + 16 > (rnnSc + nStartRBGIdx));
                Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
                // 1. calculate inv(Rnn)
                // 1.1 load Rnn (Rx * Rx) for every 48 sc = 4PRB.
                #pragma unroll(N_RX)
                for (size_t i = 0; i < N_RX; i++) {
                    auto pRnnIn_C = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][i]) + nSCIdx);
                    fRnnC[i][i] = loadu(pRnnIn_C);
                    for (size_t j = i + 1; j < N_RX; j ++) {
                        // the Rnn is stored in 32bit.
                        auto pRnnIn_C = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + nSCIdx);
                        fRnnC[i][j] = loadu(pRnnIn_C);
                        fRnnC[j][i] = negImag(fRnnC[i][j]);
                    }
                }
                matrix_inverse<T, N_RX>(fRnnC);

                if ((N_RX == 16)&&(nBufferShuffle == 0))
                {
                    auto nLen = 16;
                    if (scFlag > 0)
                        nLen = nRestLen;
                    for (size_t i = 0; i < N_RX; i++) {
                        for (size_t j = 0; j < N_RX; j ++) {
                            float *pRe = (float *)(&fRnnC[i][j]);
                            for (int16_t k = 0; k < nLen; k++ ) {
                                auto pRnnIn_C = ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][k]) + nSCIdx;
                                *(pRnnIn_C+j) = *(pRe+k);
                            }
                        }
                    }
                }
                else if ((N_RX == 8)&&(nBufferShuffle == 0))
                {
                    auto nLen = 8;
                    auto nLen1 = 8;
                    if (scFlag > 0)
                    {
                        if (nRestLen >= 8)
                        {
                            nLen1 = nRestLen - 8;
                        }
                        else
                        {
                            nLen = nRestLen;
                            nLen1 = 0;
                        }
                    }
                    for (size_t i = 0; i < N_RX; i++) {
                        for (size_t j = 0; j < N_RX; j ++) {
                            float *pRe = (float *)(&fRnnC[i][j]);

                            for (int16_t k = 0; k < nLen; k++ ) {
                                auto pRnnIn_C = ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][k]) + nSCIdx;
                                *(pRnnIn_C+j) = *(pRe+k);
                            }
                            for (int16_t k = 0; k < nLen1; k++ ) {
                                auto pRnnIn_C1 = (ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][k]) + nSCIdx) + 8;
                                *(pRnnIn_C1+j) = *(pRe+k+8);
                            }
                        }
                    }
                }
                else
                {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        for (size_t j = 0; j < N_RX; j ++) {
                            auto pRnnIn_C = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + nSCIdx);
                            storeu(pRnnIn_C, kMask, fRnnC[i][j]);
                        }
                    }
                }
            }
        }
    }
    else // average 2 DMRS symbols
    {
        for (size_t nSCIdx = nStartRBGIdx; nSCIdx < (rnnSc + nStartRBGIdx); nSCIdx += 16) {
            Mask16 scFlag = (nSCIdx + 16 > (rnnSc + nStartRBGIdx));
            Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
            // 1. calculate inv(Rnn)
            // 1.1 load Rnn (Rx * Rx) for every 48 sc = 4PRB.
            #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
            // j = i
                auto pRnnIn_C = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[0][i][i]) + nSCIdx);
                auto pRnnIn_C1 = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[1][i][i]) + nSCIdx);
                auto zmmtemp = loadu(pRnnIn_C);
                auto zmmtemp1 = loadu(pRnnIn_C1);
                zmmtemp = _mm512_add_ph(zmmtemp, zmmtemp1);
                fRnnC[i][i] = _mm512_mul_ph(zmmtemp, _mm512_set1_ph((float16)0.5));
                for (size_t j = i + 1; j < N_RX; j ++) {
                    // the Rnn is stored in 32bit.
                    auto pRnnIn_C = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[0][i][j]) + nSCIdx);
                    auto pRnnIn_C1 = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[1][i][j]) + nSCIdx);
                    auto zmmtemp = loadu(pRnnIn_C);
                    auto zmmtemp1 = loadu(pRnnIn_C1);
                    zmmtemp = _mm512_add_ph(zmmtemp, zmmtemp1);
                    fRnnC[i][j] = _mm512_mul_ph(zmmtemp, _mm512_set1_ph((float16)0.5));
                    fRnnC[j][i] = negImag(fRnnC[i][j]);
                }
            }
            matrix_inverse<T, N_RX>(fRnnC);

            if ((N_RX == 16)&&(nBufferShuffle == 0))
            {
                auto nLen = 16;
                if (scFlag > 0)
                    nLen = nRestLen;
                for (size_t i = 0; i < N_RX; i++) {
                     for (size_t j = 0; j < N_RX; j ++) {
                        float *pRe = (float *)(&fRnnC[i][j]);
                        for (int16_t k = 0; k < nLen; k++ ) {
                            auto pRnnIn_C = ptr_cast<float *>(request->pRnn_Re[0][i][k]) + nSCIdx;
                            *(pRnnIn_C+j) = *(pRe+k);
                        }
                    }
                }
            }
            else if ((N_RX == 8)&&(nBufferShuffle == 0))
            {
                auto nLen = 8;
                auto nLen1 = 8;
                if (scFlag > 0)
                {
                    if (nRestLen >= 8)
                    {
                        nLen1 = nRestLen - 8;
                    }
                    else
                    {
                        nLen = nRestLen;
                        nLen1 = 0;
                    }
                }
                for (size_t i = 0; i < N_RX; i++) {
                    for (size_t j = 0; j < N_RX; j ++) {
                        float *pRe = (float *)(&fRnnC[i][j]);
                        for (int16_t k = 0; k < nLen; k++ ) {
                            auto pRnnIn_C = ptr_cast<float *>(request->pRnn_Re[0][i][k]) + nSCIdx;
                            *(pRnnIn_C+j) = *(pRe+k);
                        }
                        for (int16_t k = 0; k < nLen1; k++ ) {
                            auto pRnnIn_C1 = (ptr_cast<float *>(request->pRnn_Re[0][i][k]) + nSCIdx) + 8;
                            *(pRnnIn_C1+j) = *(pRe+k+8);
                        }
                    }
                }
            }
            else
            {
                #pragma unroll(N_RX)
                for (size_t i = 0; i < N_RX; i++) {
                    for (size_t j = 0; j < N_RX; j ++) {
                        auto pRnnIn_C = ptr_cast<T *>(ptr_cast<float *>(request->pRnn_Re[0][i][j]) + nSCIdx);
                        storeu(pRnnIn_C, kMask, fRnnC[i][j]);
                    }
                }
            }
        }
    }
}
#endif

template<typename T, size_t N_RX>
static inline void cal_inv_rnn(bblib_pusch_symbol_processing_request *request,
                        size_t nDmrsChSymb, T fRnnRe[N_RX][N_RX], T fRnnIm[N_RX][N_RX],
                        T finvRnnRe[N_RX][N_RX], T finvRnnIm[N_RX][N_RX], size_t rnnSc, size_t nRestLen, size_t avg2sym) {
    size_t nStartRBGIdx = request->nIrcRbgStart;
    size_t nBufferShuffle = request->nBufferShuffle;
    if (0 == avg2sym) // process each DMRS symbol
    {
        // #pragma unroll(N_RX)
        for (size_t iDmrsChSymb = 0; iDmrsChSymb < nDmrsChSymb; iDmrsChSymb ++) {
            for (size_t nSCIdx = nStartRBGIdx; nSCIdx < (rnnSc + nStartRBGIdx); nSCIdx += 16) {
                Mask16 scFlag = (nSCIdx + 16 > (rnnSc + nStartRBGIdx));
                Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
                for (size_t i = 0; i < N_RX; i++) {
                    auto pRnnIn_Re = (int32_t *)request->pRnn_Re[iDmrsChSymb][i][i] + nSCIdx;
                    auto pRnnIn_Im = (int32_t *)request->pRnn_Im[iDmrsChSymb][i][i] + nSCIdx;

                    auto zmmtemp1 = _mm512_loadu_si512(pRnnIn_Re);
                    auto zmmtemp2 = _mm512_loadu_si512(pRnnIn_Im);
                    fRnnRe[i][i] = _mm512_cvtepi32_ps(zmmtemp1);
                    fRnnIm[i][i] = _mm512_cvtepi32_ps(zmmtemp2);
                    //#pragma unroll(N_RX)
                    for (size_t j = i+1; j < N_RX; j ++) {
                        auto pRnnIn_Re = ptr_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + nSCIdx;
                        auto pRnnIn_Im = ptr_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][j]) + nSCIdx;

                        auto zmmtemp1 = _mm512_loadu_si512(pRnnIn_Re);
                        auto zmmtemp2 = _mm512_loadu_si512(pRnnIn_Im);

                        fRnnRe[i][j] = _mm512_cvtepi32_ps(zmmtemp1);
                        fRnnIm[i][j] = _mm512_cvtepi32_ps(zmmtemp2);

                        fRnnRe[j][i] = fRnnRe[i][j];
                        fRnnIm[j][i] = F32vec16(0.0) - fRnnIm[i][j];
                    }
                }
                matrix_inverse<N_RX>(fRnnRe, fRnnIm, finvRnnRe, finvRnnIm);

                if ((N_RX == 16)&&(nBufferShuffle == 0))
                {
                    auto nLen = 16;
                    if (scFlag > 0)
                        nLen = nRestLen;
                    for (size_t i = 0; i < N_RX; i++) {
                        for (size_t j = 0; j < N_RX; j ++) {
                            float *pRe = (float *)(&finvRnnRe[i][j]);
                            float *pIm = (float *)(&finvRnnIm[i][j]);

                            for (int16_t k = 0; k < nLen; k++ ) {
                                auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][k]) + nSCIdx;
                                auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][k]) + nSCIdx;

                                *(pRnnIn_Re+j) = *(pRe+k);
                                *(pRnnIn_Im+j) = *(pIm+k);
                            }
                        }
                    }
                }
                else if ((N_RX == 8)&&(nBufferShuffle == 0))
                {
                    auto nLen = 8;
                    auto nLen1 = 8;
                    if (scFlag > 0)
                    {
                        if (nRestLen >= 8)
                        {
                            nLen1 = nRestLen - 8;
                        }
                        else
                        {
                            nLen = nRestLen;
                            nLen1 = 0;
                        }
                    }
                    for (size_t i = 0; i < N_RX; i++) {
                        for (size_t j = 0; j < N_RX; j ++) {
                            float *pRe = (float *)(&finvRnnRe[i][j]);
                            float *pIm = (float *)(&finvRnnIm[i][j]);

                            for (int16_t k = 0; k < nLen; k++ ) {
                                auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][k]) + nSCIdx;
                                auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][k]) + nSCIdx;

                                *(pRnnIn_Re+j) = *(pRe+k);
                                *(pRnnIn_Im+j) = *(pIm+k);
                            }

                            for (int16_t k = 0; k < nLen1; k++ ) {
                                auto pRnnIn_Re1 = reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][k]) + nSCIdx + 8;
                                auto pRnnIn_Im1 = reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][k]) + nSCIdx + 8;

                                *(pRnnIn_Re1+j) = *(pRe+k+8);
                                *(pRnnIn_Im1+j) = *(pIm+k+8);
                            }
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < N_RX; i++) {
                        for (size_t j = 0; j < N_RX; j ++) {
                            auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + nSCIdx;
                            auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][j]) + nSCIdx;
                            storeu( pRnnIn_Re, kMask, finvRnnRe[i][j]);
                            storeu( pRnnIn_Im, kMask, finvRnnIm[i][j]);
                        }
                    }
                }
            }
        }
    }
    else // average 2 DMRS symbols
    {
        for (size_t nSCIdx = nStartRBGIdx; nSCIdx < (rnnSc + nStartRBGIdx); nSCIdx += 16) {
            Mask16 scFlag = (nSCIdx + 16 > (rnnSc + nStartRBGIdx));
            Mask16 kMask = scFlag * ((static_cast<Mask16>(1) << nRestLen) - 1) + (1 - scFlag) * 0xffff;
            // 1. calculate inv(Rnn)
            // 1.1 load Rnn (Rx * Rx) for every 48 sc = 4PRB.
            for (size_t i = 0; i < N_RX; i++) {
                auto pRnnIn_Re = (int32_t *)request->pRnn_Re[0][i][i] + nSCIdx;
                auto pRnnIn_Im = (int32_t *)request->pRnn_Im[0][i][i] + nSCIdx;
                auto pRnnIn_Re1 = (int32_t *)request->pRnn_Re[1][i][i] + nSCIdx;
                auto pRnnIn_Im1 = (int32_t *)request->pRnn_Im[1][i][i] + nSCIdx;

                auto zmmtemp1 = _mm512_loadu_si512(pRnnIn_Re);
                auto zmmtemp2 = _mm512_loadu_si512(pRnnIn_Im);
                auto zmmtemp3 = _mm512_loadu_si512(pRnnIn_Re1);
                auto zmmtemp4 = _mm512_loadu_si512(pRnnIn_Im1);

                // averaging for 2 DMRS symbols
                zmmtemp1 = _mm512_add_epi32(zmmtemp1, zmmtemp3);
                zmmtemp2 = _mm512_add_epi32(zmmtemp2, zmmtemp4);
                zmmtemp1 = _mm512_srai_epi32(zmmtemp1, 1);
                zmmtemp2 = _mm512_srai_epi32(zmmtemp2, 1);

                fRnnRe[i][i] = _mm512_cvtepi32_ps(zmmtemp1);
                fRnnIm[i][i] = _mm512_cvtepi32_ps(zmmtemp2);

                for (size_t j = i + 1; j < N_RX; j ++) {
                    // the Rnn is stored in 32bit.
                    auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[0][i][j]) + nSCIdx;
                    auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[0][i][j]) + nSCIdx;
                    auto pRnnIn_Re1 = reinterpret_cast<float *>(request->pRnn_Re[1][i][j]) + nSCIdx;
                    auto pRnnIn_Im1 = reinterpret_cast<float *>(request->pRnn_Im[1][i][j]) + nSCIdx;

                    auto zmmtemp1 = _mm512_loadu_si512(pRnnIn_Re);
                    auto zmmtemp2 = _mm512_loadu_si512(pRnnIn_Im);
                    auto zmmtemp3 = _mm512_loadu_si512(pRnnIn_Re1);
                    auto zmmtemp4 = _mm512_loadu_si512(pRnnIn_Im1);

                    // averaging for 2 DMRS symbols
                    zmmtemp1 = _mm512_add_epi32(zmmtemp1, zmmtemp3);
                    zmmtemp2 = _mm512_add_epi32(zmmtemp2, zmmtemp4);
                    zmmtemp1 = _mm512_srai_epi32(zmmtemp1, 1);
                    zmmtemp2 = _mm512_srai_epi32(zmmtemp2, 1);

                    fRnnRe[i][j] = _mm512_cvtepi32_ps(zmmtemp1);
                    fRnnIm[i][j] = _mm512_cvtepi32_ps(zmmtemp2);

                    fRnnRe[j][i] = fRnnRe[i][j];
                    fRnnIm[j][i] = F32vec16(0.0) - fRnnIm[i][j];
                }
            }
            // 1.2 inv(Rnn)
            matrix_inverse<N_RX>(fRnnRe, fRnnIm, finvRnnRe, finvRnnIm);
            // 1.3 store the inv(Rnn) in the same buffer
            if ((N_RX == 16)&&(nBufferShuffle == 0))
            {
                auto nLen = 16;
                if (scFlag > 0)
                    nLen = nRestLen;
                for (size_t i = 0; i < N_RX; i++) {
                    for (size_t j = 0; j < N_RX; j ++) {
                        float *pRe = (float *)(&finvRnnRe[i][j]);
                        float *pIm = (float *)(&finvRnnIm[i][j]);

                        for (int16_t k = 0; k < nLen; k++ ) {
                            auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[0][i][k]) + nSCIdx;
                            auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[0][i][k]) + nSCIdx;

                            *(pRnnIn_Re+j) = *(pRe+k);
                            *(pRnnIn_Im+j) = *(pIm+k);
                        }
                    }
                }
            }
            else if ((N_RX == 8)&&(nBufferShuffle == 0))
            {
                auto nLen = 8;
                auto nLen1 = 8;
                if (scFlag > 0)
                {
                    if (nRestLen >= 8)
                    {
                        nLen1 = nRestLen - 8;
                    }
                    else
                    {
                        nLen = nRestLen;
                        nLen1 = 0;
                    }
                }
                for (size_t i = 0; i < N_RX; i++) {
                    for (size_t j = 0; j < N_RX; j ++) {
                        float *pRe = (float *)(&finvRnnRe[i][j]);
                        float *pIm = (float *)(&finvRnnIm[i][j]);

                        for (int16_t k = 0; k < nLen; k++ ) {
                            auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[0][i][k]) + nSCIdx;
                            auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[0][i][k]) + nSCIdx;

                            *(pRnnIn_Re+j) = *(pRe+k);
                            *(pRnnIn_Im+j) = *(pIm+k);
                        }

                        for (int16_t k = 0; k < nLen1; k++ ) {
                            auto pRnnIn_Re1 = reinterpret_cast<float *>(request->pRnn_Re[0][i][k]) + nSCIdx + 8;
                            auto pRnnIn_Im1 = reinterpret_cast<float *>(request->pRnn_Im[0][i][k]) + nSCIdx + 8;

                            *(pRnnIn_Re1+j) = *(pRe+k+8);
                            *(pRnnIn_Im1+j) = *(pIm+k+8);
                        }
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < N_RX; i++) {
                    for (size_t j = 0; j < N_RX; j ++) {
                        auto pRnnIn_Re = reinterpret_cast<float *>(request->pRnn_Re[0][i][j]) + nSCIdx;
                        auto pRnnIn_Im = reinterpret_cast<float *>(request->pRnn_Im[0][i][j]) + nSCIdx;
                        storeu( pRnnIn_Re, kMask, finvRnnRe[i][j]);
                        storeu( pRnnIn_Im, kMask, finvRnnIm[i][j]);
                    }
                }
            }
        }
    }
}

template<typename T = CI16vec16, size_t N_RX = 16>
void rnn_inverse(bblib_pusch_symbol_processing_request *request,  size_t avg2sym) {

    using FloatSimd = typename DataType<T>::FloatSimd;
    const static auto fp16Int16 = DataType<T>::fp16Int16;
#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
    FloatSimd fRnnRe[N_RX][N_RX];
#endif
#if  defined (_BBLIB_AVX512_)
    FloatSimd fRnnIm[N_RX][N_RX];
    FloatSimd finvRnnRe[N_RX][N_RX], finvRnnIm[N_RX][N_RX];
#endif
    size_t nDmrsChSymb = request->nChSymb;
    size_t nSubCarrier = request->nSubCarrier;

    size_t rnnSc = (nSubCarrier + 47) / 48;
    auto nRestLen = rnnSc & 0xf;

    if constexpr (fp16Int16 == FP16_E::INT16){
        cal_inv_rnn<FloatSimd, N_RX>(request, nDmrsChSymb, fRnnRe, fRnnIm, finvRnnRe, finvRnnIm, rnnSc, nRestLen, avg2sym);
    }
#if defined (_BBLIB_SPR_)
    else
    {
        cal_inv_rnn<FloatSimd, N_RX>(request, nDmrsChSymb, fRnnRe, rnnSc, nRestLen, avg2sym);
    }
#endif
}

#ifdef _BBLIB_SPR_
void rnn_inverse_all_5gisa(bblib_pusch_symbol_processing_request *request, size_t avg2sym) {
    uint16_t nRxAnt = request->nRxAnt;
    if (2 == nRxAnt)
        rnn_inverse<CF16vec16, 2>(request, avg2sym);
    else if (4 == nRxAnt)
        rnn_inverse<CF16vec16, 4>(request, avg2sym);
    else if (8 == nRxAnt)
        rnn_inverse<CF16vec16, 8>(request, avg2sym);
    else if (16 == nRxAnt)
        rnn_inverse<CF16vec16, 16>(request, avg2sym);
    else
        printf("Error, currently doesn't support nRxAnt = %d !\n", nRxAnt);
}
#endif

void rnn_inverse_all(bblib_pusch_symbol_processing_request *request, size_t avg2sym) {
    uint16_t nRxAnt = request->nRxAnt;
    if (2 == nRxAnt)
        rnn_inverse<CI16vec16, 2>(request, avg2sym);
    else if (4 == nRxAnt)
        rnn_inverse<CI16vec16, 4>(request, avg2sym);
    else if (8 == nRxAnt)
        rnn_inverse<CI16vec16, 8>(request, avg2sym);
    else if (16 == nRxAnt)
        rnn_inverse<CI16vec16, 16>(request, avg2sym);
    else
        printf("Error, currently doesn't support nRxAnt = %d !\n", nRxAnt);
}

// mmse llr mimo linear interpolation
template<typename T, FO_E fo_flag, size_t N_RX = 16, size_t N_TX = 16, uint8_t DMRSTYPE = 1, uint8_t NROFCDMS = 2, INTERP_E interp_flag = INTERP_E::disable>
struct IRCSymbolProcess {
using FloatSimd = typename DataType<T>::FloatSimd;
using Float = typename DataType<T>::Float;
using procDataType = typename DataType<T>::procDataType;
using invType = typename DataType<T>::invType;
const static auto fp16Int16 = DataType<T>::fp16Int16;

#ifdef _BBLIB_SPR_
    static void load_rnn(bblib_pusch_symbol_processing_request *request,
                            size_t rnnIdx, size_t iDmrsChSymb, invType finvRnnC[N_RX][N_RX]) {
        size_t nBufferShuffle = request->nBufferShuffle;
        if ((N_RX == 16 || N_RX == 8)&&(nBufferShuffle == 0))
        {
            auto nStart = (rnnIdx / N_RX) * N_RX;
            auto nIndex = rnnIdx % N_RX;
            auto nLength = N_RX * 4;
            #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                auto *pRe = reinterpret_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][nIndex]) + nStart;
                memcpy((void *)finvRnnC[i], pRe, nLength);
            }
        }
        else{
            // #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                #pragma unroll(N_RX)
                for (size_t j = 0; j < N_RX; j ++) {
                    finvRnnC[i][j] = *(ptr_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][j]) + rnnIdx);
                }
            }
        }
    }

    static void load_rnn2(bblib_pusch_symbol_processing_request *request,
                            size_t rnnIdx, size_t iDmrsChSymb, invType finvRnnC[N_RX][N_RX], invType finvRnnC1[N_RX][N_RX]) {
        size_t nBufferShuffle = request->nBufferShuffle;
        if ((N_RX == 16 || N_RX == 8)&&(nBufferShuffle == 0))
        {
            auto nStart = (rnnIdx / N_RX) * N_RX;
            auto nIndex = rnnIdx % N_RX;
            auto nLength = N_RX * 4;

            auto nStart1 = ((rnnIdx + 1) / N_RX) * N_RX;
            auto nIndex1 = (rnnIdx + 1) % N_RX;
            for (size_t i = 0; i < N_RX; i++) {
                auto *pRe = reinterpret_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][nIndex]) + nStart;
                memcpy((void *)finvRnnC[i], pRe, nLength);
                auto *pRe1 = reinterpret_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][nIndex1]) + nStart1;
                memcpy((void *)finvRnnC1[i], pRe1, nLength);
            }
        }
        else
        {
            // #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                #pragma unroll(N_RX)
                for (size_t j = 0; j < N_RX; j ++) {
                    finvRnnC[i][j] = *(ptr_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][j]) + rnnIdx);
                    finvRnnC1[i][j] = *((ptr_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][j]) + rnnIdx) + 1);
                }
            }
        }
    }
#endif

    static void load_rnn(bblib_pusch_symbol_processing_request *request,
                            size_t rnnIdx, size_t iDmrsChSymb, invType finvRnnRe[N_RX][N_RX], invType finvRnnIm[N_RX][N_RX]) {
        size_t nBufferShuffle = request->nBufferShuffle;
        if ((N_RX == 16 || N_RX == 8)&&(nBufferShuffle == 0))
        {
            auto nStart = (rnnIdx / N_RX) * N_RX;
            auto nIndex = rnnIdx % N_RX;
            auto nLength = N_RX * 4;
            for (size_t i = 0; i < N_RX; i++) {
                auto *pRe = reinterpret_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][nIndex]) + nStart;
                auto *pIm = reinterpret_cast<invType *>(request->pRnn_Im[iDmrsChSymb][i][nIndex]) + nStart;
                memcpy((void *)finvRnnRe[i], pRe, nLength);
                memcpy((void *)finvRnnIm[i], pIm, nLength);
            }
        }
        else{
            //#pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                finvRnnRe[i][i] = *(reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][i]) + rnnIdx);
                finvRnnIm[i][i] = 0.0;
                //#pragma unroll(2)
                for (size_t j = i+1; j < N_RX; j ++) {
                    finvRnnRe[i][j] = *(reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + rnnIdx);
                    finvRnnIm[i][j] = *(reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][j]) + rnnIdx);
                    finvRnnRe[j][i] = finvRnnRe[i][j];
                    finvRnnIm[j][i] = 0.0 - finvRnnIm[i][j];
                }
            }
        }
    }

    static void load_rnn2(bblib_pusch_symbol_processing_request *request,
                            size_t rnnIdx, size_t iDmrsChSymb, invType finvRnnRe[N_RX][N_RX], invType finvRnnIm[N_RX][N_RX],
                            invType finvRnnRe2[N_RX][N_RX], invType finvRnnIm2[N_RX][N_RX]) {
        size_t nBufferShuffle = request->nBufferShuffle;
        if ((N_RX == 16 || N_RX == 8)&&(nBufferShuffle == 0))
        {
            auto nStart = (rnnIdx / N_RX) * N_RX;
            auto nIndex = rnnIdx % N_RX;
            auto nLength = N_RX * 4;

            auto nStart1 = ((rnnIdx + 1) / N_RX) * N_RX;
            auto nIndex1 = (rnnIdx + 1) % N_RX;
            for (size_t i = 0; i < N_RX; i++) {
                auto *pRe = reinterpret_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][nIndex]) + nStart;
                auto *pIm = reinterpret_cast<invType *>(request->pRnn_Im[iDmrsChSymb][i][nIndex]) + nStart;
                memcpy((void *)finvRnnRe[i], pRe, nLength);
                memcpy((void *)finvRnnIm[i], pIm, nLength);

                auto *pRe1 = reinterpret_cast<invType *>(request->pRnn_Re[iDmrsChSymb][i][nIndex1]) + nStart1;
                auto *pIm1 = reinterpret_cast<invType *>(request->pRnn_Im[iDmrsChSymb][i][nIndex1]) + nStart1;
                memcpy((void *)finvRnnRe2[i], pRe1, nLength);
                memcpy((void *)finvRnnIm2[i], pIm1, nLength);
            }
        }
        else{
            //#pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                finvRnnRe[i][i] = *(reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][i]) + rnnIdx);
                finvRnnRe2[i][i] = *((reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][i]) + rnnIdx) + 1);
                finvRnnIm[i][i] = 0.0;
                finvRnnIm2[i][i] = 0.0;
                //#pragma unroll(N_RX)
                for (size_t j = i+1; j < N_RX; j ++) {
                    finvRnnRe[i][j] = *(reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + rnnIdx);
                    finvRnnIm[i][j] = *(reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][j]) + rnnIdx);
                    finvRnnRe[j][i] = finvRnnRe[i][j];
                    finvRnnIm[j][i] = 0.0 - finvRnnIm[i][j];
                    finvRnnRe2[i][j] = *((reinterpret_cast<float *>(request->pRnn_Re[iDmrsChSymb][i][j]) + rnnIdx)+ 1);
                    finvRnnIm2[i][j] = *((reinterpret_cast<float *>(request->pRnn_Im[iDmrsChSymb][i][j]) + rnnIdx)+ 1);
                    finvRnnRe2[j][i] = finvRnnRe2[i][j];
                    finvRnnIm2[j][i] = 0.0 - finvRnnIm2[i][j];
                }
            }
        }
    }


    static void data_dmrs_mux(bblib_pusch_symbol_processing_request *request, uint16_t &dataDmrsMux, uint16_t &nDmrsPortIdx) {
        uint16_t nTotalDmrsSymbol = request->nTotalDmrsSymbol;
        uint16_t * pDmrsSymbolIdx = request->pDmrsSymbolIdx;
        uint16_t nDataDmrsInter = 0;
        uint8_t  * pDmrsPortIdx = request->pDmrsPortIdx[0];
        int16_t nMappingType = request->nMappingType;
        //Check if data/dmrs are interleaved for this group
        uint8_t nNrOfCDMs = request->nNrOfCDMs;
        uint8_t nDMRSType = request->nDMRSType;
        if (((nDMRSType == 1) && (nNrOfCDMs == 2)) || ((nDMRSType == 2) && (nNrOfCDMs == 3))) {
            //if no data/dmrs interleaving, no need to scan for dmrs symbols
            nDataDmrsInter = 0;
            nTotalDmrsSymbol = 0;
        } else {
            if(nMappingType == 1){//typeB
                printf("\n TypeB does not support data interleaving!\n");
            }
            nDataDmrsInter = 1;
        }
        if (nTotalDmrsSymbol > PUSCH_MAX_DMRS_PORT_NUM) {
            nTotalDmrsSymbol = PUSCH_MAX_DMRS_PORT_NUM;
        }

        if (nDataDmrsInter == 1) {
            for (size_t nIdx = 0; nIdx < nTotalDmrsSymbol; nIdx++) {
                //if this symbol is dmrs, set flag
                const auto nDataSymbIdx = pDmrsSymbolIdx[nIdx];
                dataDmrsMux += (uint16_t)1 << nDataSymbIdx;
            }
            nDmrsPortIdx = pDmrsPortIdx[0];
        }
    }


    static void init_factor(bblib_pusch_symbol_processing_request *request,
                    FloatSimd &avxShift, FloatSimd &llr_postsnr_fxp_dynamic, int16_t &llr_range_high, int16_t &llr_range_low) {

        const auto mmse_x_left = (Float)(1 << BBLIB_MMSE_X_LEFT_SHIFT);
        llr_range_high = (1 << (request->nLlrSaturatedBits - 1)) - 1;
        llr_range_low = -1 * llr_range_high;

        const auto left_shift = mmse_x_left;
        if constexpr (fp16Int16 == FP16_E::INT16){
            avxShift = FloatSimd(left_shift);
            llr_postsnr_fxp_dynamic =_mm512_set1_ps((float)(1<<(POSTSNR_FXP_BASE+request->nLlrFxpPoints)));
        }
#if defined (_BBLIB_SPR_)
        else
        {
            avxShift = FloatSimd(left_shift, 0.0);
            llr_postsnr_fxp_dynamic = _mm512_set1_ph((float)(1<<(POSTSNR_FXP_BASE+request->nLlrFxpPoints)));
        }
#endif
    }

    template <size_t POST_SINR_FLAG = 0>
    inline FORCE_INLINE
    static void gain_sinr(F32vec16 finvARe[N_TX][N_TX], F32vec16 ftempARe[N_TX][N_TX], F32vec16 finvAIm[N_TX][N_TX],
                            F32vec16 ftempAIm[N_TX][N_TX], F32vec16 ftempGain[N_TX], F32vec16 ftempPostSINR[N_TX], F32vec16 fsumPostSINR[N_TX],
                            F32vec16 llr_postsnr_fxp_dynamic){

        for (size_t i = 0; i < N_TX; i++) {
            // calculate the gain = real(diag(inv(newH'* H + I)*newH' * H)) = real(diag(invA * A))
            auto tempGain = gainCalc (i, finvAIm, finvARe, ftempAIm, ftempARe);
            auto temp = rcp(F32vec16(1.0) - tempGain);
            if constexpr (POST_SINR_FLAG == 1) {
                fsumPostSINR[i] += _mm512_mul_ps(tempGain, temp);
            }
            ftempGain[i] = _mm512_mul_ps(tempGain,llr_beta_fxp);
            ftempPostSINR[i] =  temp * llr_postsnr_fxp_dynamic;
        }
    }

    template <size_t POST_SINR_FLAG = 0>
    inline FORCE_INLINE
    static void gain_sinr2(F32vec16 finvARe[N_TX][N_TX], F32vec16 ftempARe[N_TX][N_TX], F32vec16 finvAIm[N_TX][N_TX],
                            F32vec16 ftempAIm[N_TX][N_TX], F32vec16 ftempGain[N_TX], F32vec16 ftempGain2[N_TX], F32vec16 ftempPostSINR[N_TX],
                            F32vec16 ftempPostSINR2[N_TX], F32vec16 fsumPostSINR[N_TX], F32vec16 llr_postsnr_fxp_dynamic){

        for (size_t i = 0; i < N_TX; i++) {
            // calculate the gain = real(diag(inv(newH'* H + I)*newH' * H)) = real(diag(invA * A))
            auto tempGain = gainCalc (i, finvAIm, finvARe, ftempAIm, ftempARe);
            auto temp = rcp(F32vec16(1.0) - tempGain);
            if constexpr (POST_SINR_FLAG == 1) {
                fsumPostSINR[i] += _mm512_mul_ps(tempGain, temp);
            }
            ftempGain[i] = _mm512_mul_ps(tempGain,llr_beta_fxp);
            ftempPostSINR[i] =  temp * llr_postsnr_fxp_dynamic;

            ftempGain2[i]  = _mm512_permutex2var_epi32(ftempGain[i], use_2nd_half, ftempGain[i]);
            ftempGain[i] = _mm512_permutex2var_epi32(ftempGain[i], use_1st_half, ftempGain[i]);
            ftempPostSINR2[i] = _mm512_permutex2var_epi32(ftempPostSINR[i], use_2nd_half, ftempPostSINR[i]);
            ftempPostSINR[i] = _mm512_permutex2var_epi32(ftempPostSINR[i], use_1st_half, ftempPostSINR[i]);
        }
    }

#ifdef _BBLIB_SPR_
    template <size_t POST_SINR_FLAG = 0>
    inline FORCE_INLINE
    static void gain_sinr(CF16vec16 finvARe[N_TX][N_TX], CF16vec16 ftempARe[N_TX][N_TX], CF16vec16 ftempGain[N_TX], CF16vec16 ftempPostSINR[N_TX],
                        F32vec16 fsumPostSINR[N_TX], CF16vec16 llr_postsnr_fxp_dynamic){
        #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i++) {
            // GAIN_SINR_5GISA
            auto gain = gainCalc (i, finvARe, ftempARe);
            gain =  duplicateReal(gain);
            CF16vec16 postSinr = _mm512_rcp_ph(value_one - gain);
            if constexpr (POST_SINR_FLAG == 1) {
                CF16vec16 postSinrOut = _mm512_mul_ph(gain, postSinr);
                fsumPostSINR[i] += _mm512_cvtph_ps(postSinrOut.real());
            }
            gain = _mm512_mul_ph(gain, llr_beta_fxp_fp16);
            ftempGain[i] = min(gain, max_value);
            postSinr = _mm512_mul_ph (postSinr, llr_postsnr_fxp_dynamic);
            ftempPostSINR[i] = min(postSinr, max_value);
        }
    }

    template <size_t POST_SINR_FLAG = 0>
    inline FORCE_INLINE
    static void gain_sinr2(CF16vec16 finvARe[N_TX][N_TX], CF16vec16 ftempARe[N_TX][N_TX], CF16vec16 ftempGain[N_TX], CF16vec16 ftempGain2[N_TX], CF16vec16 ftempPostSINR[N_TX],
                                CF16vec16 ftempPostSINR2[N_TX], F32vec16 fsumPostSINR[N_TX], CF16vec16 llr_postsnr_fxp_dynamic){
        #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i++) {
            // GAIN_SINR_5GISA
            auto gain = gainCalc (i, finvARe, ftempARe);
            gain =  duplicateReal(gain);
            CF16vec16 postSinr = _mm512_rcp_ph(value_one - gain);
            if constexpr (POST_SINR_FLAG == 1) {
                CF16vec16 postSinrOut = _mm512_mul_ph(gain, postSinr);
                fsumPostSINR[i] += _mm512_cvtph_ps(postSinrOut.real());
            }
            gain = _mm512_mul_ph(gain, llr_beta_fxp_fp16);
            ftempGain[i] = min(gain, max_value);
            postSinr = _mm512_mul_ph (postSinr, llr_postsnr_fxp_dynamic);
            ftempPostSINR[i] = min(postSinr, max_value);

            ftempGain2[i]  = _mm512_permutex2var_epi32(ftempGain[i], use_2nd_half, ftempGain[i]);
            ftempGain[i] = _mm512_permutex2var_epi32(ftempGain[i], use_1st_half, ftempGain[i]);
            ftempPostSINR2[i] = _mm512_permutex2var_epi32(ftempPostSINR[i], use_2nd_half, ftempPostSINR[i]);
            ftempPostSINR[i] = _mm512_permutex2var_epi32(ftempPostSINR[i], use_1st_half, ftempPostSINR[i]);
        }

    }
#endif

    template<size_t N_CH_NUM>
    static void init_input_addr(int16_t iChSymb, int16_t nStartSymbIndex, bblib_pusch_symbol_processing_request *request,
        T * pChIn[N_CH_NUM][N_TX][N_RX], T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX]) {
        const int16_t nAlignedTotalSubCarrier = request->nTotalAlignedSubCarrier;
        const int16_t nDataSymb = request->nSymb;

        //load H (nRxAnt X 2)
        for (size_t i = 0; i < N_RX; i++) {
            if constexpr (N_CH_NUM == 2) {
                for (size_t j = 0; j < N_TX; j ++) {
                    auto pTmp = ptr_cast<int32_t *>(request->pChState[i][j]);
                    pChIn[0][j][i] = ptr_cast<T *>(pTmp);
                    pChIn[1][j][i] = ptr_cast<T *>(pTmp + nAlignedTotalSubCarrier);
                }
                // convert rx pointer
                for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                    auto nDataSymbIdx = request->pSymbIndex[iSymb];
                    pRxIn[nDataSymbIdx][i] = reinterpret_cast<T *>
                        (reinterpret_cast<int32_t *>(request->pRxSignal[i][nDataSymbIdx]));
                    _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T1);
                }
            }
            else {
                for (size_t j = 0; j < N_TX; j ++) {
                    pChIn[0][j][i] = ptr_cast<T *>(ptr_cast<int32_t *>(request->pChState[i][j]) +
                        iChSymb * nAlignedTotalSubCarrier);
                }
                for (int16_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                    const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                    pRxIn[nDataSymbIdx][i] = ptr_cast<T *>(ptr_cast<int32_t *>(request->pRxSignal[i][nDataSymbIdx]));
                    _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T1);
                }
            }
        }
    }

    static void init_table_interp(bblib_pusch_symbol_processing_request *request, int16_t nDataSymb, int16_t nDmrsChSymb,
                                    procDataType wSymAvx[BBLIB_N_SYMB_PER_SF][2], int16_t &flag_symH_upd,
                                    int16_t nDmrsIndex[BBLIB_N_SYMB_PER_SF], int16_t nDmrsIndex_AB[2][BBLIB_N_SYMB_PER_SF]){

        int16_t nMappingType = request->nMappingType;
        int16_t nGranularity = request->nGranularity;
        uint8_t nNrOfDMRSSymbols = request->nNrOfDMRSSymbols;
        for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
            auto nSymNum = request->pSymbIndex[iSymb];
            int16_t wSym0 = 0, wSym1 = 1;
            if(nMappingType == 0) {//type A
                int16_t lut_idx = request->pDmrsSymbolIdx[nNrOfDMRSSymbols] - request->pDmrsSymbolIdx[0] - 4;
                if(lut_idx > 5){
                    lut_idx = 5;
                    printf("\nIndex exceeds the length of wType_A!!\n");
                }
                if (nDmrsChSymb == 2) {
                    wSym0 = wType_A[lut_idx][nGranularity][nSymNum][0];
                    wSym1 = wType_A[lut_idx][nGranularity][nSymNum][1];
                } else if (nDmrsChSymb == 4) {
                    if(request->pDmrsSymbolIdx[0] != 2 || request->pDmrsSymbolIdx[nNrOfDMRSSymbols] != 10){
                        printf("\nOnly support l0 = 2 and pos1 = 10 for dual dmrs typeA!!\n");
                    }
                    wSym0 = wType_2p2[nMappingType][nGranularity][nSymNum][0];
                    wSym1 = wType_2p2[nMappingType][nGranularity][nSymNum][1];
                }
            }
            else{//typeB
                int16_t lut_idx = request->pDmrsSymbolIdx[nNrOfDMRSSymbols]/2 -2;
                if(lut_idx > 3){
                    lut_idx = 3;
                    printf("\nIndex exceeds the length of wType_B!!\n");
                }
                if (nDmrsChSymb == 2) {
                    wSym0 = wType_B[lut_idx][nGranularity][nSymNum][0];
                    wSym1 = wType_B[lut_idx][nGranularity][nSymNum][1];
                } else if (nDmrsChSymb == 4) {
                    if(request->pDmrsSymbolIdx[0] != 0 || request->pDmrsSymbolIdx[nNrOfDMRSSymbols] != 9){
                        printf("\nOnly support l0 = 0 and pos1 = 9 for dual dmrs typeB!!\n");
                    }
                    wSym0 = wType_2p2[nMappingType][nGranularity][nSymNum][0];
                    wSym1 = wType_2p2[nMappingType][nGranularity][nSymNum][1];
                }
            }

            float wcoeff = 1.0;
            if (fp16Int16 == FP16_E::FP16) {
                wcoeff = 16384.0;
            }
            wSymAvx[nSymNum][0] = static_cast<procDataType>((float)wSym0 / wcoeff);
            wSymAvx[nSymNum][1] = static_cast<procDataType>((float)wSym1 / wcoeff);
        }
        if(nMappingType == 0) {//type A
            int16_t lut_idx = request->pDmrsSymbolIdx[nNrOfDMRSSymbols] - request->pDmrsSymbolIdx[0] - 4;
            if (nDmrsChSymb == 2) {
                flag_symH_upd = g_flag_symH_upd_optA[lut_idx][nGranularity];
            } else if (nDmrsChSymb == 4) {
                flag_symH_upd = flag_symH_upd_2p2_opt[nMappingType][nGranularity];
            }
        }
        else{ //typeB
            int16_t lut_idx = request->pDmrsSymbolIdx[nNrOfDMRSSymbols]/2 -2;
            if (nDmrsChSymb == 2) {
                flag_symH_upd = g_flag_symH_upd_optB[lut_idx][nGranularity];
            } else if (nDmrsChSymb == 4) {
                flag_symH_upd = flag_symH_upd_2p2_opt[nMappingType][nGranularity];
            }
        }
        for (size_t i = 0; i < BBLIB_N_SYMB_PER_SF; i++) {
            nDmrsIndex[i] = nDmrsIndex_AB[nMappingType][i];
        }
    }

    FORCE_INLINE inline
    static void init_table_fo(size_t nChSymb, bblib_pusch_symbol_processing_request *request,
        T FoOffsetTable[BBLIB_N_SYMB_PER_SF][N_TX]){
        uint8_t nNrOfDMRSSymbols = request->nNrOfDMRSSymbols;
        float angelOffsetAdd = nNrOfDMRSSymbols == 2 ? 0.5 : 0;
        int16_t nStartSymbIndex = 0;
        int16_t decompOffset = 0;
        int16_t centerH = 0;
        float *FocPhaseFix = NULL;
        if(request->nNumerology == 0){
            FocPhaseFix = FocPhaseFixTable[0];
        }
        else{
            FocPhaseFix = FocPhaseFixTable[1];
        }
        if constexpr (interp_flag == INTERP_E::disable){
            for (size_t iChSymb = 0; iChSymb < nChSymb; iChSymb++) {
                auto iDmrsidx = request->pDmrsSymbolIdx[iChSymb * nNrOfDMRSSymbols];
                centerH = FocPhaseFix[iDmrsidx] + angelOffsetAdd;
                for(int32_t iSymb = 0; iSymb < request->nSymbPerDmrs[iChSymb]; iSymb++){
                    auto nDataSymbIdx = *(request->pSymbIndex + iSymb + nStartSymbIndex);
                    for (size_t j = 0; j < N_TX; j++) {
                        decompOffset = floor((FocPhaseFix[nDataSymbIdx] - centerH) * request->fEstCfo[j] * FO_LUT_SIZE / 2 / PI);
                        decompOffset = (decompOffset + FO_LUT_SIZE) % FO_LUT_SIZE;
                        FoOffsetTable[nDataSymbIdx][j] = T(*(ptr_cast<int32_t *>(&request->pFoCompScCp[decompOffset * 2])));
                    }
                }
                nStartSymbIndex += request->nSymbPerDmrs[iChSymb];
            }
        }
        else{
            int16_t nDataSymb = request->nSymb;
            auto iDmrsidx1 = request->pDmrsSymbolIdx[nNrOfDMRSSymbols];
            auto dmrsDiff = FocPhaseFix[iDmrsidx1] - request->pDmrsSymbolIdx[0];
            for (size_t j = 0; j < N_TX; j++) {
                decompOffset =floor(dmrsDiff * request->fEstCfo[j] * FO_LUT_SIZE / 2 / PI);
                decompOffset = (decompOffset + FO_LUT_SIZE) % FO_LUT_SIZE;
                FoOffsetTable[iDmrsidx1][j] = T(*(ptr_cast<int32_t *>(&request->pFoCompScCp[decompOffset * 2])));
            }
            centerH = request->pDmrsSymbolIdx[0] + angelOffsetAdd;
            for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                auto nSymNum = request->pSymbIndex[iSymb];
                for (size_t j = 0; j < N_TX; j++) {
                    decompOffset = floor((FocPhaseFix[nSymNum] - centerH) * request->fEstCfo[j] * FO_LUT_SIZE / 2 / PI);
                    decompOffset = (decompOffset + FO_LUT_SIZE) % FO_LUT_SIZE;
                    FoOffsetTable[nSymNum][j] = T(*(ptr_cast<int32_t *>(&request->pFoCompScCp[decompOffset * 2])));
                }
            }
        }
    }

    static void tx_calc(T *pRxIn[N_RX], int16_t nAgcGain, FloatSimd invMatrixMulHTansInvRnnRe[N_TX][N_RX],
                            FloatSimd invMatrixMulHTansInvRnnIm[N_TX][N_RX], FloatSimd avxShift, CI16vec16 avxxTxSymbol[N_TX],
                            T *FoTable = {NULL}, bool chUpdateFLag = 1) {
            FloatSimd yRe[N_RX];
            FloatSimd yIm[N_RX];
            #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++){
                auto Y  = *pRxIn[i];
                agc_shift(Y, nAgcGain);
                CI16vec16 reY = swapRealImag(Y);
                yIm[i] = cvt(static_cast<Is32vec16>(Y) >> 16);
                yRe[i] = cvt(static_cast<Is32vec16>(reY) >> 16);
            }
            // X = inverse(H' * invRnn * H + I) * H' * invRnn * Y
            // scale and convert to int16
            if constexpr (fo_flag == FO_E::enable) {
                txCalc<FloatSimd, N_RX, N_TX, FO_E::enable>(invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm, yRe, yIm, avxShift, avxxTxSymbol, FoTable, chUpdateFLag);
            } else {
                txCalc<FloatSimd, N_RX, N_TX>(invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm, yRe, yIm, avxShift, avxxTxSymbol);
            }

            #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                // _mm_prefetch(pRxIn[nDataSymbIdx][i] + 1, _MM_HINT_T1);
                // _mm_prefetch(pRxIn[nDataSymbIdx][i] + 2, _MM_HINT_T1);
                pRxIn[i]++;
            }
    }

#if defined (_BBLIB_SPR_)
    static void tx_calc(T *pRxIn[N_RX], FloatSimd invMatrixMulHTansInvRnnRe[N_TX][N_RX],
                            FloatSimd avxShift, CI16vec16 avxxTxSymbol[N_TX], F16vec32 avxAgcScale, T *pFoTable = {NULL}, bool chUpdateFLag = 1) {
        FloatSimd yRe[N_RX];

        #pragma unroll(N_RX)
        for (size_t i = 0; i < N_RX; i++) {
            //load Y
            yRe[i] = _mm512_cvtepi16_ph(*(CI16vec16*)pRxIn[i]);
            yRe[i] = _mm512_mul_ph(yRe[i], avxAgcScale);
            // pRxIn[i] ++;
        }
        // X = inverse(H' * invRnn * H + I) * H' * invRnn * Y
        // scale and convert to int16
        if constexpr (fo_flag == FO_E::enable) {
            txCalc<FloatSimd, N_RX, N_TX, FO_E::enable>(invMatrixMulHTansInvRnnRe, yRe, avxShift, avxxTxSymbol, pFoTable, chUpdateFLag);
        } else {
            txCalc<FloatSimd, N_RX, N_TX>(invMatrixMulHTansInvRnnRe, yRe, avxShift, avxxTxSymbol);
        }

        #pragma unroll(N_RX)
        for (size_t i = 0; i < N_RX; i ++) {
            pRxIn[i] ++;
            _mm_prefetch(pRxIn[i] + 1, _MM_HINT_T2);
            _mm_prefetch(pRxIn[i] + 2, _MM_HINT_T2);
        }
    }
#endif

    // mmse llr mimo
    // template<size_t N_RX = 16, size_t N_TX = 16, uint8_t nLayerPerUe = 2 , typename T = CI16vec16>
    static void mimo_mmse_llr_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response){
        // DECLARE_VAR_COMMON
        CI16vec16 avxxTxSymbol[N_TX];
        FloatSimd ftempGain[N_TX] = {FloatSimd()};
        FloatSimd ftempPostSINR[N_TX] = {FloatSimd()};
        F32vec16  fsumPostSINR[N_TX] = {F32vec16()};

#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
        // BUFFER_REAL
        FloatSimd ftempARe[N_TX][N_TX];
        FloatSimd finvARe[N_TX][N_TX];
        FloatSimd chRe[N_RX][N_TX];
        FloatSimd hTransInvRnnRe[N_TX][N_RX];
        FloatSimd invMatrixMulHTansInvRnnRe[N_TX][N_RX];
        invType finvRnnRe[N_RX][N_RX];
#endif
#if  defined (_BBLIB_AVX512_)
        // BUFFER_IMAGE
        FloatSimd ftempAIm[N_TX][N_TX];
        FloatSimd finvAIm[N_TX][N_TX];
        FloatSimd chIm[N_RX][N_TX];
        FloatSimd hTransInvRnnIm[N_TX][N_RX];
        FloatSimd invMatrixMulHTansInvRnnIm[N_TX][N_RX];
        invType finvRnnIm[N_RX][N_RX];
#endif
        // CTRL_VAR_COMMON
        int16_t nSubCarrier = request->nSubCarrier;
        int16_t nIrcRbgStart = request->nIrcRbgStart;
        int16_t nTime = nSubCarrier / 16;
        int16_t nTime0 = (nSubCarrier + 15) / 16;
        int16_t nRestLen = nSubCarrier - nTime * 16;
        int16_t nDisableRnnInv = request->nDisableRnnInv;
        FloatSimd avxShift = FloatSimd();
        FloatSimd llr_postsnr_fxp_dynamic = FloatSimd();
        int16_t llr_range_high = 0;
        int16_t llr_range_low = 0;
        init_factor(request, avxShift, llr_postsnr_fxp_dynamic, llr_range_high, llr_range_low);

        if(request->nLlrSaturatedBits < 2 || request->nLlrSaturatedBits > 8) {
            printf("Error! Not support nLlrSaturatedBits %d in pusch symbol process, valid range 2~8\n", request->nLlrSaturatedBits);
            return;
        }


        uint16_t nDmrsPortIdx = 0;
        uint16_t dataDmrsMux = 0;
        data_dmrs_mux(request, dataDmrsMux, nDmrsPortIdx);

        // CTRL_VAR_MMSE
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX];
        T FoOffsetTable[BBLIB_N_SYMB_PER_SF][N_TX];
        T *pChIn[1][N_TX][N_RX];
        int16_t nStartSymbIndex = 0;
        const int16_t nChSymb = request->nChSymb;
#ifdef _BBLIB_SPR_
        float16 fAgcScale = (float16)(1.0 / (request->nAgcGain));
        F16vec32 avxAgcScale = _mm512_set1_ph(fAgcScale);
#endif

#ifdef SUBMODULE_TICK
        uint64_t nSubModuleTick[PUSCH_MMSE_IRC_SUBMODULE_MAX][3];
#endif

        LOG_TICK_INIT(PUSCH_MMSE_IRC_SUBMODULE_MAX);

        if (0 == nDisableRnnInv)
            rnn_inverse<T, N_RX>(request, 0);

        if constexpr (fo_flag == FO_E::enable) {
            init_table_fo(nChSymb, request, FoOffsetTable);
        }
        // loop channel symbol
        #pragma loop_count min(1), max(14)
        for (int16_t iChSymb = 0; iChSymb < nChSymb; iChSymb++) {

            //initial ch prt and rx ptr
            init_input_addr<1>(iChSymb, nStartSymbIndex, request, pChIn, pRxIn);

            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {

                xran_decomp(request,  pRxIn, nSCIdx, nSubCarrier, iChSymb);

                int32_t nSc = (nSCIdx + 16 > nSubCarrier) * nRestLen + (nSCIdx + 16 <= nSubCarrier) * 16;
                LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_H);
                #pragma unroll(N_RX)
                for (size_t i = 0; i < N_RX; i ++) {
                    #pragma unroll(N_TX)
                        for (size_t j = 0; j < N_TX; j++) {
                            T chIn = *pChIn[0][j][i]++;

                            if constexpr (fp16Int16 == FP16_E::INT16){
                                CI16vec16 reIm = swapRealImag(chIn);
                                chIm[i][j] = cvt(static_cast<Is32vec16>(chIn) >> 16);
                                chRe[i][j] = cvt(static_cast<Is32vec16>(reIm) >> 16);
                            }
#if defined (_BBLIB_SPR_)
                            else{
                                chRe[i][j] = chIn;
                                _mm_prefetch(pChIn[0][j][i], _MM_HINT_T2);
                            }
#endif
                        }
                }
                LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_H);

                LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_RNN);
                if ((nSCIdx % 48) == 0) {
                    size_t rnnIndx = nIrcRbgStart + nSCIdx / 48;
                    if constexpr (fp16Int16 == FP16_E::INT16){
                        load_rnn(request, rnnIndx, iChSymb, finvRnnRe, finvRnnIm);

                    }
#if defined (_BBLIB_SPR_)
                    else
                    {

                        load_rnn(request, rnnIndx, iChSymb, finvRnnRe);
                    }
#endif
                }
                LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_RNN);

                if constexpr (fp16Int16 == FP16_E::INT16){
                    // newH' = H' * invRnn
                    LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);
                    HTransMulInvRnn<FloatSimd, invType, N_RX, N_TX>(chRe, chIm, finvRnnRe, finvRnnIm, hTransInvRnnRe, hTransInvRnnIm);
                    LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);

                    // newH' * H + I
                    LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);
                    HTransMulInvRnnMulHPlusI<FloatSimd, N_RX, N_TX>(hTransInvRnnRe, hTransInvRnnIm, chRe, chIm, ftempARe, ftempAIm);
                    LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);

                    // inverse(newH' * H + I)
                    LOG_TICK_START(PUSCH_MMSE_IRC_MATRIX_INVERSE);
                    matrix_inverse<N_TX>(ftempARe, ftempAIm, finvARe, finvAIm);
                    LOG_TICK_END(PUSCH_MMSE_IRC_MATRIX_INVERSE);

                    // inverse(H' * invRnn * H + I) * H' * invRnn
                    LOG_TICK_START(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);
                    invMatrixMulHTransMulInvRnn<FloatSimd, N_RX, N_TX>(finvARe, finvAIm, hTransInvRnnRe, hTransInvRnnIm,
                                        invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm);
                    LOG_TICK_END(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);

                    LOG_TICK_START(PUSCH_MMSE_IRC_GAIN_SINR);
                    if (iChSymb == 0) {
                        gain_sinr<1>(finvARe, ftempARe, finvAIm, ftempAIm, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                    } else {
                        gain_sinr<0>(finvARe, ftempARe, finvAIm, ftempAIm, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                    }
                    LOG_TICK_END(PUSCH_MMSE_IRC_GAIN_SINR);
                }
#if defined (_BBLIB_SPR_)
                else
                {
                    // newH' = H' * invRnn
                    LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);
                    HTransMulInvRnn<FloatSimd, invType, N_RX, N_TX>(chRe, finvRnnRe, hTransInvRnnRe);
                    LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);

                    // newH' * H + I
                    LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);
                    HTransMulInvRnnMulHPlusI<FloatSimd, N_RX, N_TX>(hTransInvRnnRe, chRe, ftempARe, finvARe);
                    LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);

                    // inverse(newH' * H + I)
                    LOG_TICK_START(PUSCH_MMSE_IRC_MATRIX_INVERSE);
                    matrix_inverse<T, N_TX>(finvARe);
                    LOG_TICK_END(PUSCH_MMSE_IRC_MATRIX_INVERSE);

                    // inverse(H' * invRnn * H + I) * H' * invRnn
                    LOG_TICK_START(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);
                    invMatrixMulHTransMulInvRnn<FloatSimd, N_RX, N_TX>(finvARe, hTransInvRnnRe, invMatrixMulHTansInvRnnRe);
                    LOG_TICK_END(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);

                    LOG_TICK_START(PUSCH_MMSE_IRC_GAIN_SINR);
                    if (iChSymb == 0) {
                        gain_sinr<1>(finvARe, ftempARe, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                    } else {
                        gain_sinr<0>(finvARe, ftempARe, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                    }
                    LOG_TICK_END(PUSCH_MMSE_IRC_GAIN_SINR);
                }
#endif

                for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++){

                    auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);

                    auto dataDmrsFlag = dataDmrsMux & ((uint16_t) 1) << nDataSymbIdx;

                    LOG_TICK_START(PUSCH_MMSE_IRC_TX_CALC);
                    if constexpr (fp16Int16 == FP16_E::INT16){
                        tx_calc(pRxIn[nDataSymbIdx], request->nAgcGain, invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm, avxShift, avxxTxSymbol, FoOffsetTable[nDataSymbIdx]);
                    }
#if defined (_BBLIB_SPR_)
                    else {
                        tx_calc(pRxIn[nDataSymbIdx], invMatrixMulHTansInvRnnRe, avxShift, avxxTxSymbol, avxAgcScale, FoOffsetTable[nDataSymbIdx]);
                    }
#endif
                    LOG_TICK_END(PUSCH_MMSE_IRC_TX_CALC);

                    LOG_TICK_START(PUSCH_MMSE_IRC_LLR_DEMAPER);

                    if (dataDmrsFlag == 0) {
                        DEMAPPER<T,DATA_DMRS_MUX_E::disable,DMRSTYPE,NROFCDMS>::demaper(
                            request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                            request->eModOrder, response->pLlr[nDataSymbIdx],
                            ftempPostSINR, ftempGain, avxxTxSymbol, nSc, nSCIdx, llr_range_low, llr_range_high,
                            nDmrsPortIdx);
                    } else {
                        DEMAPPER<T,DATA_DMRS_MUX_E::enable,DMRSTYPE,NROFCDMS>::demaper(
                            request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                            request->eModOrder, response->pLlr[nDataSymbIdx],
                            ftempPostSINR, ftempGain, avxxTxSymbol, nSc, nSCIdx, llr_range_low, llr_range_high,
                            nDmrsPortIdx);
                    }

                    LOG_TICK_END(PUSCH_MMSE_IRC_LLR_DEMAPER);
                }//for nSymbIdx
            }//end the nSCIdx loop
            if (iChSymb == 0) {
                if (nTime <= 3) {
                    for (size_t j = 0; j < N_TX; j++)
                    {
                        response->fPostSINR[j] = reduce_add(fsumPostSINR[j]) / nSubCarrier;
                    }
                } else {
                    for (size_t j = 0; j < N_TX; j++)
                    {
                        response->fPostSINR[j] = *(reinterpret_cast<float *>(&fsumPostSINR[j])) / nTime0;
                    }
                }
            }
            nStartSymbIndex += request->nSymbPerDmrs[iChSymb];
        }//end iChSymb loop
        LOG_TICK_REPORT(response, PUSCH_MMSE_IRC_SUBMODULE_MAX)
    }
// #endif

    // mmse llr mimo linear interpolation
    static void mimo_mmse_llr_avx512_lin_interp(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
    {
        // DECLARE_VAR_COMMON
        CI16vec16 avxxTxSymbol[N_TX];
        FloatSimd ftempGain[N_TX] = {FloatSimd()};
        FloatSimd ftempPostSINR[N_TX] = {FloatSimd()};
        F32vec16  fsumPostSINR[N_TX] = {F32vec16()};

        T chIn = T();
#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
        // BUFFER_REAL
        FloatSimd ftempARe[N_TX][N_TX];
        FloatSimd finvARe[N_TX][N_TX];
        FloatSimd chRe[N_RX][N_TX];
        FloatSimd hTransInvRnnRe[N_TX][N_RX];
        FloatSimd invMatrixMulHTansInvRnnRe[N_TX][N_RX];
        invType finvRnnRe[2][N_RX][N_RX];
#endif
#if defined (_BBLIB_AVX512_)
        // BUFFER_IMAGE
        FloatSimd ftempAIm[N_TX][N_TX];
        FloatSimd finvAIm[N_TX][N_TX];
        FloatSimd chIm[N_RX][N_TX];
        FloatSimd hTransInvRnnIm[N_TX][N_RX];
        FloatSimd invMatrixMulHTansInvRnnIm[N_TX][N_RX];
        invType finvRnnIm[2][N_RX][N_RX];
#endif
        T chTmp0[N_RX][N_TX];
        T chTmp1[N_RX][N_TX];
        // CTRL_VAR_COMMON
        int16_t nSubCarrier = request->nSubCarrier;
        int16_t nIrcRbgStart = request->nIrcRbgStart;
        int16_t nTime = nSubCarrier / 16;
        int16_t nTime0 = (nSubCarrier + 15) / 16;
        int16_t nRestLen = nSubCarrier - nTime * 16;
        uint8_t nNrOfDMRSSymbols = request->nNrOfDMRSSymbols;

        FloatSimd avxShift = FloatSimd();
        FloatSimd llr_postsnr_fxp_dynamic = FloatSimd();
        int16_t llr_range_high = 0;
        int16_t llr_range_low = 0;
        init_factor(request, avxShift, llr_postsnr_fxp_dynamic, llr_range_high, llr_range_low);
#ifdef _BBLIB_SPR_
        float16 fAgcScale = (float16)(1.0 / (request->nAgcGain));
        F16vec32 avxAgcScale = _mm512_set1_ph(fAgcScale);
#endif

        uint16_t nDmrsPortIdx = 0;
        uint16_t dataDmrsMux = 0;

        data_dmrs_mux(request, dataDmrsMux, nDmrsPortIdx);

        // CTRL_VAR_INTERP
        T FoOffsetTable[BBLIB_N_SYMB_PER_SF][N_TX];
        // T *pRxIn[N_RX];
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX];
        T *pChIn[2][N_TX][N_RX];

        procDataType wSymAvx[BBLIB_N_SYMB_PER_SF][2];
        int16_t flag_symH_upd = 0;
        int16_t nDmrsIndex[BBLIB_N_SYMB_PER_SF] = { 0 };
        int16_t dmrsIndex = 0;

        int16_t nDataSymb = request->nSymb;
        const int16_t nChSymb = request->nChSymb;
        int16_t nDmrsChSymb = request->nDmrsChSymb;
        int16_t nDisableRnnInv = request->nDisableRnnInv;
        int16_t nEnable2ScProcess = request->nEnable2ScProcess;
        int16_t *pIntrpweights[BBLIB_MAX_TX_LAYER_NUM];

        for (size_t iLayer = 0; iLayer < N_TX; iLayer++) {
            pIntrpweights[iLayer] = request->pIntrpweights[iLayer];
        }

#ifdef SUBMODULE_TICK
        uint64_t nSubModuleTick[PUSCH_MMSE_IRC_SUBMODULE_MAX][3];
#endif

        LOG_TICK_INIT(PUSCH_MMSE_IRC_SUBMODULE_MAX);

        if ((nDmrsChSymb != 2) && (nDmrsChSymb != 4)) {
            printf("Wrong API: only two DMRS symbols are supported currently with linear interpolation\n");
            return;
        }
        init_table_interp(request, nDataSymb, nDmrsChSymb, wSymAvx, flag_symH_upd, nDmrsIndex, nDmrsIndex_AB);

        nDmrsChSymb = 2;
        if constexpr (fo_flag == FO_E::enable) {
            init_table_fo(nChSymb, request, FoOffsetTable);
        }

        init_input_addr<2>(0, 0, request, pChIn, pRxIn);
        auto iDmrsidx1 = request->pDmrsSymbolIdx[nNrOfDMRSSymbols];
        if (0 == nEnable2ScProcess) // default, process for every subcarrier
        {
            if (0 == nDisableRnnInv)
                rnn_inverse<T, N_RX>(request, 0);
            // loop channel symbol
            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
                int32_t nSc = (nSCIdx + 16 > nSubCarrier) * nRestLen + (nSCIdx + 16 <= nSubCarrier) * 16;

                xran_decomp(request,  pRxIn, nSCIdx, nSubCarrier, 0);

                LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_RNN);
                if (1 == nDisableRnnInv) {  // Use averaged Rnn if nDisableRnnInv is enabled.
                    if ((nSCIdx % 48) == 0) {
                        size_t rnnIndx = nIrcRbgStart + nSCIdx / 48;
                        dmrsIndex = 0;
                        if constexpr (fp16Int16 == FP16_E::INT16){
                            load_rnn(request, rnnIndx, dmrsIndex, finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex]);
                        }
#if defined (_BBLIB_SPR_)
                        else
                        {
                            load_rnn(request, rnnIndx, dmrsIndex, finvRnnRe[dmrsIndex]);
                        }
#endif
                    }
                }
                LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_RNN);

                LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_H);
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        chTmp0[i][j] = *pChIn[0][j][i];
                        chTmp1[i][j] = *pChIn[1][j][i];
                        if constexpr (fo_flag == FO_E::enable){
                            chTmp1[i][j] = fmulconj(chTmp1[i][j], FoOffsetTable[iDmrsidx1][j]);
                        }
                    }
                }
                LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_H);

                #pragma loop_count min(1), max(14)
                for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                    procDataType w_dmrsSym0, w_dmrsSym1;
                    int16_t w_temp_dmrsSym0, w_temp_dmrsSym1;
                    uint32_t nSymNum = request->pSymbIndex[iSymb];
                    auto dataDmrsFlag = dataDmrsMux & ((uint16_t) 1) << nSymNum;
                    LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_RNN);
                    if (0 == nDisableRnnInv) {   // Use Rnn for each DMRS if nDisableRnnInv is disabled.
                        if (nDmrsIndex[nSymNum] != 0) {
                            dmrsIndex = nDmrsIndex[nSymNum] - 1;
                            if ((nSCIdx % 48) == 0) {
                                size_t rnnIndx = nIrcRbgStart + nSCIdx / 48;
                                if constexpr (fp16Int16 == FP16_E::INT16){
                                    load_rnn(request, rnnIndx, dmrsIndex, finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex]);
                                }
#if defined (_BBLIB_SPR_)
                                else
                                {
                                    load_rnn(request, rnnIndx, dmrsIndex, finvRnnRe[dmrsIndex]);
                                }
#endif
                            }
                        }
                    }
                    LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_RNN);

                    // If it is needed to update interpolated CE, otherwise previous sym parameters reused
                    if (flag_symH_upd & (1 << (BBLIB_N_SYMB_PER_SF -  1 - nSymNum))) {
                        LOG_TICK_START(PUSCH_MMSE_IRC_CALC_H);
                        #pragma unroll(N_TX)
                        for (size_t j = 0; j < N_TX; j ++) {
                            #pragma unroll(N_RX)
                            for (size_t i = 0; i < N_RX; i++) {

                                if (request->nGranularity < 6) {
                                    w_dmrsSym0 = wSymAvx[nSymNum][0];
                                    w_dmrsSym1 = wSymAvx[nSymNum][1];
                                }
                                else {
                                    w_temp_dmrsSym0 = *(pIntrpweights[j] + nSymNum*4);
                                    w_temp_dmrsSym1 = *(pIntrpweights[j] + nSymNum*4+1);
                                    w_dmrsSym0 = static_cast<procDataType>(w_temp_dmrsSym0);
                                    w_dmrsSym1 = static_cast<procDataType>(w_temp_dmrsSym1);
                                }

                                auto temp0 = mulhrs(chTmp0[i][j], T(w_dmrsSym0));
                                auto temp1 = mulhrs(chTmp1[i][j], T(w_dmrsSym1));

                                if constexpr (fp16Int16 == FP16_E::INT16){
                                    chIn = (temp0 + temp1) << 1;
                                }
#if defined (_BBLIB_SPR_)
                                else
                                {
                                    chIn = temp0 + temp1;
                                }
#endif

                                if constexpr (fp16Int16 == FP16_E::INT16){
                                    CI16vec16 reIm = swapRealImag(chIn);
                                    chIm[i][j] = cvt(static_cast<Is32vec16>(chIn) >> 16);
                                    chRe[i][j] = cvt(static_cast<Is32vec16>(reIm) >> 16);
                                }
#if defined (_BBLIB_SPR_)
                                else
                                {
                                    chRe[i][j] = chIn;
                                }
#endif
                            }
                        }
                        LOG_TICK_END(PUSCH_MMSE_IRC_CALC_H);

                        if constexpr (fp16Int16 == FP16_E::INT16){
                            // newH' = H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);
                            HTransMulInvRnn<FloatSimd, Float, N_RX, N_TX>(chRe, chIm, finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex], hTransInvRnnRe, hTransInvRnnIm);
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);

                            // newH' * H + I
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);
                            HTransMulInvRnnMulHPlusI<FloatSimd, N_RX, N_TX>(hTransInvRnnRe, hTransInvRnnIm, chRe, chIm, ftempARe, ftempAIm);
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);

                            // inverse(newH' * H + I)
                            LOG_TICK_START(PUSCH_MMSE_IRC_MATRIX_INVERSE);
                            matrix_inverse<N_TX>(ftempARe, ftempAIm, finvARe, finvAIm);
                            LOG_TICK_END(PUSCH_MMSE_IRC_MATRIX_INVERSE);

                            // inverse(H' * invRnn * H + I) * H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);
                            invMatrixMulHTransMulInvRnn<FloatSimd, N_RX, N_TX>(finvARe, finvAIm, hTransInvRnnRe, hTransInvRnnIm,
                                            invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm);
                            LOG_TICK_END(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);

                            LOG_TICK_START(PUSCH_MMSE_IRC_GAIN_SINR);
                            if (iSymb == 0) {
                                gain_sinr<1>(finvARe, ftempARe, finvAIm, ftempAIm, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            } else {
                                gain_sinr<0>(finvARe, ftempARe, finvAIm, ftempAIm, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            }
                            LOG_TICK_END(PUSCH_MMSE_IRC_GAIN_SINR);
                        }
#if defined (_BBLIB_SPR_)
                        else
                        {
                            // newH' = H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);
                            HTransMulInvRnn<FloatSimd, invType, N_RX, N_TX>(chRe, finvRnnRe[dmrsIndex], hTransInvRnnRe);
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);

                            // newH' * H + I
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);
                            HTransMulInvRnnMulHPlusI<FloatSimd, N_RX, N_TX>(hTransInvRnnRe, chRe, ftempARe, finvARe);
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);

                            // inverse(newH' * H + I)
                            LOG_TICK_START(PUSCH_MMSE_IRC_MATRIX_INVERSE);
                            matrix_inverse<T, N_TX>(finvARe);
                            LOG_TICK_END(PUSCH_MMSE_IRC_MATRIX_INVERSE);

                            // inverse(H' * invRnn * H + I) * H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);
                            invMatrixMulHTransMulInvRnn<FloatSimd, N_RX, N_TX>(finvARe, hTransInvRnnRe, invMatrixMulHTansInvRnnRe);
                            LOG_TICK_END(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);

                            LOG_TICK_START(PUSCH_MMSE_IRC_GAIN_SINR);
                            if (iSymb == 0) {
                                gain_sinr<1>(finvARe, ftempARe, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            } else {
                                gain_sinr<0>(finvARe, ftempARe, ftempGain, ftempPostSINR, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            }
                            LOG_TICK_END(PUSCH_MMSE_IRC_GAIN_SINR);
                        }
#endif
                    }
                    // X = inverse(H' * invRnn * H + I) * H' * invRnn * Y
                    // 4. x = invA * z
                    LOG_TICK_START(PUSCH_MMSE_IRC_TX_CALC);
                    if constexpr (fp16Int16 == FP16_E::INT16){
                        tx_calc(pRxIn[nSymNum], request->nAgcGain, invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm,
                                avxShift, avxxTxSymbol, FoOffsetTable[nSymNum]);
                    }
#if defined (_BBLIB_SPR_)
                    else {
                        tx_calc(pRxIn[nSymNum], invMatrixMulHTansInvRnnRe, avxShift,
                                avxxTxSymbol, avxAgcScale, FoOffsetTable[nSymNum]);
                    }
#endif
                    LOG_TICK_END(PUSCH_MMSE_IRC_TX_CALC);


                    LOG_TICK_START(PUSCH_MMSE_IRC_LLR_DEMAPER);

                    if (dataDmrsFlag == 0) {
                        DEMAPPER<T,DATA_DMRS_MUX_E::disable,DMRSTYPE,NROFCDMS>::demaper(
                            request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                            request->eModOrder, response->pLlr[nSymNum],
                            ftempPostSINR, ftempGain, avxxTxSymbol, nSc, nSCIdx, llr_range_low, llr_range_high,
                            nDmrsPortIdx);
                    } else {
                        DEMAPPER<T,DATA_DMRS_MUX_E::enable,DMRSTYPE,NROFCDMS>::demaper(
                            request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                            request->eModOrder, response->pLlr[nSymNum],
                            ftempPostSINR, ftempGain, avxxTxSymbol, nSc, nSCIdx, llr_range_low, llr_range_high,
                            nDmrsPortIdx);
                    }

                    LOG_TICK_END(PUSCH_MMSE_IRC_LLR_DEMAPER);

                }//for iSymb

                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        pChIn[0][j][i]++;
                        pChIn[1][j][i]++;
                    }
                }
            }//end the sc loop
            if (nTime <= 3) {
                for (size_t j = 0; j < N_TX; j++)
                {
                    response->fPostSINR[j] = reduce_add(fsumPostSINR[j]) / nSubCarrier;
                }
            } else {
                for (size_t j = 0; j < N_TX; j++)
                {
                    response->fPostSINR[j] = *(reinterpret_cast<float *>(&fsumPostSINR[j])) / nTime0;
                }
            }
        }
        else // process for every 2SC
        {

#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
            // BUFFER_REAL
            FloatSimd invMatrixMulHTansInvRnnRe2[N_TX][N_RX];
            invType finvRnnRe2[2][N_RX][N_RX];
#endif
#if defined (_BBLIB_AVX512_)
            // BUFFER_IMAGE
            FloatSimd invMatrixMulHTansInvRnnIm2[N_TX][N_RX];
            invType finvRnnIm2[2][N_RX][N_RX];
            FloatSimd ftempGain2[N_TX] = {FloatSimd()};
            FloatSimd ftempGaintmp2[N_TX];
            FloatSimd ftempPostSINR2[N_TX] = {FloatSimd()};
            FloatSimd ftempPostSINRtmp2[N_TX];
#endif
            if (0 == nDisableRnnInv)
                rnn_inverse<T, N_RX>(request, 1);

            nTime = nSubCarrier / 32;
            nTime0 = (nSubCarrier + 31) / 32;
            nRestLen = nSubCarrier - nTime * 32;

            // for functional test
            //nSubCarrier = nTime * 32;

            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 32) {
                int32_t nScTmp = (nSCIdx + 32 > nSubCarrier) * nRestLen + (nSCIdx + 32 <= nSubCarrier) * 32;
                int32_t nSc = (nScTmp > 16) ? 16 : nScTmp;
                int32_t nSc1 = (nScTmp > 16) ? (nScTmp - 16) : 0;

                xran_decomp(request,  pRxIn, nSCIdx, nSubCarrier, 0);

                LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_RNN);
                if (nSCIdx % 96 == 0){
                    size_t rnnIndx = nIrcRbgStart + nSCIdx / 48;
                    if constexpr (fp16Int16 == FP16_E::INT16){
                        load_rnn2(request, rnnIndx, dmrsIndex, finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex],
                                    finvRnnRe2[dmrsIndex], finvRnnIm2[dmrsIndex]);
                    }
#if defined (_BBLIB_SPR_)
                    else
                    {
                        load_rnn2(request, rnnIndx, dmrsIndex, finvRnnRe[dmrsIndex], finvRnnRe2[dmrsIndex]);
                    }
#endif
                }
                LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_RNN);

                LOG_TICK_START(PUSCH_MMSE_IRC_LOAD_H);
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        T temp0 = *pChIn[0][j][i];
                        T temp1 = *(pChIn[0][j][i] + 1);
                        chTmp0[i][j] = _mm512_permutex2var_epi32(temp0, m512LoadGran2, temp1);
                        temp0 = *pChIn[1][j][i];
                        temp1 = *(pChIn[1][j][i] + 1);
                        chTmp1[i][j] = _mm512_permutex2var_epi32(temp0, m512LoadGran2, temp1);
                        if constexpr (fo_flag == FO_E::enable){
                            chTmp1[i][j] = fmulconj(chTmp1[i][j], FoOffsetTable[iDmrsidx1][j]);
                        }
                    }
                }
                LOG_TICK_END(PUSCH_MMSE_IRC_LOAD_H);

                #pragma loop_count min(1), max(14)
                for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                    auto nSymNum = request->pSymbIndex[iSymb];
                    auto dataDmrsFlag = dataDmrsMux & ((uint16_t) 1) << nSymNum;

                    // convert rx pointer
                    // If it is needed to update interpolated CE, otherwise previous sym parameters reused
                    if (flag_symH_upd & (1 << (BBLIB_N_SYMB_PER_SF -  1 - nSymNum))) {
                        LOG_TICK_START(PUSCH_MMSE_IRC_CALC_H);
                        #pragma unroll(N_TX)
                        for (size_t j = 0; j < N_TX; j ++) {
                            #pragma unroll(N_RX)
                            for (size_t i = 0; i < N_RX; i++) {
                                auto temp0 = mulhrs(chTmp0[i][j], T(wSymAvx[nSymNum][0]));
                                auto temp1 = mulhrs(chTmp1[i][j], T(wSymAvx[nSymNum][1]));

                                if constexpr (fp16Int16 == FP16_E::INT16){
                                    chIn = (temp0 + temp1) << 1;
                                }
#if defined (_BBLIB_SPR_)
                                else
                                {
                                    chIn = temp0 + temp1;
                                }
#endif

                                if constexpr (fp16Int16 == FP16_E::INT16){
                                    CI16vec16 reIm = swapRealImag(chIn);
                                    chIm[i][j] = cvt(static_cast<Is32vec16>(chIn) >> 16);
                                    chRe[i][j] = cvt(static_cast<Is32vec16>(reIm) >> 16);
                                }
#if defined (_BBLIB_SPR_)
                                else
                                {
                                    chRe[i][j] = chIn;
                                }
#endif
                            }
                        }
                        LOG_TICK_END(PUSCH_MMSE_IRC_CALC_H);

                        if constexpr (fp16Int16 == FP16_E::INT16){
                            // newH' = H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);
                            if (nSCIdx % 96 == 0){
                                HTransMulInvRnn2<FloatSimd, Float, N_RX, N_TX>(chRe, chIm, finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex], finvRnnRe2[dmrsIndex], finvRnnIm2[dmrsIndex], hTransInvRnnRe, hTransInvRnnIm, 0);
                            } else if (nSCIdx % 96 == 32){
                                HTransMulInvRnn2<FloatSimd, Float, N_RX, N_TX>(chRe, chIm, finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex], finvRnnRe2[dmrsIndex], finvRnnIm2[dmrsIndex], hTransInvRnnRe, hTransInvRnnIm, 1);
                            } else if (nSCIdx % 96 == 64){
                                HTransMulInvRnn2<FloatSimd, Float, N_RX, N_TX>(chRe, chIm, finvRnnRe2[dmrsIndex], finvRnnIm2[dmrsIndex], finvRnnRe[dmrsIndex], finvRnnIm[dmrsIndex], hTransInvRnnRe, hTransInvRnnIm, 0);
                            }
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);

                            // newH' * H + I
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);
                            HTransMulInvRnnMulHPlusI<FloatSimd, N_RX, N_TX>(hTransInvRnnRe, hTransInvRnnIm, chRe, chIm, ftempARe, ftempAIm);
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);

                            // inverse(newH' * H + I)
                            LOG_TICK_START(PUSCH_MMSE_IRC_MATRIX_INVERSE);
                            matrix_inverse<N_TX>(ftempARe, ftempAIm, finvARe, finvAIm);
                            LOG_TICK_END(PUSCH_MMSE_IRC_MATRIX_INVERSE);

                            // inverse(H' * invRnn * H + I) * H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);
                            invMatrixMulHTransMulInvRnn2<FloatSimd, N_RX, N_TX>(finvARe, finvAIm, hTransInvRnnRe, hTransInvRnnIm,
                                        invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm,invMatrixMulHTansInvRnnRe2, invMatrixMulHTansInvRnnIm2);
                            LOG_TICK_END(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);

                            LOG_TICK_START(PUSCH_MMSE_IRC_GAIN_SINR);
                            if (iSymb == 0) {
                                gain_sinr2<1>(finvARe, ftempARe, finvAIm, ftempAIm, ftempGain, ftempGain2, ftempPostSINR, ftempPostSINR2, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            } else {
                                gain_sinr2<0>(finvARe, ftempARe, finvAIm, ftempAIm, ftempGain, ftempGain2, ftempPostSINR, ftempPostSINR2, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            }
                            LOG_TICK_END(PUSCH_MMSE_IRC_GAIN_SINR);
                        }
#if defined (_BBLIB_SPR_)
                        else
                        {
                            // newH' = H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);
                            if (nSCIdx % 96 == 0){
                                HTransMulInvRnn2<FloatSimd, invType, N_RX, N_TX>(chRe, finvRnnRe[dmrsIndex], finvRnnRe2[dmrsIndex], hTransInvRnnRe, 0);
                            } else if (nSCIdx % 96 == 32){
                                HTransMulInvRnn2<FloatSimd, invType, N_RX, N_TX>(chRe, finvRnnRe[dmrsIndex], finvRnnRe2[dmrsIndex], hTransInvRnnRe, 1);
                            } else if (nSCIdx % 96 == 64){
                                HTransMulInvRnn2<FloatSimd, invType, N_RX, N_TX>(chRe, finvRnnRe2[dmrsIndex], finvRnnRe[dmrsIndex], hTransInvRnnRe, 0);
                            }
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN);

                            // newH' * H + I
                            LOG_TICK_START(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);
                            HTransMulInvRnnMulHPlusI<FloatSimd, N_RX, N_TX>(hTransInvRnnRe, chRe, ftempARe, finvARe);
                            LOG_TICK_END(PUSCH_MMSE_IRC_H_TRANS_MUL_INV_RNN_MUL_H_PLUS_I);

                            // inverse(newH' * H + I)
                            LOG_TICK_START(PUSCH_MMSE_IRC_MATRIX_INVERSE);
                            matrix_inverse<T, N_TX>(finvARe);
                            LOG_TICK_END(PUSCH_MMSE_IRC_MATRIX_INVERSE);

                            // inverse(H' * invRnn * H + I) * H' * invRnn
                            LOG_TICK_START(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);
                            invMatrixMulHTransMulInvRnn2<FloatSimd, N_RX, N_TX>(finvARe, hTransInvRnnRe, invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnRe2);
                            LOG_TICK_END(PUSCH_MMSE_IRC_INV_MATRIX_MUL_H_TRANS_MUL_INV_RNN);

                            LOG_TICK_START(PUSCH_MMSE_IRC_GAIN_SINR);
                            if (iSymb == 0) {
                                gain_sinr2<1>(finvARe, ftempARe, ftempGain, ftempGain2, ftempPostSINR, ftempPostSINR2, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            } else {
                                gain_sinr2<0>(finvARe, ftempARe, ftempGain, ftempGain2, ftempPostSINR, ftempPostSINR2, fsumPostSINR, llr_postsnr_fxp_dynamic);
                            }
                            LOG_TICK_END(PUSCH_MMSE_IRC_GAIN_SINR);
                        }
#endif
                    }
                    // X = inverse(H' * invRnn * H + I) * H' * invRnn * Y
                    // 4. x = invA * z
                    LOG_TICK_START(PUSCH_MMSE_IRC_TX_CALC);
                    if constexpr (fp16Int16 == FP16_E::INT16){
                        tx_calc(pRxIn[nSymNum], request->nAgcGain, invMatrixMulHTansInvRnnRe, invMatrixMulHTansInvRnnIm,
                                avxShift, avxxTxSymbol, FoOffsetTable[nSymNum]);
                    }
#if defined (_BBLIB_SPR_)
                    else {
                        tx_calc(pRxIn[nSymNum], invMatrixMulHTansInvRnnRe,
                                avxShift, avxxTxSymbol, avxAgcScale, FoOffsetTable[nSymNum]);
                    }
#endif

                    LOG_TICK_START(PUSCH_MMSE_IRC_LLR_DEMAPER);

                    if (dataDmrsFlag == 0) {
                        DEMAPPER<T,DATA_DMRS_MUX_E::disable,DMRSTYPE,NROFCDMS>::demaper(
                            request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                            request->eModOrder, response->pLlr[nSymNum],
                            ftempPostSINR, ftempGain, avxxTxSymbol, nSc, nSCIdx, llr_range_low, llr_range_high,
                            nDmrsPortIdx);
                    } else {
                        DEMAPPER<T,DATA_DMRS_MUX_E::enable,DMRSTYPE,NROFCDMS>::demaper(
                            request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                            request->eModOrder, response->pLlr[nSymNum],
                            ftempPostSINR, ftempGain, avxxTxSymbol, nSc, nSCIdx, llr_range_low, llr_range_high,
                            nDmrsPortIdx);
                    }
                    LOG_TICK_END(PUSCH_MMSE_IRC_LLR_DEMAPER);

                    if (likely(nSc1 != 0))
                    {
                        LOG_TICK_START(PUSCH_MMSE_IRC_TX_CALC);
                        if constexpr (fp16Int16 == FP16_E::INT16){
                            tx_calc(pRxIn[nSymNum], request->nAgcGain, invMatrixMulHTansInvRnnRe2, invMatrixMulHTansInvRnnIm2,
                                    avxShift, avxxTxSymbol, FoOffsetTable[nSymNum]);
                        }
#if defined (_BBLIB_SPR_)
                        else {
                            tx_calc(pRxIn[nSymNum], invMatrixMulHTansInvRnnRe2,
                                    avxShift, avxxTxSymbol, avxAgcScale, FoOffsetTable[nSymNum]);
                        }
#endif
                        LOG_TICK_END(PUSCH_MMSE_IRC_TX_CALC);

                        LOG_TICK_START(PUSCH_MMSE_IRC_LLR_DEMAPER);

                        if (dataDmrsFlag == 0) {
                            DEMAPPER<T,DATA_DMRS_MUX_E::disable,DMRSTYPE,NROFCDMS>::demaper(
                                request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                                request->eModOrder, response->pLlr[nSymNum],
                                ftempPostSINR2, ftempGain2, avxxTxSymbol, nSc1, nSCIdx + 16, llr_range_low, llr_range_high,
                                nDmrsPortIdx);
                        } else {
                            #pragma unroll(N_TX)
                            for(int32_t i = 0; i < N_TX; i++){
                                ftempGaintmp2[i] = ftempGain2[i];
                                ftempPostSINRtmp2[i] = ftempPostSINR2[i];
                            }
                            DEMAPPER<T,DATA_DMRS_MUX_E::enable,DMRSTYPE,NROFCDMS>::demaper(
                                request->nUeInGroup,request->pLayerNumPerUE,request->nLayerInGroup,
                                request->eModOrder, response->pLlr[nSymNum],
                                ftempPostSINRtmp2, ftempGaintmp2, avxxTxSymbol, nSc1, nSCIdx + 16, llr_range_low, llr_range_high,
                                nDmrsPortIdx);
                        }

                        LOG_TICK_END(PUSCH_MMSE_IRC_LLR_DEMAPER);
                    }
                }//for iSymb

                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        pChIn[0][j][i] = pChIn[0][j][i] + 2;
                        pChIn[1][j][i] = pChIn[1][j][i] + 2;
                    }
                }
            }//end the sc loop
            if (nTime <= 3) {
                for (size_t j = 0; j < N_TX; j++)
                {
                    response->fPostSINR[j] = reduce_add(fsumPostSINR[j]) / (nSubCarrier / 2);
                }
            } else {
                for (size_t j = 0; j < N_TX; j++)
                {
                    response->fPostSINR[j] = *(reinterpret_cast<float *>(&fsumPostSINR[j])) / nTime0;
                }
            }
        }
        LOG_TICK_REPORT(response, PUSCH_MMSE_IRC_SUBMODULE_MAX)
    }// #endif

    static void mimo_mmse_llr_avx512_common(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response *response) {
        if constexpr (interp_flag == INTERP_E::enable) {
            mimo_mmse_llr_avx512_lin_interp(request,response);}
        else{
            mimo_mmse_llr_avx512(request,response);}
    }
};

template<typename T, FO_E fo_flag, size_t N_RX = 16, size_t N_TX = 16, INTERP_E interp_flag = INTERP_E::disable>
inline void mimo_mmse_llr_avx512_interp_dmrstype(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
{

    if ((request->nDMRSType == 1) && (request->nNrOfCDMs == 2))
        IRCSymbolProcess<T, fo_flag, N_RX, N_TX,1,2,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else if((request->nDMRSType == 1) && (request->nNrOfCDMs == 1))
        IRCSymbolProcess<T, fo_flag, N_RX, N_TX,1,1,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else if((request->nDMRSType == 2) && (request->nNrOfCDMs == 3))
        IRCSymbolProcess<T, fo_flag, N_RX, N_TX,2,3,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else if((request->nDMRSType == 2) && (request->nNrOfCDMs == 2))
        IRCSymbolProcess<T, fo_flag, N_RX, N_TX,2,2,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else
        IRCSymbolProcess<T, fo_flag, N_RX, N_TX,2,1,interp_flag>::mimo_mmse_llr_avx512_common(request, response);

}

template<size_t N_RX = 16, size_t N_TX = 16, typename T = CI16vec16>
void mimo_mmse_llr_avx512_interp(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
{
    if ((request->nLinInterpEnable == 1) &&
        (request->nDmrsChSymb != 1)) {
            if(request->nEnableFoComp == 1){
                mimo_mmse_llr_avx512_interp_dmrstype<T, FO_E::enable, N_RX, N_TX, INTERP_E::enable>(request, response);
            }else{
                mimo_mmse_llr_avx512_interp_dmrstype<T, FO_E::disable, N_RX, N_TX, INTERP_E::enable>(request, response);
            }
    } else {
        if(request->nEnableFoComp == 1){
            mimo_mmse_llr_avx512_interp_dmrstype<T, FO_E::enable, N_RX, N_TX, INTERP_E::disable>(request, response);
        }else{
            mimo_mmse_llr_avx512_interp_dmrstype<T, FO_E::disable, N_RX, N_TX, INTERP_E::disable>(request, response);
        }
    }
    return;
}


/*! \brief PUSCH symbol processing, include MMSE MIMO detection,layer demap and LLR demap.
    \param [in] request Input request structure for PUSCH symbol processing.
    \param [out] response Output response structure for PUSCH symbol processing.
    \return 0 for success, and -1 for error
*/
template<typename T = CI16vec16>
int32_t bblib_pusch_irc_symbol_processing_detection(
    bblib_pusch_symbol_processing_request *request,
    bblib_pusch_symbol_processing_response *response)
{
    uint16_t nLayer = request->nLayerInGroup;
    uint16_t nRxAnt = request->nRxAnt;
    // uint16_t nUeInGroup = request->nUeInGroup;
    // uint16_t nLayerPerUE = request->nLayerPerUE;
    int32_t n_return = -1;

    if(unlikely(0==request->nSubCarrier)) {
        printf("bblib_pusch_symbol_processing_avx512: Error! nSubCarrier == 0\n");
        return n_return;
    }

        if (4 == nRxAnt ){
            mimo_mmse_llr_avx512_interp<4, 4, T>(request, response);
            n_return = 0;
        } else if(8 == nRxAnt){
            mimo_mmse_llr_avx512_interp<8, 4, T>(request, response);
            n_return = 0;
        } else if(16 == nRxAnt){
            mimo_mmse_llr_avx512_interp<16, 4, T>(request, response);
            n_return = 0;
        }
#if 0
    if (1 == nLayer) {
        if (1 == nRxAnt ){
            mimo_mmse_llr_avx512_interp<1, 1, T>(request, response);
            n_return = 0;
        } else if(2 == nRxAnt){
            mimo_mmse_llr_avx512_interp<2, 1, T>(request, response);
            n_return = 0;
        } else if(4 == nRxAnt){
            mimo_mmse_llr_avx512_interp<4, 1, T>(request, response);
            n_return = 0;
        } else if(8 == nRxAnt){
            mimo_mmse_llr_avx512_interp<8, 1, T>(request, response);
            n_return = 0;
        } else if(16 == nRxAnt){
            mimo_mmse_llr_avx512_interp<16, 1, T>(request, response);
            n_return = 0;
        }
    } else if(2 == nLayer) {
        if (2 == nRxAnt ){
            mimo_mmse_llr_avx512_interp<2, 2, T>(request, response);
            n_return = 0;
        } else if(4 == nRxAnt){
            mimo_mmse_llr_avx512_interp<4, 2, T>(request, response);
            n_return = 0;
        } else if(8 == nRxAnt){
            mimo_mmse_llr_avx512_interp<8, 2, T>(request, response);
            n_return = 0;
        } else if(16 == nRxAnt){
            mimo_mmse_llr_avx512_interp<16, 2, T>(request, response);
            n_return = 0;
        }
    } else if(4 == nLayer) {
        if (4 == nRxAnt ){
            mimo_mmse_llr_avx512_interp<4, 4, T>(request, response);
            n_return = 0;
        } else if(8 == nRxAnt){
            mimo_mmse_llr_avx512_interp<8, 4, T>(request, response);
            n_return = 0;
        } else if(16 == nRxAnt){
            mimo_mmse_llr_avx512_interp<16, 4, T>(request, response);
            n_return = 0;
        }
    } else if(8 == nLayer) {
        if (8 == nRxAnt ){
            mimo_mmse_llr_avx512_interp<8, 8, T>(request, response);
            n_return = 0;
        } else if(16 == nRxAnt){
            mimo_mmse_llr_avx512_interp<16, 8, T>(request, response);
            n_return = 0;
        }
    }
#endif
    if (n_return) {
        printf("bblib_irc_pusch_symbol_processing_avx512: Error! don't support this combination -> nLayer = %d, nRxAnt = %d, nTpFlag = %d\n",
            nLayer, nRxAnt, request->nTpFlag);
    }

    return (n_return);
}


/*! \brief PUSCH symbol processing detection, with LLR soft bits calculation. With AVX512 intrinsics.
    \param [in] request Input request structure for PUSCH symbol processing.
    \param [out] response Output response structure for PUSCH symbol processing..
    \return 0 for success, and -1 for error
*/
int32_t bblib_pusch_irc_symbol_processing_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response) {
    LOG_TICK_INSTANCE(response);
    return bblib_pusch_irc_symbol_processing_detection<CI16vec16>(request, response);

}
#ifdef _BBLIB_SPR_
int32_t bblib_pusch_irc_symbol_processing_avx512_5gisa(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response) {
    LOG_TICK_INSTANCE(response);
    return bblib_pusch_irc_symbol_processing_detection<CF16vec16>(request, response);
}
#endif
#endif
