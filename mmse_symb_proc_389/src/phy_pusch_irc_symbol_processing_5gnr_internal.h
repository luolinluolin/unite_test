/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_pusch_irc_symbol_processing_internal.h
    \brief  Internal API for pusch irc symbol processing.
*/

/*******************************************************************************
* Include public/global header files
********************************************************************************/
#ifndef _PHY_PUSCH_IRC_SYMBOL_PROCESSING_INTERNAL_H_
#define _PHY_PUSCH_IRC_SYMBOL_PROCESSING_INTERNAL_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX
#include "bblib_common_const.h"
#include "phy_rx_mimo_mmse.h"
#include "phy_matrix_inv_cholesky.h"
#include "simd_insts.hpp"
#include "demapper.hpp"

#ifdef _BBLIB_AVX512_

#ifdef __cplusplus
#include <iostream>
#pragma once
using namespace std;
#endif
using namespace W_SDK;

#define N_TX_1 1
#define N_TX_2 2
#define N_RX_2 2
#define N_RX_4 4
#define N_RX_16 16
#define N_TX_4 4
#define N_RX_8 8
#define N_TX_8 8
#define MAX_SC_TIME 205 //3276/16=205
#define MAX_TP_SC_TIME 75 //1200/16=75
#define POSTSNR_FXP_BASE 4 //at least 16S4 to ensure the output LLR not loose any integer digit

#define MAX_IDFT_SIZE 1200

//#define SNR_SMOOTH //to enable the postSINR averaging to smooth the fxp quantized error

/* Threshold of the MMSE gain, to eiminate the error in _mm512_rcp14_ps, which has 10^-5 precision. */
static const __m512 m_gain_threshold = _mm512_set1_ps(0.9999);  // 0.9999

/* Constants that needed in LLR demapping */
static const __m512i qpsk_data_dmrs_inter_type1_port01_idx = _mm512_set_epi16(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
static const __m512i qpsk_data_dmrs_inter_type1_port23_idx = _mm512_set_epi16(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);

static const __m512i qam16_data_dmrs_inter_type1_port01_idx = _mm512_set_epi32(14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1);
static const __m512i qam16_data_dmrs_inter_type1_port23_idx = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);

template<int N = 16>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[N][N], F32vec16 matBIm[N][N],
    F32vec16 matInvBRe[N][N], F32vec16 matInvBIm[N][N]);

template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[16][16], F32vec16 matBIm[16][16],
    F32vec16 matInvBRe[16][16], F32vec16 matInvBIm[16][16]) {
    #define type_cast reinterpret_cast<__m512 (*)[16]>
    matrix_inv_cholesky_16x16(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[8][8], F32vec16 matBIm[8][8],
    F32vec16 matInvBRe[8][8], F32vec16 matInvBIm[8][8]) {
    #define type_cast reinterpret_cast<__m512 (*)[8]>
    matrix_inv_cholesky_8x8(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[7][7], F32vec16 matBIm[7][7],
    F32vec16 matInvBRe[7][7], F32vec16 matInvBIm[7][7]) {
    #define type_cast reinterpret_cast<__m512 (*)[7]>
    matrix_inv_cholesky_7x7(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[6][6], F32vec16 matBIm[6][6],
    F32vec16 matInvBRe[6][6], F32vec16 matInvBIm[6][6]) {
    #define type_cast reinterpret_cast<__m512 (*)[6]>
    matrix_inv_cholesky_6x6(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[5][5], F32vec16 matBIm[5][5],
    F32vec16 matInvBRe[5][5], F32vec16 matInvBIm[5][5]) {
    #define type_cast reinterpret_cast<__m512 (*)[5]>
    matrix_inv_cholesky_5x5(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[4][4], F32vec16 matBIm[4][4],
    F32vec16 matInvBRe[4][4], F32vec16 matInvBIm[4][4]) {
    #define type_cast reinterpret_cast<__m512 (*)[4]>
    matrix_inv_cholesky_4x4(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[3][3], F32vec16 matBIm[3][3],
    F32vec16 matInvBRe[3][3], F32vec16 matInvBIm[3][3]) {
    #define type_cast reinterpret_cast<__m512 (*)[3]>
    matrix_inv_cholesky_3x3(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[2][2], F32vec16 matBIm[2][2],
    F32vec16 matInvBRe[2][2], F32vec16 matInvBIm[2][2]) {
    /*
    #define type_cast reinterpret_cast<__m512 (*)[2]>
    matrix_inv_cholesky_2x2(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
    */
    // 2. invA = inv(H' * H + Sigma2*I), 1x1 matrix inversion
    matInvBRe[0][0] = matBRe[1][1];
    matInvBRe[0][1] = _mm512_sub_ps(F32vec16(0.0), matBRe[1][0]);
    matInvBRe[1][0] = _mm512_sub_ps(F32vec16(0.0), matBRe[0][1]);
    matInvBRe[1][1] = matBRe[0][0];

    matInvBIm[0][1] = matBIm[1][0];
    matInvBIm[1][0] = matBIm[0][1];

    // 2) calculate the determinant of A, det(A) = a00*a11 - a01*a10;
    auto avxfdetARe = _mm512_mul_ps(matBRe[0][0], matBRe[1][1]);
    avxfdetARe = _mm512_fnmadd_ps(matBRe[0][1], matBRe[0][1], avxfdetARe);
    avxfdetARe = _mm512_fnmadd_ps(matBIm[0][1], matBIm[0][1], avxfdetARe);

    // 3) detA = 1 / detA
    avxfdetARe = _mm512_rcp14_ps(avxfdetARe);

    // 4) invA = (A*) * detA
    matInvBRe[0][0] = _mm512_mul_ps(matInvBRe[0][0], avxfdetARe);
    matInvBRe[0][1] = _mm512_mul_ps(matInvBRe[0][1], avxfdetARe);
    matInvBRe[1][0] = _mm512_mul_ps(matInvBRe[1][0], avxfdetARe);
    matInvBRe[1][1] = _mm512_mul_ps(matInvBRe[1][1], avxfdetARe);

    matInvBIm[0][0] = F32vec16(0.0);
    matInvBIm[1][1] = F32vec16(0.0);
    matInvBIm[0][1] = _mm512_mul_ps(matInvBIm[0][1], avxfdetARe);
    matInvBIm[1][0] = _mm512_mul_ps(matInvBIm[1][0], avxfdetARe);
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[1][1], F32vec16 matBIm[1][1],
    F32vec16 matInvBRe[1][1], F32vec16 matInvBIm[1][1]) {
        // auto matABS = matBRe[0][0] * matBRe[0][0] + matBIm[0][0] * matBIm[0][0];
        // auto matRcp = rcp(matABS);
        // matInvBRe[0][0] = matBRe[0][0] * matRcp;
        // matInvBIm[0][0] = -matBIm[0][0] * matRcp;
        matInvBRe[0][0] = rcp(matBRe[0][0]);
        matInvBIm[0][0] = matBIm[0][0];
}


/* multiply complex: (A + iB)*(C+iD) = AC-BD + i(AD+BC) */
 #define COMPLEX_MULT_AVX512(input0, input1, outPtr) \
 { \
    __m512i ReRe, ImIm, negImPosRe; \
    __m512i tmp1, tmp2, result; \
    ReRe = _mm512_shuffle_epi8(input0, m512_sw_r); \
    ImIm = _mm512_shuffle_epi8(input0, m512_sw_i); \
    tmp1 = _mm512_shuffle_epi8(input1, m512IQ_switch); \
    negImPosRe = _mm512_mullo_epi16(tmp1, m512Neg_I); \
    \
    tmp1 = _mm512_mulhrs_epi16(ReRe, input1); \
    tmp2 = _mm512_mulhrs_epi16(ImIm, negImPosRe); \
    result = _mm512_adds_epi16(tmp1, tmp2); \
    _mm512_store_si512((__m512i *) outPtr, result); \
 }

//Copy the sign of the a values into the b values.
static inline __m512i copy_sign_epi16(__m512i a, __m512i b)
{
  return _mm512_mask_sub_epi16(b, _mm512_movepi16_mask(a),
                                    _mm512_setzero_si512(), b);
}

//Copy the inverted sign of the a values into the b values.
static inline __m512i copy_inverted_sign_epi16(__m512i a, __m512i b)
{
  return _mm512_mask_sub_epi16(b, _mm512_movepi16_mask(_mm512_sub_epi16(_mm512_setzero_si512(),a)),
                                    _mm512_setzero_si512(), b);
}

// return the a value if a>b, other wise return b
inline __m512 select_high_float(__m512 a, __m512 b)
{
    return _mm512_mask_mov_ps(b,_mm512_cmpnle_ps_mask(a,b),a);
}

inline FORCE_INLINE
void agc_shift(Is16vec32& xmmY, int16_t nShift)
{
    int32_t nBitShift = 0;
    if (unlikely(nShift > 0)) {
        nBitShift = nShift;
        xmmY = srai(xmmY, nBitShift);
    }
    else if (nShift < 0) {
        nBitShift = -nShift;
        xmmY = slli(xmmY, nBitShift);
    }
}
#endif
#endif
