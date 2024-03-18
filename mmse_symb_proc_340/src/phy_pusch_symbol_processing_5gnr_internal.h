/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_pusch_symbol_processing_5gnr_internal.h
    \brief  Internal API for MMSE MIMO detection, with post SINR calculation.
*/

/*******************************************************************************
* Include public/global header files
********************************************************************************/
#ifndef _PHY_PUSCH_SYMBOL_PROCESSING_INTERNAL_H_
#define _PHY_PUSCH_SYMBOL_PROCESSING_INTERNAL_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX
//#include "phy_dft_idft.h"
#include "phy_rx_mimo_mmse.h"
#include "phy_matrix_inv_cholesky.h"
#include "simd_insts.hpp"
#include "dvec_int16.hpp"
#include "demapper.hpp"

#define MAX_TP_SC_TIME 205 //3276/16=205
#define MAX_IDFT_SIZE 3276
#ifdef _BBLIB_AVX512_


#ifdef __cplusplus
#include <iostream>
#pragma once
using namespace std;
#endif
using namespace W_SDK;

/* multiply complex: (A + iB)*(C+iD) = AC-BD + i(AD+BC) */
 #define COMPLEX_MULT_512(input0, input1, outPtr) \
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
    _mm512_storeu_si512((__m512i *) outPtr, result); \
 }

//Copy the sign of the a values into the b values.
static FORCE_INLINE inline __m512i copy_sign_epi16(__m512i a, __m512i b)
{
  return _mm512_mask_sub_epi16(b, _mm512_movepi16_mask(a),
                                    _mm512_setzero_si512(), b);
}

//Copy the inverted sign of the a values into the b values.
static FORCE_INLINE inline __m512i copy_inverted_sign_epi16(__m512i a, __m512i b)
{
  return _mm512_mask_sub_epi16(b, _mm512_movepi16_mask(_mm512_sub_epi16(_mm512_setzero_si512(),a)),
                                    _mm512_setzero_si512(), b);
}

static __mmask32  nMaskNegQ = 0x55555555;
// static __mmask32  nMaskNegI = 0xaaaaaaaa;
/* multiply complex: (A + iB)*(C+iD) = AC-BD + i(AD+BC) */
#define COMPLEX_MULT_AVX512(input0,input1, outPtr) \
{ \
    /* select real or image part from a complex value */ \
    __m512i ReRe = _mm512_shuffle_epi8(input0, m512_sw_r); \
    __m512i ImIm = _mm512_shuffle_epi8(input0, m512_sw_i); \
    /* swap real or image part and negative image part from a complex value */ \
    /* switch IQ */ \
    __m512i tmp1 =  _mm512_rol_epi32(input1,16);/* t1,t0,t3,t2,t5,t4,t7,t6 */ \
    /* negative the Q part */ \
    __m512i negImPosRe = _mm512_mask_sub_epi16(tmp1, nMaskNegQ, _mm512_setzero_si512(), tmp1); /* -t1,t0,-t3,t2,-t5,t4,-t7,t6 */ \
    /* multiply complex */ \
    tmp1 = _mm512_mulhrs_epi16(ReRe, input1); \
    __m512i tmp2 = _mm512_mulhrs_epi16(ImIm, negImPosRe); \
    __m512i result = _mm512_adds_epi16(tmp1, tmp2); \
    _mm512_storeu_si512((__m512i *) outPtr, result); \
}

template<int N = 16>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[4][4], F32vec16 matBIm[4][4],
    F32vec16 matInvBRe[4][4], F32vec16 matInvBIm[4][4]) {
    #define type_cast reinterpret_cast<__m512 (*)[4]>
    matrix_inv_lemma_4x4(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm),
        BBLIB_MMSE_LEMMA_SCALING);
    #undef type_cast
}

inline FORCE_INLINE
void agc_shift(CI16vec16& xmmY, int16_t nShift)
{
    int32_t nBitShift = 0;
    if (unlikely(nShift > 0)) {
        nBitShift = nShift;
        xmmY = _mm512_srai_epi16(xmmY, nBitShift);
    }
    else if (nShift < 0) {
        nBitShift = -nShift;
        xmmY = _mm512_slli_epi16(xmmY, nBitShift);
    }
}

inline FORCE_INLINE
void agc_shift_overflow_detection(CI16vec16& xmmY, int16_t nShift, uint32_t & nOverflow)
{
    int32_t nBitShift = 0;
    Is16vec32 signMask = _mm512_set1_epi16(0x8000);
    Is16vec32 xmmYtmp, signY, signShiftY, signFlip;
    Is32vec16 xmmOverflow;
    Is32vec16 signLowMask = _mm512_set1_epi32(0x8000);
    Is32vec16 signHighMask = _mm512_set1_epi32(0x80000000);
    xmmYtmp = xmmY;
    signY = _mm512_and_si512(xmmY, signMask);
    if (nShift > 0) {
        nBitShift = nShift;
        xmmY = _mm512_srai_epi16(xmmY, nBitShift);
    }
    else if (nShift < 0) {
        nBitShift = -nShift;
        xmmY = _mm512_slli_epi16(xmmY, nBitShift);
    }
    signShiftY = _mm512_and_si512(xmmY, signMask);
    signFlip = _mm512_xor_si512(signY, signShiftY);
    //xmmOverflow = _mm512_popcnt_epi32(signFlip);
    xmmOverflow = _mm512_and_si512(signFlip, signLowMask);
    xmmOverflow = _mm512_srli_epi32(xmmOverflow, 15);
    nOverflow += _mm512_reduce_add_epi32(xmmOverflow);
    xmmOverflow = _mm512_and_si512(signFlip, signHighMask);
    xmmOverflow = _mm512_srli_epi32(xmmOverflow, 31);
    nOverflow += _mm512_reduce_add_epi32(xmmOverflow);
}

#endif
#endif
