/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_rx_mimo_mmse_internal.h
    \brief  Internal API for MMSE MIMO detection, with post SINR calculation.
*/

/*******************************************************************************
* Include public/global header files
********************************************************************************/
#ifndef _PHY_RX_MIMO_MMSE_INTERNAL_H_
#define _PHY_RX_MIMO_MMSE_INTERNAL_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX
#include "bblib_common_const.h"
#include "phy_matrix_inv_cholesky.h"
#include "simd_insts.hpp"
#ifdef _BBLIB_SPR_
#include "matrix.hpp"
#endif

#ifdef _BBLIB_AVX512_
#ifdef __cplusplus
#include <iostream>
#pragma once
using namespace std;
#endif
using namespace W_SDK;


void matrix_inv_block_8x8(__m512 ps_B_r[8][8], __m512 ps_B_i[8][8], __m512 ps_D_r[8][8], __m512 ps_D_i[8][8]);


static const __m512i m512_sw_r = _mm512_set_epi8(13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0, \
                                           13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0, \
                                           13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0, \
                                           13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0);

static const __m512i m512_sw_i = _mm512_set_epi8(15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2, \
                                           15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2, \
                                           15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2, \
                                           15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2);

static const __m512i m512IQ_switch = _mm512_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                              13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                              13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                              13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2);

static const __m512i m512Neg_Q = _mm512_set_epi16(0xffff,0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001,  \
                                             0xffff,0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001);


static const __m512i m512Neg_I = _mm512_set_epi16(0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, \
                                            0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff, 0x0001,0xffff, 0x0001, 0xffff);

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
    _mm512_store_si512((__m512i *) outPtr, result); \
 }

template<typename SIMD> struct DataType;
#ifdef _BBLIB_SPR_
template<> struct DataType<CF16vec16> {
    using FloatType = CF16vec16;
    using Float = float16; 
};
#endif
template<> struct DataType<Is16vec32> {
    using FloatType = F32vec16; 
    using Float = float; 
};

#endif
#endif