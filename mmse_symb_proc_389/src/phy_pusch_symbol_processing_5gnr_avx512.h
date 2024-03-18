/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_pusch_symbol_processing_5gnr_avx512.h
    \brief  Macros used for MMSE MIMO detection, with post SINR calculation.
*/

/*******************************************************************************
* Include public/global header files
********************************************************************************/
#ifndef _PHY_PUSCH_SYMBOL_PROCESSING_AVX512_H_
#define _PHY_PUSCH_SYMBOL_PROCESSING_AVX512_H_
#ifdef _BBLIB_AVX512_
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h> // AVX
#include "simd_insts.hpp"
#include "phy_pusch_symbol_processing_5gnr_internal.h"


using namespace W_SDK;


#define llr_demap_1layer_dft_normal(modOrder)\
{\
    switch(modOrder)\
    {\
        case BBLIB_HALF_PI_BPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:*/\
            /*even idx: LLR0 = (real(In)+imag(In))*sqrt(2)/(1-beta);*/\
            /*odd idx:  LLR0 = (-real(In)+imag(In))*sqrt(2)/(1-beta)*/\
            \
            /*I0,Q0,I1,Q1,...I15,Q15 ---> x, I0,Q0,I1,Q1,...Q14,I15*/\
            avxtempARe[0] = _mm512_bslli_epi128(avxxTxSymbolIdft[nSCIdx],2);\
            /*if even: x,I0,Q0,-I1,Q1,I2,Q2,-I3,Q3...-I15*/\
            /*if odd:  x,-I0,Q0,I1,Q1,-I2,Q2,I3,Q3...I15*/\
            avxtempARe[0] = _mm512_mask_sub_epi16(avxtempARe[0],halfPiSubMask,alliZero,avxtempARe[0]);\
            /*if even: x+I0,I0+Q0,Q0+I1,-I1+Q1,Q1+I2,I2+Q2,Q2+I3,-I3+Q3,...-I15+Q15*/\
            /*if odd:  x+I0,-I0+Q0,Q0+I1,I1+Q1,Q1+I2,-I2+Q2,Q2+I3,I3+Q3,...I15+Q15*/\
            avxtempARe[0] = _mm512_adds_epi16(avxxTxSymbolIdft[nSCIdx],avxtempARe[0]);\
            \
            /*output: 16S13*(16S13*16Sx) -> 16S(nLlrFxpPoints)*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxtempARe[0],_mm512_mulhrs_epi16(xRe[0],i_p_1_m_sqrt2));\
            avxtempARe[0] = limit_to_saturated_range(avxtempARe[0], llr_range_low, llr_range_high);\
            \
            avxtempBRe[0] = _mm512_permutexvar_epi16(half_pi_select_first_eight,avxtempARe[0]);\
            avxtempCRe[0] = _mm512_permutexvar_epi16(half_pi_select_second_eight,avxtempARe[0]);\
            avxtempBRe[0] = _mm512_packs_epi16(avxtempBRe[0], avxtempCRe[0]);\
            _mm512_mask_storeu_epi8(*ppSoft,0xffff,avxtempBRe[0]);\
            \
            *ppSoft += 16;\
            break;\
        }\
        case BBLIB_QPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:*/\
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/\
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/\
            \
            /*output 16S13*16S13->16S11*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxxTxSymbolIdft[nSCIdx], i_p_2_m_sqrt2);\
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxtempARe[0],xRe[0]);\
            avxtempARe[0] = limit_to_saturated_range(avxtempARe[0], llr_range_low, llr_range_high);\
            \
            avxtempARe[0] = _mm512_packs_epi16(avxtempARe[0], avxtempARe[0]);\
            \
            avxtempARe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,avxtempARe[0]);\
            \
            _mm512_mask_storeu_epi16 (*ppSoft,0xffff,avxtempARe[0]);\
            \
            *ppSoft += 32;\
            break;\
        }\
        case BBLIB_QAM16:\
        {\
            /*permute the beta and postSNR as well*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm\
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta\
            \
            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta\
            \
            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0\
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0\
            \
            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0\
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/\
            \
            /*first calculate the threshold 2*beta/sqrt(10)*/\
            /*output: 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt10);\
            \
            /*calculate the factor 4/sqrt(10)/(1-beta)*/\
            /*output: 16S13*16Sx->16S(x-2)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt10);\
            \
            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/\
            avxtempCRe[0] = copy_inverted_sign_epi16(avxxTxSymbolIdft[nSCIdx],_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt10));\
            \
            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/\
            /*output: 16S13*16S13->16S11*/\
            avxtempAIm[0] = _mm512_adds_epi16(avxxTxSymbolIdft[nSCIdx],avxtempCRe[0]);\
            avxtempAIm[0] = _mm512_mulhrs_epi16(avxtempAIm[0],i_p_8_d_sqrt10);\
            avxtempCRe[0] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbolIdft[nSCIdx]),avxtempARe[0],avxtempAIm[0],_mm512_mulhrs_epi16(avxxTxSymbolIdft[nSCIdx],i_p_4_d_sqrt10));\
            \
            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],xRe[0]);\
            avxtempCRe[0] = limit_to_saturated_range(avxtempCRe[0], llr_range_low, llr_range_high);\
            \
            /*calculate LLR2 and LLR3*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempBIm[0] = _mm512_subs_epi16(avxtempARe[0],_mm512_abs_epi16(avxxTxSymbolIdft[nSCIdx]));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],avxtempBRe[0]);\
            avxtempBIm[0] = limit_to_saturated_range(avxtempBIm[0], llr_range_low, llr_range_high);\
            \
            /*pack to int8 and shuffle*/\
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/\
            \
            xRe[0] = _mm512_packs_epi16(avxtempCRe[0],avxtempBIm[0]);\
            \
            xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);\
            _mm512_storeu_si512 (*ppSoft, xRe[0]);\
            \
            *ppSoft = *ppSoft + 64;\
            break;\
        }\
        case BBLIB_QAM64:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)\
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)\
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)\
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)\
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0\
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)\
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)\
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)\
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)\
            \
            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(42)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt42);\
            \
            /*calculate the threshold 4*beta/sqrt(42)*/\
            avxtempAIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt42);\
            \
            /*calculate the threshold 6*beta/sqrt(42)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt42);\
            \
            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt42);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbolIdft[nSCIdx]);\
            \
            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_16_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/\
            avxtempCRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_12_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/\
            avxtempCIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt42));\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempCIm[0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));\
            \
            /*if abs(real(In))>4*beta/sqrt(42), return avxtempARe[0], otherwise return avxtempZIm[0]*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempDRe[0],avxtempZIm[0]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempDRe[0] = copy_sign_epi16(avxxTxSymbolIdft[nSCIdx],avxtempDRe[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDRe[0] = _mm512_mulhrs_epi16(avxtempDRe[0],xRe[0]);\
            avxtempDRe[0] = limit_to_saturated_range(avxtempDRe[0], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt42),xtemp1);\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_8_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCRe[0] = _mm512_subs_epi16(avxtempAIm[0],xtemp1);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_4_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42),xtemp1);\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempDIm[0],avxtempCIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],xRe[0]);\
            avxtempDIm[0] = limit_to_saturated_range(avxtempDIm[0], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/\
            /*output 16S13*/\
            xtemp2 = select_high_low_epi16(xtemp1,avxtempAIm[0],_mm512_subs_epi16(avxtempBRe[0],xtemp1),_mm512_subs_epi16(xtemp1,avxtempARe[0]));\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            xtemp2 = _mm512_mulhrs_epi16(xtemp2,avxtempZRe[0]);\
            xtemp2 = limit_to_saturated_range(xtemp2, llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempDRe[0],\
                                        avxtempDIm[0]);\
            xIm[0] = _mm512_packs_epi16(xtemp2,\
                                        alliZero);\
            \
            avxtempCRe[0] = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);\
            _mm512_storeu_si512 (*ppSoft, avxtempCRe[0]);\
            \
            avxtempCIm[0]= _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);\
            _mm512_mask_storeu_epi16(*ppSoft+64,0xffff,avxtempCIm[0]);\
            \
            *ppSoft = *ppSoft + 96;\
            \
            break;\
        }\
        default:\
        {\
            printf("\nInvalid modulation request, mod order %d\n",modOrder);\
            exit(-1);\
        }\
    }\
}

#define llr_demap_1layer_dft_tail(modOrder,pLlrStoreK)\
{\
    switch(modOrder)\
    {\
        case BBLIB_HALF_PI_BPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:*/\
            /*even idx: LLR0 = (real(In)+imag(In))*sqrt(2)/(1-beta);*/\
            /*odd idx:  LLR0 = (-real(In)+imag(In))*sqrt(2)/(1-beta)*/\
            \
            /*I0,Q0,I1,Q1,...I15,Q15 ---> x, I0,Q0,I1,Q1,...Q14,I15*/\
            avxtempARe[0] = _mm512_bslli_epi128(avxxTxSymbolIdft[nSCIdx],2);\
            /*if even: x,I0,Q0,-I1,Q1,I2,Q2,-I3,Q3...-I15*/\
            /*if odd:  x,-I0,Q0,I1,Q1,-I2,Q2,I3,Q3...I15*/\
            avxtempARe[0] = _mm512_mask_sub_epi16(avxtempARe[0],halfPiSubMask,alliZero,avxtempARe[0]);\
            /*if even: x+I0,I0+Q0,Q0+I1,-I1+Q1,Q1+I2,I2+Q2,Q2+I3,-I3+Q3,...-I15+Q15*/\
            /*if odd:  x+I0,-I0+Q0,Q0+I1,I1+Q1,Q1+I2,-I2+Q2,Q2+I3,I3+Q3,...I15+Q15*/\
            avxtempARe[0] = _mm512_adds_epi16(avxxTxSymbolIdft[nSCIdx],avxtempARe[0]);\
            /*output: 16S13*(16S13*16Sx) -> 16S(nLlrFxpPoints)*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxtempARe[0],_mm512_mulhrs_epi16(xRe[0],i_p_1_m_sqrt2));\
            avxtempARe[0] = limit_to_saturated_range(avxtempARe[0], llr_range_low, llr_range_high);\
            \
            avxtempBRe[0] = _mm512_permutexvar_epi16(half_pi_select_first_eight,avxtempARe[0]);\
            avxtempCRe[0] = _mm512_permutexvar_epi16(half_pi_select_second_eight,avxtempARe[0]);\
            avxtempBRe[0] = _mm512_packs_epi16(avxtempBRe[0], avxtempCRe[0]);\
            _mm512_mask_storeu_epi8(*ppSoft,nHalfPiLlrStoreKs,avxtempBRe[0]);\
            \
            *ppSoft += nRestLen;\
            break;\
        }\
        case BBLIB_QPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:*/\
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/\
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/\
            \
            /*output 16S13*16S13->16S11*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxxTxSymbolIdft[nSCIdx], i_p_2_m_sqrt2);\
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxtempARe[0],xRe[0]);\
            avxtempARe[0] = limit_to_saturated_range(avxtempARe[0], llr_range_low, llr_range_high);\
            \
            avxtempARe[0] = _mm512_packs_epi16(avxtempARe[0], avxtempARe[0]);\
            \
            avxtempARe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,avxtempARe[0]);\
            \
            _mm512_mask_storeu_epi16 (*ppSoft,pLlrStoreK[0],avxtempARe[0]);\
            \
            *ppSoft += nRestLen*2;\
            break;\
        }\
        case BBLIB_QAM16:\
        {\
            /*permute the beta and postSNR as well*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm\
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta\
            \
            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta\
            \
            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0\
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0\
            \
            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0\
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/\
            \
            /*first calculate the threshold 2*beta/sqrt(10)*/\
            /*output: 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt10);\
            \
            /*calculate the factor 4/sqrt(10)/(1-beta)*/\
            /*output: 16S13*16Sx->16S(x-2)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt10);\
            \
            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/\
            avxtempCRe[0] = copy_inverted_sign_epi16(avxxTxSymbolIdft[nSCIdx],_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt10));\
            \
            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/\
            /*output: 16S13*16S13->16S11*/\
            avxtempAIm[0] = _mm512_adds_epi16(avxxTxSymbolIdft[nSCIdx],avxtempCRe[0]);\
            avxtempAIm[0] = _mm512_mulhrs_epi16(avxtempAIm[0],i_p_8_d_sqrt10);\
            avxtempCRe[0] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbolIdft[nSCIdx]),avxtempARe[0],avxtempAIm[0],_mm512_mulhrs_epi16(avxxTxSymbolIdft[nSCIdx],i_p_4_d_sqrt10));\
            \
            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],xRe[0]);\
            avxtempCRe[0] = limit_to_saturated_range(avxtempCRe[0], llr_range_low, llr_range_high);\
            \
            /*calculate LLR2 and LLR3*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempBIm[0] = _mm512_subs_epi16(avxtempARe[0],_mm512_abs_epi16(avxxTxSymbolIdft[nSCIdx]));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],avxtempBRe[0]);\
            avxtempBIm[0] = limit_to_saturated_range(avxtempBIm[0], llr_range_low, llr_range_high);\
            \
            /*pack to int8 and shuffle*/\
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/\
            \
            xRe[0] = _mm512_packs_epi16(avxtempCRe[0],avxtempBIm[0]);\
            xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);\
            _mm512_mask_storeu_epi16 (*ppSoft,pLlrStoreK[0],xRe[0]);\
            \
            *ppSoft += nRestLen*4;\
            break;\
        }\
        case BBLIB_QAM64:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)\
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)\
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)\
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)\
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0\
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)\
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)\
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)\
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)\
            \
            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(42)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt42);\
            \
            /*calculate the threshold 4*beta/sqrt(42)*/\
            avxtempAIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt42);\
            \
            /*calculate the threshold 6*beta/sqrt(42)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt42);\
            \
            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt42);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbolIdft[nSCIdx]);\
            \
            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_16_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/\
            avxtempCRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_12_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/\
            avxtempCIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt42));\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempCIm[0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));\
            \
            /*if abs(real(In))>4*beta/sqrt(42), return avxtempARe[0], otherwise return avxtempZIm[0]*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempDRe[0],avxtempZIm[0]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempDRe[0] = copy_sign_epi16(avxxTxSymbolIdft[nSCIdx],avxtempDRe[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDRe[0] = _mm512_mulhrs_epi16(avxtempDRe[0],xRe[0]);\
            avxtempDRe[0] = limit_to_saturated_range(avxtempDRe[0], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt42),xtemp1);\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_8_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCRe[0] = _mm512_subs_epi16(avxtempAIm[0],xtemp1);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_4_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42),xtemp1);\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempDIm[0],avxtempCIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],xRe[0]);\
            avxtempDIm[0] = limit_to_saturated_range(avxtempDIm[0], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/\
            /*output 16S13*/\
            xtemp2 = select_high_low_epi16(xtemp1,avxtempAIm[0],_mm512_subs_epi16(avxtempBRe[0],xtemp1),_mm512_subs_epi16(xtemp1,avxtempARe[0]));\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            xtemp2 = _mm512_mulhrs_epi16(xtemp2,avxtempZRe[0]);\
            xtemp2 = limit_to_saturated_range(xtemp2, llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempDRe[0],\
                                        avxtempDIm[0]);\
            xIm[0] = _mm512_packs_epi16(xtemp2,\
                                        alliZero);\
            \
            avxtempCRe[0] = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);\
            _mm512_mask_storeu_epi16 (*ppSoft,pLlrStoreK[0], avxtempCRe[0]);\
            \
            avxtempCIm[0]= _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);\
            _mm512_mask_storeu_epi16(*ppSoft+64,pLlrStoreK[1],avxtempCIm[0]);\
            \
            *ppSoft = *ppSoft + nRestLen*6;\
            \
            break;\
        }\
        default:\
        {\
            printf("\nInvalid modulation request, mod order %d\n",modOrder);\
            exit(-1);\
        }\
    }\
}

#define llr_demap_1layer_normal_data_dmrs_type1_inter(modOrder, nDmrsPortIdx)\
{\
    /*Move DMRS symbols to the upper bytes by using permute.*/\
    /*Only store LLRs from actual data sub-carriers and ignore DMRS*/\
    if ((nDmrsPortIdx == 0) || (nDmrsPortIdx == 1))\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempPostSINR[0]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port01_idx, avxxTxSymbol[0]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempGain[0]);\
        }\
    }\
    else\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempPostSINR[0]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port23_idx, avxxTxSymbol[0]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempGain[0]);\
        }\
    }\
    \
    switch(modOrder)\
    {\
        case BBLIB_QPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:*/\
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/\
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/\
            \
            /*output 16S13*16S13->16S11*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxxTxSymbol_inter[0], i_p_2_m_sqrt2);\
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxtempARe[0],xRe[0]);\
            avxtempARe[0] = limit_to_saturated_range(avxtempARe[0], llr_range_low, llr_range_high);\
            \
            avxtempARe[0] = _mm512_packs_epi16(avxtempARe[0], avxtempARe[0]);\
            \
            avxtempARe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,avxtempARe[0]);\
            \
            _mm512_mask_storeu_epi16 (ppSoft,0x00ff,avxtempARe[0]);\
            \
            ppSoft += 16;\
            break;\
        }\
        case BBLIB_QAM16:\
        {\
            /*permute the beta and postSNR as well*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm\
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta\
            \
            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta\
            \
            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0\
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0\
            \
            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0\
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/\
            \
            /*first calculate the threshold 2*beta/sqrt(10)*/\
            /*output: 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt10);\
            \
            /*calculate the factor 4/sqrt(10)/(1-beta)*/\
            /*output: 16S13*16Sx->16S(x-2)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt10);\
            \
            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/\
            avxtempCRe[0] = copy_inverted_sign_epi16(avxxTxSymbol_inter[0],_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt10));\
            \
            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/\
            /*output: 16S13*16S13->16S11*/\
            avxtempAIm[0] = _mm512_adds_epi16(avxxTxSymbol_inter[0],avxtempCRe[0]);\
            avxtempAIm[0] = _mm512_mulhrs_epi16(avxtempAIm[0],i_p_8_d_sqrt10);\
            avxtempCRe[0] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol_inter[0]),avxtempARe[0],avxtempAIm[0],_mm512_mulhrs_epi16(avxxTxSymbol_inter[0],i_p_4_d_sqrt10));\
            \
            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],xRe[0]);\
            avxtempCRe[0] = limit_to_saturated_range(avxtempCRe[0], llr_range_low, llr_range_high);\
            \
            /*calculate LLR2 and LLR3*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempBIm[0] = _mm512_subs_epi16(avxtempARe[0],_mm512_abs_epi16(avxxTxSymbol_inter[0]));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],avxtempBRe[0]);\
            avxtempBIm[0] = limit_to_saturated_range(avxtempBIm[0], llr_range_low, llr_range_high);\
            \
            /*pack to int8 and shuffle*/\
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/\
            \
            xRe[0] = _mm512_packs_epi16(avxtempCRe[0],avxtempBIm[0]);\
            \
            xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);\
            _mm512_mask_storeu_epi16 (ppSoft,0xffff,xRe[0]);\
            \
            ppSoft = ppSoft + 32;\
            break;\
        }\
        case BBLIB_QAM64:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)\
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)\
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)\
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)\
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0\
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)\
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)\
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)\
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)\
            \
            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(42)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt42);\
            \
            /*calculate the threshold 4*beta/sqrt(42)*/\
            avxtempAIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt42);\
            \
            /*calculate the threshold 6*beta/sqrt(42)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt42);\
            \
            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt42);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            \
            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_16_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/\
            avxtempCRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_12_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/\
            avxtempCIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt42));\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempCIm[0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));\
            \
            /*if abs(real(In))>4*beta/sqrt(42), return avxtempARe[0], otherwise return avxtempZIm[0]*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempDRe[0],avxtempZIm[0]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempDRe[0] = copy_sign_epi16(avxxTxSymbol_inter[0],avxtempDRe[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDRe[0] = _mm512_mulhrs_epi16(avxtempDRe[0],xRe[0]);\
            avxtempDRe[0] = limit_to_saturated_range(avxtempDRe[0], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt42),xtemp1);\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_8_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCRe[0] = _mm512_subs_epi16(avxtempAIm[0],xtemp1);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_4_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42),xtemp1);\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempDIm[0],avxtempCIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],xRe[0]);\
            avxtempDIm[0] = limit_to_saturated_range(avxtempDIm[0], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/\
            /*output 16S13*/\
            xtemp2 = select_high_low_epi16(xtemp1,avxtempAIm[0],_mm512_subs_epi16(avxtempBRe[0],xtemp1),_mm512_subs_epi16(xtemp1,avxtempARe[0]));\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            xtemp2 = _mm512_mulhrs_epi16(xtemp2,avxtempZRe[0]);\
            xtemp2 = limit_to_saturated_range(xtemp2, llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempDRe[0],\
                                        avxtempDIm[0]);\
            xIm[0] = _mm512_packs_epi16(xtemp2,\
                                        alliZero);\
            \
            avxtempCRe[0] = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);\
            _mm512_mask_storeu_epi16(ppSoft, 0xffffff, avxtempCRe[0]);\
            \
            ppSoft = ppSoft + 48;\
            \
            break;\
        }\
        case BBLIB_QAM256:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 32/sqrt(170)*(real(In)+InnerFactor0) when abs(real(In))>14*beta/sqrt(170)\
            InnerCompoundReal0_1 = 28/sqrt(170)*(real(In)+InnerFactor1) when 14*beta/sqrt(170)<abs(real(In))<=12*beta/sqrt(170)\
            InnerCompoundReal0_2 = 24/sqrt(170)*(real(In)r+InnerFactor2) when 12*beta/sqrt(170)<abs(real(In))<=10*beta/sqrt(170)\
            InnerCompoundReal0_3 = 20/sqrt(170)*(real(In)r+InnerFactor3) when 10*beta/sqrt(170)<abs(real(In))<=8*beta/sqrt(170)\
            InnerCompoundReal0_4 = 16/sqrt(170)*(real(In)r+InnerFactor4) when 8*beta/sqrt(170)<abs(real(In))<=6*beta/sqrt(170)\
            InnerCompoundReal0_5 = 12/sqrt(170)*(real(In)r+InnerFactor5) when 6*beta/sqrt(170)<abs(real(In))<=4*beta/sqrt(170)\
            InnerCompoundReal0_6 = 8/sqrt(170)*(real(In)r+InnerFactor6) when 4*beta/sqrt(170)<abs(real(In))<=2*beta/sqrt(170)\
            InnerCompoundReal0_7 = 4/sqrt(170)*real(In) when abs(real(In))<=2*beta/sqrt(170)\
            InnerFactor0 = 7*beta/(sqrt(170)); InnerFactor1 = 6*beta/(sqrt(170));InnerFactor2 = 5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = 4*beta/(sqrt(170)); InnerFactor4 = 3*beta/(sqrt(170));InnerFactor5 = 2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = beta/(sqrt(170)), when real(In)<0\
            InnerFactor0 = -7*beta/(sqrt(170)); InnerFactor1 = -6*beta/(sqrt(170));InnerFactor2 = -5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = -4*beta/(sqrt(170)); InnerFactor4 = -3*beta/(sqrt(170));InnerFactor5 = -2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = -beta/(sqrt(170)), when real(In)<0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 16/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170)) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal1_1 = 12/sqrt(170)*(-abs(real(r))+10*beta/sqrt(170)) when 12*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal1_2 = 8/sqrt(170)*(-abs(real(r))+9*beta/sqrt(170)) when 10*beta/sqrt(170)<=abs(real(r))<=12*beta/sqrt(170)\
            InnerCompoundReal1_3 = 4/sqrt(170)*(-abs(real(r))+8*beta/sqrt(170)) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal1_4 = 8/sqrt(170)*(-abs(real(r))+7*beta/sqrt(170)) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal1_5 = 12/sqrt(170)*(-abs(real(r))+6*beta/sqrt(170)) when 4*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal1_6 = 16/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170)) when 2*beta/sqrt(170)<=abs(real(r))\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2_0 = 8/sqrt(170)*(-abs(real(r))+13*beta/sqrt(170) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal2_1 = 4/sqrt(170)*(-abs(real(r))+12*beta/sqrt(170) when 10*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal2_2 = 8/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal2_3 = -8/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal2_4 = -4/sqrt(170)*(-abs(real(r))+4*beta/sqrt(170) when 2*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal2_5 = -8/sqrt(170)*(-abs(real(r))+3*beta/sqrt(170) when 2*beta/sqrt(170)>=abs(real(r))\
            \
            LLR5 = InnerCompoundImag2/(1-beta)\
            \
            LLR6 = 4/sqrt(170)*InnerCompoundReal3/(1-beta)\
            InnerCompoundReal3_0 = -1*abs(real(r))+14*beta/sqrt(170) when abs(real(r))>=12*beta/sqrt(42)\
            InnerCompoundReal3_1 = -1*abs(real(r))+10*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<12*beta/sqrt(42)\
            InnerCompoundReal3_2 = -1*abs(real(r))+6*beta/sqrt(170) when 4*beta/sqrt(170)<=abs(real(r))<8*beta/sqrt(42)\
            InnerCompoundReal3_3 = -1*abs(real(r))+2*beta/sqrt(170) when abs(real(r))<4*beta/sqrt(170)\
            \
            LLR7 = 4/sqrt(170)*InnerCompoundImag3/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(170)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt170);\
            \
            /*calculate the threshold 4*beta/sqrt(170)*/\
            avxtempAIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt170);\
            \
            /*calculate the threshold 6*beta/sqrt(170)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt170);\
            \
            /*calculate the threshold 8*beta/sqrt(170)*/\
            avxtempBIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the threshold 10*beta/sqrt(170)*/\
            avxtempCRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_10_d_sqrt170);\
            \
            /*calculate the threshold 12*beta/sqrt(170)*/\
            avxtempCIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_12_d_sqrt170);\
            \
            /*calculate the threshold 14*beta/sqrt(170)*/\
            avxtempDRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_14_d_sqrt170);\
            \
            /*calculate the common scaling factor 4/sqrt(170)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt170);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            \
            /*InnerCompoundReal0_0 = 32/sqrt(170)*(abs(real(In))-7*beta/sqrt(170))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170));\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],i_p_32_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_1: 28/sqrt(170)*(abs(real(In))- 6*beta/sqrt(170))*/\
            avxtempERe[0] = _mm512_subs_epi16(xtemp1,avxtempBRe[0]);\
            avxtempERe[0] = _mm512_mulhrs_epi16(avxtempERe[0],i_p_28_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_2: 24/sqrt(170)*(abs(real(In))-5*beta/sqrt(170))*/\
            avxtempEIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempEIm[0] = _mm512_mulhrs_epi16(avxtempEIm[0],i_p_24_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_3: 20/sqrt(170)*(abs(real(In))-4*beta/sqrt(170))*/\
            avxtempFRe[0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0]);\
            avxtempFRe[0] = _mm512_mulhrs_epi16(avxtempFRe[0],i_p_20_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_4: 16/sqrt(170)*(abs(real(In))-3*beta/sqrt(170))*/\
            avxtempFIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempFIm[0] = _mm512_mulhrs_epi16(avxtempFIm[0],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_5: 12/sqrt(170)*(abs(real(In))-2*beta/sqrt(170))*/\
            avxtempGRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            avxtempGRe[0] = _mm512_mulhrs_epi16(avxtempGRe[0],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_6: 8/sqrt(170)*(abs(real(In))-1*beta/sqrt(170))*/\
            avxtempGIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt170));\
            avxtempGIm[0] = _mm512_mulhrs_epi16(avxtempGIm[0],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempDRe[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal0_2, otherwise return InnerCompoundReal_0_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_0_1*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempCIm[0],avxtemp1,avxtempZIm[0]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal0_4, otherwise return InnerCompoundReal_0_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempFIm[0],avxtempGRe[0]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempARe[0], otherwise return avxtempBRe[0]*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempBIm[0],avxtemp1,avxtempHRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal0_6, otherwise return 4/sqrt(170)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempGIm[0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt170));\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return avxtempARe[0], otherwise return avxtempBRe[0]*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtemp1,avxtempZIm[0]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtemp1 = copy_sign_epi16(avxxTxSymbol_inter[0],avxtemp1);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtemp1 = _mm512_mulhrs_epi16(avxtemp1,xRe[0]);\
            avxtemp1 = limit_to_saturated_range(avxtemp1, llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 16/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_1: 12/sqrt(170)*(10*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0] = _mm512_subs_epi16(avxtempCRe[0],xtemp1);\
            avxtempERe[0] = _mm512_mulhrs_epi16(avxtempERe[0],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(170)*(9*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_9_d_sqrt170),xtemp1);\
            avxtempEIm[0] = _mm512_mulhrs_epi16(avxtempEIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_3: 4/sqrt(170)*(8*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFRe[0] = _mm512_subs_epi16(avxtempBIm[0],xtemp1);\
            avxtempFRe[0] = _mm512_mulhrs_epi16(avxtempFRe[0],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_4: 8/sqrt(170)*(7*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170),xtemp1);\
            avxtempFIm[0] = _mm512_mulhrs_epi16(avxtempFIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 12/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGRe[0] = _mm512_subs_epi16(avxtempBRe[0],xtemp1);\
            avxtempGRe[0] = _mm512_mulhrs_epi16(avxtempGRe[0],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 16/sqrt(170)*(5*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170),xtemp1);\
            avxtempGIm[0] = _mm512_mulhrs_epi16(avxtempGIm[0],i_p_16_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempDRe[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal1_2, otherwise return InnerCompoundReal_1_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_1_1*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempCIm[0],avxtemp2,avxtempZIm[0]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempFIm[0],avxtempGRe[0]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtemp2,avxtempHRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal1_5, otherwise return InnerCompoundReal1_6*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempARe[0],avxtemp2,avxtempGIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtemp2 = _mm512_mulhrs_epi16(avxtemp2,xRe[0]);\
            avxtemp2 = limit_to_saturated_range(avxtemp2, llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*calculate the the InnerCompoundReal2_0: 8/sqrt(170)*(13*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_13_d_sqrt170),xtemp1);\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_1: 4/sqrt(170)*(12*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0] = _mm512_subs_epi16(avxtempCIm[0],xtemp1);\
            avxtempERe[0] = _mm512_mulhrs_epi16(avxtempERe[0],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_2: 8/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempEIm[0] = _mm512_mulhrs_epi16(avxtempEIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_3: 8/sqrt(170)*(abs(real(In)-5*beta/sqrt(170))*/\
            avxtempFRe[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempFRe[0] = _mm512_mulhrs_epi16(avxtempFRe[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_4: 4/sqrt(170)*(abs(real(In)-4*beta/sqrt(170))*/\
            avxtempFIm[0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0]);\
            avxtempFIm[0] = _mm512_mulhrs_epi16(avxtempFIm[0],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_5: 8/sqrt(170)*(abs(real(In)-3*beta/sqrt(170))*/\
            avxtempGRe[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempGRe[0] = _mm512_mulhrs_epi16(avxtempGRe[0],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal2_0, otherwise return InnerCompoundReal2_1*/\
            avxtemp3 = select_high_low_epi16(xtemp1,avxtempDRe[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return InnerCompoundReal2_2, otherwise return InnerCompoundReal_2_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempBIm[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_1_1*/\
            avxtemp3 = select_high_low_epi16(xtemp1,avxtempCRe[0],avxtemp3,avxtempZIm[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal2_4, otherwise return InnerCompoundReal_2_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempFIm[0],avxtempGRe[0]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_1_1*/\
            avxtemp3 = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtemp3,avxtempHRe[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtemp3 = _mm512_mulhrs_epi16(avxtemp3,xRe[0]);\
            avxtemp3 = limit_to_saturated_range(avxtemp3, llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR6*/\
            /*calculate the the InnerCompoundReal3_0: 4/sqrt(170)*(14*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0] = _mm512_subs_epi16(avxtempDRe[0],xtemp1);\
            \
            /*calculate the the InnerCompoundReal3_1: 4/sqrt(170)*(abs(real(In)-10*beta/sqrt(170))*/\
            avxtempERe[0] = _mm512_subs_epi16(xtemp1,avxtempCRe[0]);\
            \
            /*calculate the the InnerCompoundReal3_2: 4/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0] = _mm512_subs_epi16(avxtempBRe[0],xtemp1);\
            \
            /*calculate the the InnerCompoundReal3_3: 4/sqrt(170)*(abs(real(In)-2*beta/sqrt(170))*/\
            avxtempFRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return InnerCompoundReal3_0, otherwise return InnerCompoundReal2_1*/\
            avxtemp4 = select_high_low_epi16(xtemp1,avxtempCIm[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal3_2, otherwise return InnerCompoundReal_3_4*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempDRe[0], otherwise return avxtempZIm[0]*/\
            avxtemp4 = select_high_low_epi16(xtemp1,avxtempBIm[0],avxtemp4,avxtempZIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR6*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtemp4 = _mm512_mulhrs_epi16(avxtemp4,avxtempZRe[0]);\
            avxtemp4 = limit_to_saturated_range(avxtemp4, llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtemp1,\
                                    avxtemp2);\
            xIm[0] = _mm512_packs_epi16(avxtemp3,\
                                    avxtemp4);\
            \
            avxtemp1 = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xIm[0]);\
            _mm512_storeu_si512 (ppSoft, avxtemp1);\
            \
            ppSoft = ppSoft + 64;\
            \
            break;\
        }\
        default:\
        {\
            printf("\nInvalid modulation request, mod order %d\n",modOrder);\
            exit(-1);\
        }\
    }\
}

#define llr_demap_1layer_tail_data_dmrs_type1_inter(modOrder, nDmrsPortIdx)\
{\
    /*Move DMRS symbols to the upper bytes by using permute.*/\
    /*Only store LLRs from actual data sub-carriers and ignore DMRS, total length will always be 1/2 amount*/\
    nRestLen_inter = nRestLen >> 1;\
    if ((nDmrsPortIdx == 0) || (nDmrsPortIdx == 1))\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempPostSINR[0]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port01_idx, avxxTxSymbol[0]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempGain[0]);\
        }\
    }\
    else\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempPostSINR[0]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port23_idx, avxxTxSymbol[0]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempGain[0]);\
        }\
    }\
    \
    switch(modOrder)\
    {\
        case BBLIB_QPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            nLlrStoreK_inter[0] = 0xffffU >> (16 - nRestLen_inter);\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:*/\
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/\
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/\
            \
            /*output 16S13*16S13->16S11*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxxTxSymbol_inter[0], i_p_2_m_sqrt2);\
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(avxtempARe[0],xRe[0]);\
            avxtempARe[0] = limit_to_saturated_range(avxtempARe[0], llr_range_low, llr_range_high);\
            \
            avxtempARe[0] = _mm512_packs_epi16(avxtempARe[0], avxtempARe[0]);\
            \
            avxtempARe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,avxtempARe[0]);\
            \
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0],avxtempARe[0]);\
            \
            ppSoft += nRestLen_inter*2;\
            break;\
        }\
        case BBLIB_QAM16:\
        {\
            /*permute the beta and postSNR as well*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            nLlrStoreK_inter[0] = 0xffffffffU >> (32 - nRestLen_inter * 2);\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm\
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta\
            \
            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta\
            \
            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0\
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0\
            \
            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0\
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/\
            \
            /*first calculate the threshold 2*beta/sqrt(10)*/\
            /*output: 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt10);\
            \
            /*calculate the factor 4/sqrt(10)/(1-beta)*/\
            /*output: 16S13*16Sx->16S(x-2)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt10);\
            \
            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/\
            avxtempCRe[0] = copy_inverted_sign_epi16(avxxTxSymbol_inter[0],_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt10));\
            \
            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/\
            /*output: 16S13*16S13->16S11*/\
            avxtempAIm[0] = _mm512_adds_epi16(avxxTxSymbol_inter[0],avxtempCRe[0]);\
            avxtempAIm[0] = _mm512_mulhrs_epi16(avxtempAIm[0],i_p_8_d_sqrt10);\
            avxtempCRe[0] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol_inter[0]),avxtempARe[0],avxtempAIm[0],_mm512_mulhrs_epi16(avxxTxSymbol_inter[0],i_p_4_d_sqrt10));\
            \
            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],xRe[0]);\
            avxtempCRe[0] = limit_to_saturated_range(avxtempCRe[0], llr_range_low, llr_range_high);\
            \
            /*calculate LLR2 and LLR3*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempBIm[0] = _mm512_subs_epi16(avxtempARe[0],_mm512_abs_epi16(avxxTxSymbol_inter[0]));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],avxtempBRe[0]);\
            avxtempBIm[0] = limit_to_saturated_range(avxtempBIm[0], llr_range_low, llr_range_high);\
            \
            /*pack to int8 and shuffle*/\
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/\
            \
            xRe[0] = _mm512_packs_epi16(avxtempCRe[0],avxtempBIm[0]);\
            xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);\
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0],xRe[0]);\
            \
            ppSoft += nRestLen_inter*4;\
            break;\
        }\
        case BBLIB_QAM64:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            nLlrStoreK_inter[0] = (nRestLen_inter > 10) ? 0xffffffffU : (0xffffffffU >> (32 - nRestLen_inter * 3));\
            nLlrStoreK_inter[1] = (nRestLen_inter > 10) ? (0xffffffffU >> (31 - (nRestLen_inter - 11) * 3)) : 0;\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)\
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)\
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)\
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)\
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0\
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)\
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)\
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)\
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)\
            \
            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(42)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt42);\
            \
            /*calculate the threshold 4*beta/sqrt(42)*/\
            avxtempAIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt42);\
            \
            /*calculate the threshold 6*beta/sqrt(42)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt42);\
            \
            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt42);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            \
            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42));\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_16_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/\
            avxtempCRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_12_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/\
            avxtempCIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt42));\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempCIm[0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));\
            \
            /*if abs(real(In))>4*beta/sqrt(42), return avxtempARe[0], otherwise return avxtempZIm[0]*/\
            avxtempDRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempDRe[0],avxtempZIm[0]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempDRe[0] = copy_sign_epi16(avxxTxSymbol_inter[0],avxtempDRe[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDRe[0] = _mm512_mulhrs_epi16(avxtempDRe[0],xRe[0]);\
            avxtempDRe[0] = limit_to_saturated_range(avxtempDRe[0], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt42),xtemp1);\
            avxtempBIm[0] = _mm512_mulhrs_epi16(avxtempBIm[0],i_p_8_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCRe[0] = _mm512_subs_epi16(avxtempAIm[0],xtemp1);\
            avxtempCRe[0] = _mm512_mulhrs_epi16(avxtempCRe[0],i_p_4_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42),xtemp1);\
            avxtempCIm[0] = _mm512_mulhrs_epi16(avxtempCIm[0],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempBIm[0],avxtempCRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/\
            avxtempDIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempDIm[0],avxtempCIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],xRe[0]);\
            avxtempDIm[0] = limit_to_saturated_range(avxtempDIm[0], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/\
            /*output 16S13*/\
            xtemp2 = select_high_low_epi16(xtemp1,avxtempAIm[0],_mm512_subs_epi16(avxtempBRe[0],xtemp1),_mm512_subs_epi16(xtemp1,avxtempARe[0]));\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            xtemp2 = _mm512_mulhrs_epi16(xtemp2,avxtempZRe[0]);\
            xtemp2 = limit_to_saturated_range(xtemp2, llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempDRe[0],\
                                        avxtempDIm[0]);\
            xIm[0] = _mm512_packs_epi16(xtemp2,\
                                        alliZero);\
            \
            avxtempCRe[0] = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);\
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0], avxtempCRe[0]);\
            \
            avxtempCIm[0]= _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);\
            _mm512_mask_storeu_epi16(ppSoft+64,nLlrStoreK_inter[1],avxtempCIm[0]);\
            \
            ppSoft = ppSoft + nRestLen_inter*6;\
            \
            break;\
        }\
        case BBLIB_QAM256:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe[0]:1/(1-beta), 16Sx*/\
            /*xIm[0]:beta,       16S15 0~0.9999*/\
            nLlrStoreK_inter[0] = (nRestLen_inter > 8) ? 0xffffffffU : (0xffffffffU >> (32 - nRestLen_inter * 4));\
            nLlrStoreK_inter[1] = (nRestLen_inter > 8) ? (0xffffffffU >> (32 - (nRestLen_inter - 8) * 4)) : 0;\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 32/sqrt(170)*(real(In)+InnerFactor0) when abs(real(In))>14*beta/sqrt(170)\
            InnerCompoundReal0_1 = 28/sqrt(170)*(real(In)+InnerFactor1) when 14*beta/sqrt(170)<abs(real(In))<=12*beta/sqrt(170)\
            InnerCompoundReal0_2 = 24/sqrt(170)*(real(In)r+InnerFactor2) when 12*beta/sqrt(170)<abs(real(In))<=10*beta/sqrt(170)\
            InnerCompoundReal0_3 = 20/sqrt(170)*(real(In)r+InnerFactor3) when 10*beta/sqrt(170)<abs(real(In))<=8*beta/sqrt(170)\
            InnerCompoundReal0_4 = 16/sqrt(170)*(real(In)r+InnerFactor4) when 8*beta/sqrt(170)<abs(real(In))<=6*beta/sqrt(170)\
            InnerCompoundReal0_5 = 12/sqrt(170)*(real(In)r+InnerFactor5) when 6*beta/sqrt(170)<abs(real(In))<=4*beta/sqrt(170)\
            InnerCompoundReal0_6 = 8/sqrt(170)*(real(In)r+InnerFactor6) when 4*beta/sqrt(170)<abs(real(In))<=2*beta/sqrt(170)\
            InnerCompoundReal0_7 = 4/sqrt(170)*real(In) when abs(real(In))<=2*beta/sqrt(170)\
            InnerFactor0 = 7*beta/(sqrt(170)); InnerFactor1 = 6*beta/(sqrt(170));InnerFactor2 = 5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = 4*beta/(sqrt(170)); InnerFactor4 = 3*beta/(sqrt(170));InnerFactor5 = 2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = beta/(sqrt(170)), when real(In)<0\
            InnerFactor0 = -7*beta/(sqrt(170)); InnerFactor1 = -6*beta/(sqrt(170));InnerFactor2 = -5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = -4*beta/(sqrt(170)); InnerFactor4 = -3*beta/(sqrt(170));InnerFactor5 = -2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = -beta/(sqrt(170)), when real(In)<0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 16/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170)) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal1_1 = 12/sqrt(170)*(-abs(real(r))+10*beta/sqrt(170)) when 12*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal1_2 = 8/sqrt(170)*(-abs(real(r))+9*beta/sqrt(170)) when 10*beta/sqrt(170)<=abs(real(r))<=12*beta/sqrt(170)\
            InnerCompoundReal1_3 = 4/sqrt(170)*(-abs(real(r))+8*beta/sqrt(170)) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal1_4 = 8/sqrt(170)*(-abs(real(r))+7*beta/sqrt(170)) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal1_5 = 12/sqrt(170)*(-abs(real(r))+6*beta/sqrt(170)) when 4*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal1_6 = 16/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170)) when 2*beta/sqrt(170)<=abs(real(r))\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2_0 = 8/sqrt(170)*(-abs(real(r))+13*beta/sqrt(170) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal2_1 = 4/sqrt(170)*(-abs(real(r))+12*beta/sqrt(170) when 10*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal2_2 = 8/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal2_3 = -8/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal2_4 = -4/sqrt(170)*(-abs(real(r))+4*beta/sqrt(170) when 2*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal2_5 = -8/sqrt(170)*(-abs(real(r))+3*beta/sqrt(170) when 2*beta/sqrt(170)>=abs(real(r))\
            \
            LLR5 = InnerCompoundImag2/(1-beta)\
            \
            LLR6 = 4/sqrt(170)*InnerCompoundReal3/(1-beta)\
            InnerCompoundReal3_0 = -1*abs(real(r))+14*beta/sqrt(170) when abs(real(r))>=12*beta/sqrt(42)\
            InnerCompoundReal3_1 = -1*abs(real(r))+10*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<12*beta/sqrt(42)\
            InnerCompoundReal3_2 = -1*abs(real(r))+6*beta/sqrt(170) when 4*beta/sqrt(170)<=abs(real(r))<8*beta/sqrt(42)\
            InnerCompoundReal3_3 = -1*abs(real(r))+2*beta/sqrt(170) when abs(real(r))<4*beta/sqrt(170)\
            \
            LLR7 = 4/sqrt(170)*InnerCompoundImag3/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(170)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt170);\
            \
            /*calculate the threshold 4*beta/sqrt(170)*/\
            avxtempAIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt170);\
            \
            /*calculate the threshold 6*beta/sqrt(170)*/\
            avxtempBRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt170);\
            \
            /*calculate the threshold 8*beta/sqrt(170)*/\
            avxtempBIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the threshold 10*beta/sqrt(170)*/\
            avxtempCRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_10_d_sqrt170);\
            \
            /*calculate the threshold 12*beta/sqrt(170)*/\
            avxtempCIm[0] = _mm512_mulhrs_epi16(xIm[0],i_p_12_d_sqrt170);\
            \
            /*calculate the threshold 14*beta/sqrt(170)*/\
            avxtempDRe[0] = _mm512_mulhrs_epi16(xIm[0],i_p_14_d_sqrt170);\
            \
            /*calculate the common scaling factor 4/sqrt(170)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt170);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            \
            /*InnerCompoundReal0_0 = 32/sqrt(170)*(abs(real(In))-7*beta/sqrt(170))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170));\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],i_p_32_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_1: 28/sqrt(170)*(abs(real(In))- 6*beta/sqrt(170))*/\
            avxtempERe[0] = _mm512_subs_epi16(xtemp1,avxtempBRe[0]);\
            avxtempERe[0] = _mm512_mulhrs_epi16(avxtempERe[0],i_p_28_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_2: 24/sqrt(170)*(abs(real(In))-5*beta/sqrt(170))*/\
            avxtempEIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempEIm[0] = _mm512_mulhrs_epi16(avxtempEIm[0],i_p_24_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_3: 20/sqrt(170)*(abs(real(In))-4*beta/sqrt(170))*/\
            avxtempFRe[0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0]);\
            avxtempFRe[0] = _mm512_mulhrs_epi16(avxtempFRe[0],i_p_20_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_4: 16/sqrt(170)*(abs(real(In))-3*beta/sqrt(170))*/\
            avxtempFIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempFIm[0] = _mm512_mulhrs_epi16(avxtempFIm[0],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_5: 12/sqrt(170)*(abs(real(In))-2*beta/sqrt(170))*/\
            avxtempGRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            avxtempGRe[0] = _mm512_mulhrs_epi16(avxtempGRe[0],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_6: 8/sqrt(170)*(abs(real(In))-1*beta/sqrt(170))*/\
            avxtempGIm[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt170));\
            avxtempGIm[0] = _mm512_mulhrs_epi16(avxtempGIm[0],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempDRe[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal0_2, otherwise return InnerCompoundReal_0_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_0_1*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempCIm[0],avxtemp1,avxtempZIm[0]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal0_4, otherwise return InnerCompoundReal_0_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtempFIm[0],avxtempGRe[0]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempARe[0], otherwise return avxtempBRe[0]*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempBIm[0],avxtemp1,avxtempHRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal0_6, otherwise return 4/sqrt(170)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempGIm[0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt170));\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return avxtempARe[0], otherwise return avxtempBRe[0]*/\
            avxtemp1 = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtemp1,avxtempZIm[0]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtemp1 = copy_sign_epi16(avxxTxSymbol_inter[0],avxtemp1);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtemp1 = _mm512_mulhrs_epi16(avxtemp1,xRe[0]);\
            avxtemp1 = limit_to_saturated_range(avxtemp1, llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 16/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_1: 12/sqrt(170)*(10*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0] = _mm512_subs_epi16(avxtempCRe[0],xtemp1);\
            avxtempERe[0] = _mm512_mulhrs_epi16(avxtempERe[0],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(170)*(9*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_9_d_sqrt170),xtemp1);\
            avxtempEIm[0] = _mm512_mulhrs_epi16(avxtempEIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_3: 4/sqrt(170)*(8*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFRe[0] = _mm512_subs_epi16(avxtempBIm[0],xtemp1);\
            avxtempFRe[0] = _mm512_mulhrs_epi16(avxtempFRe[0],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_4: 8/sqrt(170)*(7*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170),xtemp1);\
            avxtempFIm[0] = _mm512_mulhrs_epi16(avxtempFIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 12/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGRe[0] = _mm512_subs_epi16(avxtempBRe[0],xtemp1);\
            avxtempGRe[0] = _mm512_mulhrs_epi16(avxtempGRe[0],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 16/sqrt(170)*(5*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170),xtemp1);\
            avxtempGIm[0] = _mm512_mulhrs_epi16(avxtempGIm[0],i_p_16_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempDRe[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal1_2, otherwise return InnerCompoundReal_1_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_1_1*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempCIm[0],avxtemp2,avxtempZIm[0]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempFIm[0],avxtempGRe[0]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtemp2,avxtempHRe[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal1_5, otherwise return InnerCompoundReal1_6*/\
            avxtemp2 = select_high_low_epi16(xtemp1,avxtempARe[0],avxtemp2,avxtempGIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtemp2 = _mm512_mulhrs_epi16(avxtemp2,xRe[0]);\
            avxtemp2 = limit_to_saturated_range(avxtemp2, llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*calculate the the InnerCompoundReal2_0: 8/sqrt(170)*(13*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_13_d_sqrt170),xtemp1);\
            avxtempDIm[0] = _mm512_mulhrs_epi16(avxtempDIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_1: 4/sqrt(170)*(12*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0] = _mm512_subs_epi16(avxtempCIm[0],xtemp1);\
            avxtempERe[0] = _mm512_mulhrs_epi16(avxtempERe[0],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_2: 8/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempEIm[0] = _mm512_mulhrs_epi16(avxtempEIm[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_3: 8/sqrt(170)*(abs(real(In)-5*beta/sqrt(170))*/\
            avxtempFRe[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempFRe[0] = _mm512_mulhrs_epi16(avxtempFRe[0],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_4: 4/sqrt(170)*(abs(real(In)-4*beta/sqrt(170))*/\
            avxtempFIm[0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0]);\
            avxtempFIm[0] = _mm512_mulhrs_epi16(avxtempFIm[0],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_5: 8/sqrt(170)*(abs(real(In)-3*beta/sqrt(170))*/\
            avxtempGRe[0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempGRe[0] = _mm512_mulhrs_epi16(avxtempGRe[0],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal2_0, otherwise return InnerCompoundReal2_1*/\
            avxtemp3 = select_high_low_epi16(xtemp1,avxtempDRe[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return InnerCompoundReal2_2, otherwise return InnerCompoundReal_2_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempBIm[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_1_1*/\
            avxtemp3 = select_high_low_epi16(xtemp1,avxtempCRe[0],avxtemp3,avxtempZIm[0]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal2_4, otherwise return InnerCompoundReal_2_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempARe[0],avxtempFIm[0],avxtempGRe[0]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return avxtempARe[0], otherwise return InnerCompoundReal_1_1*/\
            avxtemp3 = select_high_low_epi16(xtemp1,avxtempBRe[0],avxtemp3,avxtempHRe[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtemp3 = _mm512_mulhrs_epi16(avxtemp3,xRe[0]);\
            avxtemp3 = limit_to_saturated_range(avxtemp3, llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR6*/\
            /*calculate the the InnerCompoundReal3_0: 4/sqrt(170)*(14*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0] = _mm512_subs_epi16(avxtempDRe[0],xtemp1);\
            \
            /*calculate the the InnerCompoundReal3_1: 4/sqrt(170)*(abs(real(In)-10*beta/sqrt(170))*/\
            avxtempERe[0] = _mm512_subs_epi16(xtemp1,avxtempCRe[0]);\
            \
            /*calculate the the InnerCompoundReal3_2: 4/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0] = _mm512_subs_epi16(avxtempBRe[0],xtemp1);\
            \
            /*calculate the the InnerCompoundReal3_3: 4/sqrt(170)*(abs(real(In)-2*beta/sqrt(170))*/\
            avxtempFRe[0] = _mm512_subs_epi16(xtemp1,avxtempARe[0]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return InnerCompoundReal3_0, otherwise return InnerCompoundReal2_1*/\
            avxtemp4 = select_high_low_epi16(xtemp1,avxtempCIm[0],avxtempDIm[0],avxtempERe[0]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal3_2, otherwise return InnerCompoundReal_3_4*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempAIm[0],avxtempEIm[0],avxtempFRe[0]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempDRe[0], otherwise return avxtempZIm[0]*/\
            avxtemp4 = select_high_low_epi16(xtemp1,avxtempBIm[0],avxtemp4,avxtempZIm[0]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR6*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtemp4 = _mm512_mulhrs_epi16(avxtemp4,avxtempZRe[0]);\
            avxtemp4 = limit_to_saturated_range(avxtemp4, llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtemp1,\
                                    avxtemp2);\
            xIm[0] = _mm512_packs_epi16(avxtemp3,\
                                    avxtemp4);\
            \
            avxtemp1 = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xIm[0]);\
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0], avxtemp1);\
            \
            avxtemp2= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xIm[0]);\
            _mm512_mask_storeu_epi16 (ppSoft+64,nLlrStoreK_inter[1], avxtemp2);\
            \
            ppSoft = ppSoft + nRestLen_inter*8;\
            \
            break;\
        }\
        default:\
        {\
            printf("\nInvalid modulation request, mod order %d\n",modOrder);\
            exit(-1);\
        }\
    }\
}

#define llr_demap_2layer_normal_data_dmrs_type1_inter(modOrder, nDmrsPortIdx)\
{\
    /*Move DMRS symbols to the upper bytes by using permute.*/\
    /*Only store LLRs from actual data sub-carriers and ignore DMRS*/\
    if ((nDmrsPortIdx == 0) || (nDmrsPortIdx == 1))\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempPostSINR[0]);\
        ftempPostSINR_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempPostSINR[1]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port01_idx, avxxTxSymbol[0]);\
        avxxTxSymbol_inter[1] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port01_idx, avxxTxSymbol[1]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempGain[0]);\
            ftempGain_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempGain[1]);\
        }\
    }\
    else\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempPostSINR[0]);\
        ftempPostSINR_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempPostSINR[1]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port23_idx, avxxTxSymbol[0]);\
        avxxTxSymbol_inter[1] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port23_idx, avxxTxSymbol[1]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempGain[0]);\
            ftempGain_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempGain[1]);\
        }\
    }\
    \
    /* layermapping for data */\
    xtemp1 = _mm512_permutex2var_epi32(avxxTxSymbol_inter[0], m512_permutex_each32_first256, avxxTxSymbol_inter[1]);\
    avxxTxSymbol_inter[1] = _mm512_permutex2var_epi32(avxxTxSymbol_inter[0], m512_permutex_each32_second256, avxxTxSymbol_inter[1]);\
    avxxTxSymbol_inter[0] = xtemp1;\
    \
    switch(modOrder)\
    {\
        case BBLIB_QPSK:\
        {\
            /*layermapping for snr*/\
            /*xRe:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            /*Algorithm:*/\
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/\
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/\
            \
            /*output 16S13*16S13->16S11*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(avxxTxSymbol_inter[0], i_p_2_m_sqrt2);\
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(avxtempARe[0][0],xRe[0]);\
            /*do left shift according to requried output range*/\
            avxtempARe[0][0] = limit_to_saturated_range(avxtempARe[0][0], llr_range_low, llr_range_high);\
            \
            avxtempARe[0][1] = _mm512_mulhrs_epi16(avxxTxSymbol_inter[1], i_p_2_m_sqrt2);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(avxtempARe[0][1],xRe[1]);\
            avxtempARe[0][1] = limit_to_saturated_range(avxtempARe[0][1], llr_range_low, llr_range_high);\
            \
            avxtempARe[0][0] = _mm512_packs_epi16(avxtempARe[0][0], avxtempARe[0][1]);\
            \
            avxtempARe[0][0] = _mm512_permutexvar_epi64(qpsk_perm_idx,avxtempARe[0][0]);\
            \
            /*Only store the data LLRs*/\
            _mm512_mask_storeu_epi16 (ppSoft,0xffff,avxtempARe[0][0]);\
            \
            ppSoft += 32;\
            break;\
        }\
        case BBLIB_QAM16:\
        {\
            /*permute the beta and postSNR as well*/\
            /*xRe:1/(1-beta), 16Sx*/\
            /*xIm:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempGain_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta\
            \
            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta\
            \
            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0\
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0\
            \
            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0\
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/\
            \
            /*first calculate the threshold 2*beta/sqrt(10)*/\
            /*output: 16S15*16S13->16S13*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt10);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_2_d_sqrt10);\
            \
            /*calculate the factor 4/sqrt(10)/(1-beta)*/\
            /*output: 16S13*16Sx->16S(x-2)*/\
            avxtempBRe[0][0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt10);\
            avxtempBRe[0][1] = _mm512_mulhrs_epi16(xRe[1],i_p_4_d_sqrt10);\
            \
            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/\
            avxtempARe[1][0] = copy_inverted_sign_epi16(avxxTxSymbol_inter[0],_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt10));\
            avxtempARe[1][1] = copy_inverted_sign_epi16(avxxTxSymbol_inter[1],_mm512_mulhrs_epi16(xIm[1],i_p_1_d_sqrt10));\
            \
            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/\
            /*output: 16S13*16S13->16S11*/\
            avxtempAIm[0][0] = _mm512_adds_epi16(avxxTxSymbol_inter[0],avxtempARe[1][0]);\
            avxtempAIm[0][0] = _mm512_mulhrs_epi16(avxtempAIm[0][0],i_p_8_d_sqrt10);\
            avxtempARe[1][0] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol_inter[0]),avxtempARe[0][0],avxtempAIm[0][0],_mm512_mulhrs_epi16(avxxTxSymbol_inter[0],i_p_4_d_sqrt10));\
            \
            avxtempAIm[0][1] = _mm512_adds_epi16(avxxTxSymbol_inter[1],avxtempARe[1][1]);\
            avxtempAIm[0][1] = _mm512_mulhrs_epi16(avxtempAIm[0][1],i_p_8_d_sqrt10);\
            avxtempARe[1][1] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol_inter[1]),avxtempARe[0][1],avxtempAIm[0][1],_mm512_mulhrs_epi16(avxxTxSymbol_inter[1],i_p_4_d_sqrt10));\
            \
            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[1][0] = _mm512_mulhrs_epi16(avxtempARe[1][0],xRe[0]);\
            avxtempARe[1][1] = _mm512_mulhrs_epi16(avxtempARe[1][1],xRe[1]);\
            avxtempARe[1][0] = limit_to_saturated_range(avxtempARe[1][0], llr_range_low, llr_range_high);\
            avxtempARe[1][1] = limit_to_saturated_range(avxtempARe[1][1], llr_range_low, llr_range_high);\
            \
            /*calculate LLR2 and LLR3*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempBRe[1][0] = _mm512_subs_epi16(avxtempARe[0][0],_mm512_abs_epi16(avxxTxSymbol_inter[0]));\
            avxtempBRe[1][0] = _mm512_mulhrs_epi16(avxtempBRe[1][0],avxtempBRe[0][0]);\
            avxtempBRe[1][0] = limit_to_saturated_range(avxtempBRe[1][0], llr_range_low, llr_range_high);\
            \
            avxtempBRe[1][1] = _mm512_subs_epi16(avxtempARe[0][1],_mm512_abs_epi16(avxxTxSymbol_inter[1]));\
            avxtempBRe[1][1] = _mm512_mulhrs_epi16(avxtempBRe[1][1],avxtempBRe[0][1]);\
            avxtempBRe[1][1] = limit_to_saturated_range(avxtempBRe[1][1], llr_range_low, llr_range_high);\
            \
            /*pack to int8 and permute*/\
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/\
            /*xRe[0]: LLR0_16,LLR1_16,LLR0_17,LLR1_17... LLR3_31*/\
            xRe[0] = _mm512_packs_epi16(avxtempARe[1][0],avxtempBRe[1][0]);\
            xIm[0] = _mm512_packs_epi16(avxtempARe[1][1],avxtempBRe[1][1]);\
            \
            xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);\
            \
            /*Only store the data LLRs*/\
            _mm512_storeu_si512 (ppSoft, xRe[0]);\
            \
            ppSoft = ppSoft + 64;\
            break;\
        }\
        case BBLIB_QAM64:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe:1/(1-beta), 16Sx*/\
            /*xIm:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempGain_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)\
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)\
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)\
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)\
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0\
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)\
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)\
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)\
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)\
            \
            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(42)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt42);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_2_d_sqrt42);\
            \
            /*calculate the threshold 4*beta/sqrt(42)*/\
            avxtempAIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt42);\
            avxtempAIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_4_d_sqrt42);\
            \
            /*calculate the threshold 6*beta/sqrt(42)*/\
            avxtempBRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt42);\
            avxtempBRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_6_d_sqrt42);\
            \
            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt42);\
            avxtempZRe[1] = _mm512_mulhrs_epi16(xRe[1],i_p_4_d_sqrt42);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            xtemp2 = _mm512_abs_epi16(avxxTxSymbol_inter[1]);\
            \
            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42));\
            avxtempBIm[0][0] = _mm512_mulhrs_epi16(avxtempBIm[0][0],i_p_16_d_sqrt42);\
            avxtempBIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt42));\
            avxtempBIm[0][1] = _mm512_mulhrs_epi16(avxtempBIm[0][1],i_p_16_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/\
            avxtempCRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempARe[0][0]);\
            avxtempCRe[0][0] = _mm512_mulhrs_epi16(avxtempCRe[0][0],i_p_12_d_sqrt42);\
            avxtempCRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempARe[0][1]);\
            avxtempCRe[0][1] = _mm512_mulhrs_epi16(avxtempCRe[0][1],i_p_12_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/\
            avxtempCIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt42));\
            avxtempCIm[0][0] = _mm512_mulhrs_epi16(avxtempCIm[0][0],i_p_8_d_sqrt42);\
            avxtempCIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_1_d_sqrt42));\
            avxtempCIm[0][1] = _mm512_mulhrs_epi16(avxtempCIm[0][1],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempBIm[0][0],avxtempCRe[0][0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempBIm[0][1],avxtempCRe[0][1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempCIm[0][0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempCIm[0][1],_mm512_mulhrs_epi16(xtemp2,i_p_4_d_sqrt42));\
            \
            /*if abs(real(In))>4*beta/sqrt(42), return avxtempARe, otherwise return avxtempZIm*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempARe[1][0],avxtempZIm[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempARe[1][1],avxtempZIm[1]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempARe[1][0] = copy_sign_epi16(avxxTxSymbol_inter[0],avxtempARe[1][0]);\
            avxtempARe[1][1] = copy_sign_epi16(avxxTxSymbol_inter[1],avxtempARe[1][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[1][0] = _mm512_mulhrs_epi16(avxtempARe[1][0],xRe[0]);\
            \
            avxtempARe[1][1] = _mm512_mulhrs_epi16(avxtempARe[1][1],xRe[1]);\
            \
            avxtempARe[1][0] = limit_to_saturated_range(avxtempARe[1][0], llr_range_low, llr_range_high);\
            avxtempARe[1][1] = limit_to_saturated_range(avxtempARe[1][1], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt42),xtemp1);\
            avxtempBIm[0][0] = _mm512_mulhrs_epi16(avxtempBIm[0][0],i_p_8_d_sqrt42);\
            avxtempBIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt42),xtemp2);\
            avxtempBIm[0][1] = _mm512_mulhrs_epi16(avxtempBIm[0][1],i_p_8_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCRe[0][0] = _mm512_subs_epi16(avxtempAIm[0][0],xtemp1);\
            avxtempCRe[0][0] = _mm512_mulhrs_epi16(avxtempCRe[0][0],i_p_4_d_sqrt42);\
            avxtempCRe[0][1] = _mm512_subs_epi16(avxtempAIm[0][1],xtemp2);\
            avxtempCRe[0][1] = _mm512_mulhrs_epi16(avxtempCRe[0][1],i_p_4_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42),xtemp1);\
            avxtempCIm[0][0] = _mm512_mulhrs_epi16(avxtempCIm[0][0],i_p_8_d_sqrt42);\
            avxtempCIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt42),xtemp2);\
            avxtempCIm[0][1] = _mm512_mulhrs_epi16(avxtempCIm[0][1],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempBIm[0][0],avxtempCRe[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempBIm[0][1],avxtempCRe[0][1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempBRe[1][0],avxtempCIm[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempBRe[1][1],avxtempCIm[0][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempBRe[1][0] = _mm512_mulhrs_epi16(avxtempBRe[1][0],xRe[0]);\
            avxtempBRe[1][0] = limit_to_saturated_range(avxtempBRe[1][0], llr_range_low, llr_range_high);\
            avxtempBRe[1][1] = _mm512_mulhrs_epi16(avxtempBRe[1][1],xRe[1]);\
            avxtempBRe[1][1] = limit_to_saturated_range(avxtempBRe[1][1], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/\
            /*output 16S13*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],_mm512_subs_epi16(avxtempBRe[0][0],xtemp1),_mm512_subs_epi16(xtemp1,avxtempARe[0][0]));\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],_mm512_subs_epi16(avxtempBRe[0][1],xtemp2),_mm512_subs_epi16(xtemp2,avxtempARe[0][1]));\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempCRe[1][0] = _mm512_mulhrs_epi16(avxtempCRe[1][0],avxtempZRe[0]);\
            avxtempCRe[1][0] = limit_to_saturated_range(avxtempCRe[1][0], llr_range_low, llr_range_high);\
            avxtempCRe[1][1] = _mm512_mulhrs_epi16(avxtempCRe[1][1],avxtempZRe[1]);\
            avxtempCRe[1][1] = limit_to_saturated_range(avxtempCRe[1][1], llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempARe[1][0],\
                                        avxtempBRe[1][0]);\
            xRe[1] = _mm512_packs_epi16(avxtempCRe[1][0],\
                                        alliZero);\
            xIm[0] = _mm512_packs_epi16(alliZero,\
                                        avxtempARe[1][1]);\
            xIm[1] = _mm512_packs_epi16(avxtempBRe[1][1],\
                                        avxtempCRe[1][1]);\
            \
            avxtempCIm[0][0] = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xRe[1]);\
            /*Only store the data LLRs*/\
            _mm512_storeu_si512 (ppSoft, avxtempCIm[0][0]);\
            \
            avxtempCIm[0][1] = _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xRe[1]);\
            avxtempCIm[1][0] = _mm512_permutex2var_epi16(xIm[0], qam64_third_32, xIm[1]);\
            avxtempCIm[1][1] = _mm512_permutex2var_epi16(xIm[0], qam64_last_64, xIm[1]);\
            \
            avxtempCIm[0][1] = _mm512_permutex2var_epi64(avxtempCIm[0][1],qam64_index_lo_hi,avxtempCIm[1][0]);\
            /*Only store the data LLRs*/\
            _mm512_mask_storeu_epi16 (ppSoft+64, 0xffff, avxtempCIm[0][1]);\
            \
            ppSoft = ppSoft + 96;\
            break;\
        }\
        case BBLIB_QAM256:\
        {\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe:1/(1-beta), 16Sx*/\
            /*xIm:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempGain_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            \
            xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 32/sqrt(170)*(real(In)+InnerFactor0) when abs(real(In))>14*beta/sqrt(170)\
            InnerCompoundReal0_1 = 28/sqrt(170)*(real(In)+InnerFactor1) when 14*beta/sqrt(170)<abs(real(In))<=12*beta/sqrt(170)\
            InnerCompoundReal0_2 = 24/sqrt(170)*(real(In)r+InnerFactor2) when 12*beta/sqrt(170)<abs(real(In))<=10*beta/sqrt(170)\
            InnerCompoundReal0_3 = 20/sqrt(170)*(real(In)r+InnerFactor3) when 10*beta/sqrt(170)<abs(real(In))<=8*beta/sqrt(170)\
            InnerCompoundReal0_4 = 16/sqrt(170)*(real(In)r+InnerFactor4) when 8*beta/sqrt(170)<abs(real(In))<=6*beta/sqrt(170)\
            InnerCompoundReal0_5 = 12/sqrt(170)*(real(In)r+InnerFactor5) when 6*beta/sqrt(170)<abs(real(In))<=4*beta/sqrt(170)\
            InnerCompoundReal0_6 = 8/sqrt(170)*(real(In)r+InnerFactor6) when 4*beta/sqrt(170)<abs(real(In))<=2*beta/sqrt(170)\
            InnerCompoundReal0_7 = 4/sqrt(170)*real(In) when abs(real(In))<=2*beta/sqrt(170)\
            InnerFactor0 = 7*beta/(sqrt(170)); InnerFactor1 = 6*beta/(sqrt(170));InnerFactor2 = 5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = 4*beta/(sqrt(170)); InnerFactor4 = 3*beta/(sqrt(170));InnerFactor5 = 2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = beta/(sqrt(170)), when real(In)<0\
            InnerFactor0 = -7*beta/(sqrt(170)); InnerFactor1 = -6*beta/(sqrt(170));InnerFactor2 = -5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = -4*beta/(sqrt(170)); InnerFactor4 = -3*beta/(sqrt(170));InnerFactor5 = -2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = -beta/(sqrt(170)), when real(In)<0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 16/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170)) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal1_1 = 12/sqrt(170)*(-abs(real(r))+10*beta/sqrt(170)) when 12*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal1_2 = 8/sqrt(170)*(-abs(real(r))+9*beta/sqrt(170)) when 10*beta/sqrt(170)<=abs(real(r))<=12*beta/sqrt(170)\
            InnerCompoundReal1_3 = 4/sqrt(170)*(-abs(real(r))+8*beta/sqrt(170)) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal1_4 = 8/sqrt(170)*(-abs(real(r))+7*beta/sqrt(170)) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal1_5 = 12/sqrt(170)*(-abs(real(r))+6*beta/sqrt(170)) when 4*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal1_6 = 16/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170)) when 2*beta/sqrt(170)<=abs(real(r))\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2_0 = 8/sqrt(170)*(-abs(real(r))+13*beta/sqrt(170) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal2_1 = 4/sqrt(170)*(-abs(real(r))+12*beta/sqrt(170) when 10*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal2_2 = 8/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal2_3 = -8/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal2_4 = -4/sqrt(170)*(-abs(real(r))+4*beta/sqrt(170) when 2*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal2_5 = -8/sqrt(170)*(-abs(real(r))+3*beta/sqrt(170) when 2*beta/sqrt(170)>=abs(real(r))\
            \
            LLR5 = InnerCompoundImag2/(1-beta)\
            \
            LLR6 = 4/sqrt(170)*InnerCompoundReal3/(1-beta)\
            InnerCompoundReal3_0 = -1*abs(real(r))+14*beta/sqrt(170) when abs(real(r))>=12*beta/sqrt(42)\
            InnerCompoundReal3_1 = -1*abs(real(r))+10*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<12*beta/sqrt(42)\
            InnerCompoundReal3_2 = -1*abs(real(r))+6*beta/sqrt(170) when 4*beta/sqrt(170)<=abs(real(r))<8*beta/sqrt(42)\
            InnerCompoundReal3_3 = -1*abs(real(r))+2*beta/sqrt(170) when abs(real(r))<4*beta/sqrt(170)\
            \
            LLR7 = 4/sqrt(170)*InnerCompoundImag3/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(170)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt170);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_2_d_sqrt170);\
            \
            /*calculate the threshold 4*beta/sqrt(170)*/\
            avxtempAIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt170);\
            avxtempAIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_4_d_sqrt170);\
            \
            /*calculate the threshold 6*beta/sqrt(170)*/\
            avxtempBRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt170);\
            avxtempBRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_6_d_sqrt170);\
            \
            /*calculate the threshold 8*beta/sqrt(170)*/\
            avxtempBIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_8_d_sqrt170);\
            avxtempBIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_8_d_sqrt170);\
            \
            /*calculate the threshold 10*beta/sqrt(170)*/\
            avxtempCRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_10_d_sqrt170);\
            avxtempCRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_10_d_sqrt170);\
            \
            /*calculate the threshold 12*beta/sqrt(170)*/\
            avxtempCIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_12_d_sqrt170);\
            avxtempCIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_12_d_sqrt170);\
            \
            /*calculate the threshold 14*beta/sqrt(170)*/\
            avxtempDRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_14_d_sqrt170);\
            avxtempDRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_14_d_sqrt170);\
            \
            /*calculate the common scaling factor 4/sqrt(170)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt170);\
            avxtempZRe[1] = _mm512_mulhrs_epi16(xRe[1],i_p_4_d_sqrt170);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            xtemp2 = _mm512_abs_epi16(avxxTxSymbol_inter[1]);\
            \
            /*InnerCompoundReal0_0 = 32/sqrt(170)*(abs(real(In))-7*beta/sqrt(170))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170));\
            avxtempDIm[0][0] = _mm512_mulhrs_epi16(avxtempDIm[0][0],i_p_32_d_sqrt170);\
            avxtempDIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_7_d_sqrt170));\
            avxtempDIm[0][1] = _mm512_mulhrs_epi16(avxtempDIm[0][1],i_p_32_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_1: 28/sqrt(170)*(abs(real(In))- 6*beta/sqrt(170))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(xtemp1,avxtempBRe[0][0]);\
            avxtempERe[0][0] = _mm512_mulhrs_epi16(avxtempERe[0][0],i_p_28_d_sqrt170);\
            avxtempERe[0][1] = _mm512_subs_epi16(xtemp2,avxtempBRe[0][1]);\
            avxtempERe[0][1] = _mm512_mulhrs_epi16(avxtempERe[0][1],i_p_28_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_2: 24/sqrt(170)*(abs(real(In))-5*beta/sqrt(170))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempEIm[0][0] = _mm512_mulhrs_epi16(avxtempEIm[0][0],i_p_24_d_sqrt170);\
            avxtempEIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt170));\
            avxtempEIm[0][1] = _mm512_mulhrs_epi16(avxtempEIm[0][1],i_p_24_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_3: 20/sqrt(170)*(abs(real(In))-4*beta/sqrt(170))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0][0]);\
            avxtempFRe[0][0] = _mm512_mulhrs_epi16(avxtempFRe[0][0],i_p_20_d_sqrt170);\
            avxtempFRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempAIm[0][1]);\
            avxtempFRe[0][1] = _mm512_mulhrs_epi16(avxtempFRe[0][1],i_p_20_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_4: 16/sqrt(170)*(abs(real(In))-3*beta/sqrt(170))*/\
            avxtempFIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempFIm[0][0] = _mm512_mulhrs_epi16(avxtempFIm[0][0],i_p_16_d_sqrt170);\
            avxtempFIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt170));\
            avxtempFIm[0][1] = _mm512_mulhrs_epi16(avxtempFIm[0][1],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_5: 12/sqrt(170)*(abs(real(In))-2*beta/sqrt(170))*/\
            avxtempGRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempARe[0][0]);\
            avxtempGRe[0][0] = _mm512_mulhrs_epi16(avxtempGRe[0][0],i_p_12_d_sqrt170);\
            avxtempGRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempARe[0][1]);\
            avxtempGRe[0][1] = _mm512_mulhrs_epi16(avxtempGRe[0][1],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_6: 8/sqrt(170)*(abs(real(In))-1*beta/sqrt(170))*/\
            avxtempGIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt170));\
            avxtempGIm[0][0] = _mm512_mulhrs_epi16(avxtempGIm[0][0],i_p_8_d_sqrt170);\
            avxtempGIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_1_d_sqrt170));\
            avxtempGIm[0][1] = _mm512_mulhrs_epi16(avxtempGIm[0][1],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempDRe[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempDRe[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal0_2, otherwise return InnerCompoundReal_0_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempCRe[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_0_1*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempCIm[0][0],avxtempARe[1][0],avxtempZIm[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempCIm[0][1],avxtempARe[1][1],avxtempZIm[1]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal0_4, otherwise return InnerCompoundReal_0_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempFIm[0][0],avxtempGRe[0][0]);\
            avxtempHRe[1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempFIm[0][1],avxtempGRe[0][1]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempARe, otherwise return avxtempBRe*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempBIm[0][0],avxtempARe[1][0],avxtempHRe[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempBIm[0][1],avxtempARe[1][1],avxtempHRe[1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal0_6, otherwise return 4/sqrt(170)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempGIm[0][0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt170));\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempGIm[0][1],_mm512_mulhrs_epi16(xtemp2,i_p_4_d_sqrt170));\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return avxtempARe, otherwise return avxtempBRe*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempARe[1][0],avxtempZIm[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempARe[1][1],avxtempZIm[1]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempARe[1][0] = copy_sign_epi16(avxxTxSymbol_inter[0],avxtempARe[1][0]);\
            avxtempARe[1][1] = copy_sign_epi16(avxxTxSymbol_inter[1],avxtempARe[1][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[1][0] = _mm512_mulhrs_epi16(avxtempARe[1][0],xRe[0]);\
            \
            avxtempARe[1][1] = _mm512_mulhrs_epi16(avxtempARe[1][1],xRe[1]);\
            \
            avxtempARe[1][0] = limit_to_saturated_range(avxtempARe[1][0], llr_range_low, llr_range_high);\
            avxtempARe[1][1] = limit_to_saturated_range(avxtempARe[1][1], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 16/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempDIm[0][0] = _mm512_mulhrs_epi16(avxtempDIm[0][0],i_p_16_d_sqrt170);\
            avxtempDIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_11_d_sqrt170),xtemp2);\
            avxtempDIm[0][1] = _mm512_mulhrs_epi16(avxtempDIm[0][1],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_1: 12/sqrt(170)*(10*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(avxtempCRe[0][0],xtemp1);\
            avxtempERe[0][0] = _mm512_mulhrs_epi16(avxtempERe[0][0],i_p_12_d_sqrt170);\
            avxtempERe[0][1] = _mm512_subs_epi16(avxtempCRe[0][1],xtemp2);\
            avxtempERe[0][1] = _mm512_mulhrs_epi16(avxtempERe[0][1],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(170)*(9*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_9_d_sqrt170),xtemp1);\
            avxtempEIm[0][0] = _mm512_mulhrs_epi16(avxtempEIm[0][0],i_p_8_d_sqrt170);\
            avxtempEIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_9_d_sqrt170),xtemp2);\
            avxtempEIm[0][1] = _mm512_mulhrs_epi16(avxtempEIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_3: 4/sqrt(170)*(8*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(avxtempBIm[0][0],xtemp1);\
            avxtempFRe[0][0] = _mm512_mulhrs_epi16(avxtempFRe[0][0],i_p_4_d_sqrt170);\
            avxtempFRe[0][1] = _mm512_subs_epi16(avxtempBIm[0][1],xtemp2);\
            avxtempFRe[0][1] = _mm512_mulhrs_epi16(avxtempFRe[0][1],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_4: 8/sqrt(170)*(7*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170),xtemp1);\
            avxtempFIm[0][0] = _mm512_mulhrs_epi16(avxtempFIm[0][0],i_p_8_d_sqrt170);\
            avxtempFIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_7_d_sqrt170),xtemp2);\
            avxtempFIm[0][1] = _mm512_mulhrs_epi16(avxtempFIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 12/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGRe[0][0] = _mm512_subs_epi16(avxtempBRe[0][0],xtemp1);\
            avxtempGRe[0][0] = _mm512_mulhrs_epi16(avxtempGRe[0][0],i_p_12_d_sqrt170);\
            avxtempGRe[0][1] = _mm512_subs_epi16(avxtempBRe[0][1],xtemp2);\
            avxtempGRe[0][1] = _mm512_mulhrs_epi16(avxtempGRe[0][1],i_p_12_d_sqrt170);\
        \
            /*calculate the the InnerCompoundReal1_5: 16/sqrt(170)*(5*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170),xtemp1);\
            avxtempGIm[0][0] = _mm512_mulhrs_epi16(avxtempGIm[0][0],i_p_16_d_sqrt170);\
            avxtempGIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt170),xtemp2);\
            avxtempGIm[0][1] = _mm512_mulhrs_epi16(avxtempGIm[0][1],i_p_16_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempDRe[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempDRe[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal1_2, otherwise return InnerCompoundReal_1_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempCRe[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_1_1*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempCIm[0][0],avxtempBRe[1][0],avxtempZIm[0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempCIm[0][1],avxtempBRe[1][1],avxtempZIm[1]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempFIm[0][0],avxtempGRe[0][0]);\
            avxtempHRe[1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempFIm[0][1],avxtempGRe[0][1]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempBRe[1][0],avxtempHRe[0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempBRe[1][1],avxtempHRe[1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal1_5, otherwise return InnerCompoundReal1_6*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempBRe[1][0],avxtempGIm[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempBRe[1][1],avxtempGIm[0][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempBRe[1][0] = _mm512_mulhrs_epi16(avxtempBRe[1][0],xRe[0]);\
            avxtempBRe[1][0] = limit_to_saturated_range(avxtempBRe[1][0], llr_range_low, llr_range_high);\
            avxtempBRe[1][1] = _mm512_mulhrs_epi16(avxtempBRe[1][1],xRe[1]);\
            avxtempBRe[1][1] = limit_to_saturated_range(avxtempBRe[1][1], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*calculate the the InnerCompoundReal2_0: 8/sqrt(170)*(13*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_13_d_sqrt170),xtemp1);\
            avxtempDIm[0][0] = _mm512_mulhrs_epi16(avxtempDIm[0][0],i_p_8_d_sqrt170);\
            avxtempDIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_13_d_sqrt170),xtemp2);\
            avxtempDIm[0][1] = _mm512_mulhrs_epi16(avxtempDIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_1: 4/sqrt(170)*(12*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(avxtempCIm[0][0],xtemp1);\
            avxtempERe[0][0] = _mm512_mulhrs_epi16(avxtempERe[0][0],i_p_4_d_sqrt170);\
            avxtempERe[0][1] = _mm512_subs_epi16(avxtempCIm[0][1],xtemp2);\
            avxtempERe[0][1] = _mm512_mulhrs_epi16(avxtempERe[0][1],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_2: 8/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempEIm[0][0] = _mm512_mulhrs_epi16(avxtempEIm[0][0],i_p_8_d_sqrt170);\
            avxtempEIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_11_d_sqrt170),xtemp2);\
            avxtempEIm[0][1] = _mm512_mulhrs_epi16(avxtempEIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_3: 8/sqrt(170)*(abs(real(In)-5*beta/sqrt(170))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempFRe[0][0] = _mm512_mulhrs_epi16(avxtempFRe[0][0],i_p_8_d_sqrt170);\
            avxtempFRe[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt170));\
            avxtempFRe[0][1] = _mm512_mulhrs_epi16(avxtempFRe[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_4: 4/sqrt(170)*(abs(real(In)-4*beta/sqrt(170))*/\
            avxtempFIm[0][0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0][0]);\
            avxtempFIm[0][0] = _mm512_mulhrs_epi16(avxtempFIm[0][0],i_p_4_d_sqrt170);\
            avxtempFIm[0][1] = _mm512_subs_epi16(xtemp2,avxtempAIm[0][1]);\
            avxtempFIm[0][1] = _mm512_mulhrs_epi16(avxtempFIm[0][1],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_5: 8/sqrt(170)*(abs(real(In)-3*beta/sqrt(170))*/\
            avxtempGRe[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempGRe[0][0] = _mm512_mulhrs_epi16(avxtempGRe[0][0],i_p_8_d_sqrt170);\
            avxtempGRe[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt170));\
            avxtempGRe[0][1] = _mm512_mulhrs_epi16(avxtempGRe[0][1],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal2_0, otherwise return InnerCompoundReal2_1*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempDRe[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempDRe[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return InnerCompoundReal2_2, otherwise return InnerCompoundReal_2_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempBIm[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempBIm[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_1_1*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempCRe[0][0],avxtempCRe[1][0],avxtempZIm[0]);\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempCRe[0][1],avxtempCRe[1][1],avxtempZIm[1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal2_4, otherwise return InnerCompoundReal_2_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempFIm[0][0],avxtempGRe[0][0]);\
            avxtempHRe[1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempFIm[0][1],avxtempGRe[0][1]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_1_1*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempCRe[1][0],avxtempHRe[0]);\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempCRe[1][1],avxtempHRe[1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempCRe[1][0] = _mm512_mulhrs_epi16(avxtempCRe[1][0],xRe[0]);\
            avxtempCRe[1][0] = limit_to_saturated_range(avxtempCRe[1][0], llr_range_low, llr_range_high);\
            avxtempCRe[1][1] = _mm512_mulhrs_epi16(avxtempCRe[1][1],xRe[1]);\
            avxtempCRe[1][1] = limit_to_saturated_range(avxtempCRe[1][1], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR6*/\
            /*calculate the the InnerCompoundReal3_0: 4/sqrt(170)*(14*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(avxtempDRe[0][0],xtemp1);\
            avxtempDIm[0][1] = _mm512_subs_epi16(avxtempDRe[0][1],xtemp2);\
            \
            /*calculate the the InnerCompoundReal3_1: 4/sqrt(170)*(abs(real(In)-10*beta/sqrt(170))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(xtemp1,avxtempCRe[0][0]);\
            avxtempERe[0][1] = _mm512_subs_epi16(xtemp2,avxtempCRe[0][1]);\
            \
            /*calculate the the InnerCompoundReal3_2: 4/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(avxtempBRe[0][0],xtemp1);\
            avxtempEIm[0][1] = _mm512_subs_epi16(avxtempBRe[0][1],xtemp2);\
            \
            /*calculate the the InnerCompoundReal3_3: 4/sqrt(170)*(abs(real(In)-2*beta/sqrt(170))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempARe[0][0]);\
            avxtempFRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempARe[0][1]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return InnerCompoundReal3_0, otherwise return InnerCompoundReal2_1*/\
            avxtempDRe[1][0] = select_high_low_epi16(xtemp1,avxtempCIm[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempDRe[1][1] = select_high_low_epi16(xtemp2,avxtempCIm[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal3_2, otherwise return InnerCompoundReal_3_4*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempDRe, otherwise return avxtempZIm*/\
            avxtempDRe[1][0] = select_high_low_epi16(xtemp1,avxtempBIm[0][0],avxtempDRe[1][0],avxtempZIm[0]);\
            avxtempDRe[1][1] = select_high_low_epi16(xtemp2,avxtempBIm[0][1],avxtempDRe[1][1],avxtempZIm[1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR6*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempDRe[1][0] = _mm512_mulhrs_epi16(avxtempDRe[1][0],avxtempZRe[0]);\
            avxtempDRe[1][0] = limit_to_saturated_range(avxtempDRe[1][0], llr_range_low, llr_range_high);\
            avxtempDRe[1][1] = _mm512_mulhrs_epi16(avxtempDRe[1][1],avxtempZRe[1]);\
            avxtempDRe[1][1] = limit_to_saturated_range(avxtempDRe[1][1], llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempARe[1][0],\
                                        avxtempBRe[1][0]);\
            xRe[1] = _mm512_packs_epi16(avxtempCRe[1][0],\
                                        avxtempDRe[1][0]);\
            xIm[0] = _mm512_packs_epi16(avxtempARe[1][1],\
                                        avxtempBRe[1][1]);\
            xIm[1] = _mm512_packs_epi16(avxtempCRe[1][1],\
                                        avxtempDRe[1][1]);\
            /*Only store the data LLRs*/\
            avxtempCIm[0][0] = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xRe[1]);\
            _mm512_storeu_si512 (ppSoft, avxtempCIm[0][0]);\
            \
            /*Only store the data LLRs*/\
            avxtempCIm[0][1]= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xRe[1]);\
            _mm512_storeu_si512 (ppSoft+64, avxtempCIm[0][1]);\
            \
            ppSoft = ppSoft + 128;\
            break;\
        }\
        default:\
        {\
            printf("\nInvalid modulation request, mod order %d\n",modOrder);\
            exit(-1);\
        }\
    }\
}

#define llr_demap_2layer_tail_data_dmrs_type1_inter(modOrder, nDmrsPortIdx)\
{\
    /*Move DMRS symbols to the upper bytes by using permute.*/\
    /*Only store LLRs from actual data sub-carriers and ignore DMRS, total length will always be 1/2 amount*/\
    nRestLen_inter = nRestLen >> 1;\
    if ((nDmrsPortIdx == 0) || (nDmrsPortIdx == 1))\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempPostSINR[0]);\
        ftempPostSINR_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempPostSINR[1]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port01_idx, avxxTxSymbol[0]);\
        avxxTxSymbol_inter[1] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port01_idx, avxxTxSymbol[1]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempGain[0]);\
            ftempGain_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port01_idx, ftempGain[1]);\
        }\
    }\
    else\
    {\
        ftempPostSINR_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempPostSINR[0]);\
        ftempPostSINR_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempPostSINR[1]);\
        avxxTxSymbol_inter[0] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port23_idx, avxxTxSymbol[0]);\
        avxxTxSymbol_inter[1] = _mm512_permutexvar_epi32(data_dmrs_inter_type1_port23_idx, avxxTxSymbol[1]);\
        if ((modOrder == BBLIB_QAM16) || (modOrder == BBLIB_QAM64) || (modOrder == BBLIB_QAM256))\
        {\
            ftempGain_inter[0] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempGain[0]);\
            ftempGain_inter[1] = _mm512_permutexvar_ps(data_dmrs_inter_type1_port23_idx, ftempGain[1]);\
        }\
    }\
    \
    /* layermapping for data */\
    xtemp1 = _mm512_permutex2var_epi32(avxxTxSymbol_inter[0], m512_permutex_each32_first256, avxxTxSymbol_inter[1]);\
    avxxTxSymbol_inter[1] = _mm512_permutex2var_epi32(avxxTxSymbol_inter[0], m512_permutex_each32_second256, avxxTxSymbol_inter[1]);\
    avxxTxSymbol_inter[0] = xtemp1;\
    \
    switch(modOrder)\
    {\
        case BBLIB_QPSK:\
        {\
            /*Calculate new mask for storing since we removed DMRS*/\
            nLlrStoreK_inter[0] = 0xffffffffU>>(32-nRestLen_inter*2);\
            /*layermapping for snr*/\
            /*xRe:1/(1-beta), 16Sx*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            /*Algorithm:*/\
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/\
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/\
            \
            /*output 16S13*16S13->16S11*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(avxxTxSymbol_inter[0], i_p_2_m_sqrt2);\
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(avxtempARe[0][0],xRe[0]);\
            /*do left shift according to requried output range*/\
            avxtempARe[0][0] = limit_to_saturated_range(avxtempARe[0][0], llr_range_low, llr_range_high);\
            \
            avxtempARe[0][1] = _mm512_mulhrs_epi16(avxxTxSymbol_inter[1], i_p_2_m_sqrt2);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(avxtempARe[0][1],xRe[1]);\
            avxtempARe[0][1] = limit_to_saturated_range(avxtempARe[0][1], llr_range_low, llr_range_high);\
            \
            avxtempARe[0][0] = _mm512_packs_epi16(avxtempARe[0][0], avxtempARe[0][1]);\
            \
            avxtempARe[0][0] = _mm512_permutexvar_epi64(qpsk_perm_idx,avxtempARe[0][0]);\
            \
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0],avxtempARe[0][0]);\
            ppSoft += nRestLen_inter*4;\
            break;\
        }\
        case BBLIB_QAM16:\
        {\
            /*Calculate new mask for storing since we removed DMRS*/\
            nLlrStoreK_inter[0] = (nRestLen_inter>=8)?0xffffffffU:(0xffffffffU>>(32-nRestLen_inter*4));\
            nLlrStoreK_inter[1] = (nRestLen_inter>8)?(0xffffffffU>>(32-(nRestLen_inter-8)*4)):0;\
            /*permute the beta and postSNR as well*/\
            /*xRe:1/(1-beta), 16Sx*/\
            /*xIm:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempGain_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta\
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta\
            \
            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta\
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta\
            \
            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0\
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0\
            \
            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0\
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/\
            \
            /*first calculate the threshold 2*beta/sqrt(10)*/\
            /*output: 16S15*16S13->16S13*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt10);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_2_d_sqrt10);\
            \
            /*calculate the factor 4/sqrt(10)/(1-beta)*/\
            /*output: 16S13*16Sx->16S(x-2)*/\
            avxtempBRe[0][0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt10);\
            avxtempBRe[0][1] = _mm512_mulhrs_epi16(xRe[1],i_p_4_d_sqrt10);\
            \
            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/\
            avxtempARe[1][0] = copy_inverted_sign_epi16(avxxTxSymbol_inter[0],_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt10));\
            avxtempARe[1][1] = copy_inverted_sign_epi16(avxxTxSymbol_inter[1],_mm512_mulhrs_epi16(xIm[1],i_p_1_d_sqrt10));\
            \
            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/\
            /*output: 16S13*16S13->16S11*/\
            avxtempAIm[0][0] = _mm512_adds_epi16(avxxTxSymbol_inter[0],avxtempARe[1][0]);\
            avxtempAIm[0][0] = _mm512_mulhrs_epi16(avxtempAIm[0][0],i_p_8_d_sqrt10);\
            avxtempARe[1][0] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol_inter[0]),avxtempARe[0][0],avxtempAIm[0][0],_mm512_mulhrs_epi16(avxxTxSymbol_inter[0],i_p_4_d_sqrt10));\
            \
            avxtempAIm[0][1] = _mm512_adds_epi16(avxxTxSymbol_inter[1],avxtempARe[1][1]);\
            avxtempAIm[0][1] = _mm512_mulhrs_epi16(avxtempAIm[0][1],i_p_8_d_sqrt10);\
            avxtempARe[1][1] = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol_inter[1]),avxtempARe[0][1],avxtempAIm[0][1],_mm512_mulhrs_epi16(avxxTxSymbol_inter[1],i_p_4_d_sqrt10));\
            \
            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[1][0] = _mm512_mulhrs_epi16(avxtempARe[1][0],xRe[0]);\
            avxtempARe[1][1] = _mm512_mulhrs_epi16(avxtempARe[1][1],xRe[1]);\
            avxtempARe[1][0] = limit_to_saturated_range(avxtempARe[1][0], llr_range_low, llr_range_high);\
            avxtempARe[1][1] = limit_to_saturated_range(avxtempARe[1][1], llr_range_low, llr_range_high);\
            \
            /*calculate LLR2 and LLR3*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempBRe[1][0] = _mm512_subs_epi16(avxtempARe[0][0],_mm512_abs_epi16(avxxTxSymbol_inter[0]));\
            avxtempBRe[1][0] = _mm512_mulhrs_epi16(avxtempBRe[1][0],avxtempBRe[0][0]);\
            avxtempBRe[1][0] = limit_to_saturated_range(avxtempBRe[1][0], llr_range_low, llr_range_high);\
            \
            avxtempBRe[1][1] = _mm512_subs_epi16(avxtempARe[0][1],_mm512_abs_epi16(avxxTxSymbol_inter[1]));\
            avxtempBRe[1][1] = _mm512_mulhrs_epi16(avxtempBRe[1][1],avxtempBRe[0][1]);\
            avxtempBRe[1][1] = limit_to_saturated_range(avxtempBRe[1][1], llr_range_low, llr_range_high);\
            \
            /*pack to int8 and permute*/\
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/\
            /*xRe[0]: LLR0_16,LLR1_16,LLR0_17,LLR1_17... LLR3_31*/\
            xRe[0] = _mm512_packs_epi16(avxtempARe[1][0],avxtempBRe[1][0]);\
            xIm[0] = _mm512_packs_epi16(avxtempARe[1][1],avxtempBRe[1][1]);\
            \
            xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);\
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0],xRe[0]);\
            \
            xIm[0] = _mm512_shuffle_epi8(xIm[0], qam16_shuffle_idx);\
            _mm512_mask_storeu_epi16 (ppSoft+64,nLlrStoreK_inter[1],xIm[0]);\
            ppSoft = ppSoft + nRestLen_inter*8;\
            \
            break;\
        }\
        case BBLIB_QAM64:\
        {\
            /*Calculate new mask for storing since we removed DMRS*/\
            nLlrStoreK_inter[0] = (nRestLen_inter>5)?0xffffffffU:(0xffffffffU>>(32-nRestLen_inter*6));\
            nLlrStoreK_inter[1] = (nRestLen_inter>5)?((nRestLen>10)?0xffffffffU:(0xffffffffU>>(28-(nRestLen_inter-6)*6))):0;\
            nLlrStoreK_inter[2] = (nRestLen_inter>10)?(0xffffffffU>>(30-(nRestLen_inter-11)*6)):0;\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe:1/(1-beta), 16Sx*/\
            /*xIm:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempGain_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)\
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)\
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)\
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)\
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0\
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)\
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)\
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)\
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)\
            \
            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(42)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt42);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_2_d_sqrt42);\
            \
            /*calculate the threshold 4*beta/sqrt(42)*/\
            avxtempAIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt42);\
            avxtempAIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_4_d_sqrt42);\
            \
            /*calculate the threshold 6*beta/sqrt(42)*/\
            avxtempBRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt42);\
            avxtempBRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_6_d_sqrt42);\
            \
            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt42);\
            avxtempZRe[1] = _mm512_mulhrs_epi16(xRe[1],i_p_4_d_sqrt42);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            xtemp2 = _mm512_abs_epi16(avxxTxSymbol_inter[1]);\
            \
            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42));\
            avxtempBIm[0][0] = _mm512_mulhrs_epi16(avxtempBIm[0][0],i_p_16_d_sqrt42);\
            avxtempBIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt42));\
            avxtempBIm[0][1] = _mm512_mulhrs_epi16(avxtempBIm[0][1],i_p_16_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/\
            avxtempCRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempARe[0][0]);\
            avxtempCRe[0][0] = _mm512_mulhrs_epi16(avxtempCRe[0][0],i_p_12_d_sqrt42);\
            avxtempCRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempARe[0][1]);\
            avxtempCRe[0][1] = _mm512_mulhrs_epi16(avxtempCRe[0][1],i_p_12_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/\
            avxtempCIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt42));\
            avxtempCIm[0][0] = _mm512_mulhrs_epi16(avxtempCIm[0][0],i_p_8_d_sqrt42);\
            avxtempCIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_1_d_sqrt42));\
            avxtempCIm[0][1] = _mm512_mulhrs_epi16(avxtempCIm[0][1],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempBIm[0][0],avxtempCRe[0][0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempBIm[0][1],avxtempCRe[0][1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempCIm[0][0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempCIm[0][1],_mm512_mulhrs_epi16(xtemp2,i_p_4_d_sqrt42));\
            \
            /*if abs(real(In))>4*beta/sqrt(42), return avxtempARe, otherwise return avxtempZIm*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempARe[1][0],avxtempZIm[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempARe[1][1],avxtempZIm[1]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempARe[1][0] = copy_sign_epi16(avxxTxSymbol_inter[0],avxtempARe[1][0]);\
            avxtempARe[1][1] = copy_sign_epi16(avxxTxSymbol_inter[1],avxtempARe[1][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[1][0] = _mm512_mulhrs_epi16(avxtempARe[1][0],xRe[0]);\
            \
            avxtempARe[1][1] = _mm512_mulhrs_epi16(avxtempARe[1][1],xRe[1]);\
            \
            avxtempARe[1][0] = limit_to_saturated_range(avxtempARe[1][0], llr_range_low, llr_range_high);\
            avxtempARe[1][1] = limit_to_saturated_range(avxtempARe[1][1], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempBIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt42),xtemp1);\
            avxtempBIm[0][0] = _mm512_mulhrs_epi16(avxtempBIm[0][0],i_p_8_d_sqrt42);\
            avxtempBIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt42),xtemp2);\
            avxtempBIm[0][1] = _mm512_mulhrs_epi16(avxtempBIm[0][1],i_p_8_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCRe[0][0] = _mm512_subs_epi16(avxtempAIm[0][0],xtemp1);\
            avxtempCRe[0][0] = _mm512_mulhrs_epi16(avxtempCRe[0][0],i_p_4_d_sqrt42);\
            avxtempCRe[0][1] = _mm512_subs_epi16(avxtempAIm[0][1],xtemp2);\
            avxtempCRe[0][1] = _mm512_mulhrs_epi16(avxtempCRe[0][1],i_p_4_d_sqrt42);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/\
            avxtempCIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt42),xtemp1);\
            avxtempCIm[0][0] = _mm512_mulhrs_epi16(avxtempCIm[0][0],i_p_8_d_sqrt42);\
            avxtempCIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt42),xtemp2);\
            avxtempCIm[0][1] = _mm512_mulhrs_epi16(avxtempCIm[0][1],i_p_8_d_sqrt42);\
            \
            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempBIm[0][0],avxtempCRe[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempBIm[0][1],avxtempCRe[0][1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempBRe[1][0],avxtempCIm[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempBRe[1][1],avxtempCIm[0][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempBRe[1][0] = _mm512_mulhrs_epi16(avxtempBRe[1][0],xRe[0]);\
            avxtempBRe[1][0] = limit_to_saturated_range(avxtempBRe[1][0], llr_range_low, llr_range_high);\
            avxtempBRe[1][1] = _mm512_mulhrs_epi16(avxtempBRe[1][1],xRe[1]);\
            avxtempBRe[1][1] = limit_to_saturated_range(avxtempBRe[1][1], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/\
            /*output 16S13*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],_mm512_subs_epi16(avxtempBRe[0][0],xtemp1),_mm512_subs_epi16(xtemp1,avxtempARe[0][0]));\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],_mm512_subs_epi16(avxtempBRe[0][1],xtemp2),_mm512_subs_epi16(xtemp2,avxtempARe[0][1]));\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempCRe[1][0] = _mm512_mulhrs_epi16(avxtempCRe[1][0],avxtempZRe[0]);\
            avxtempCRe[1][0] = limit_to_saturated_range(avxtempCRe[1][0], llr_range_low, llr_range_high);\
            avxtempCRe[1][1] = _mm512_mulhrs_epi16(avxtempCRe[1][1],avxtempZRe[1]);\
            avxtempCRe[1][1] = limit_to_saturated_range(avxtempCRe[1][1], llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempARe[1][0],\
                                        avxtempBRe[1][0]);\
            xRe[1] = _mm512_packs_epi16(avxtempCRe[1][0],\
                                        alliZero);\
            xIm[0] = _mm512_packs_epi16(alliZero,\
                                        avxtempARe[1][1]);\
            xIm[1] = _mm512_packs_epi16(avxtempBRe[1][1],\
                                        avxtempCRe[1][1]);\
            \
            avxtempCIm[0][0] = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xRe[1]);\
            _mm512_mask_storeu_epi16(ppSoft,nLlrStoreK_inter[0],avxtempCIm[0][0]);\
            \
            avxtempCIm[0][1] = _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xRe[1]);\
            avxtempCIm[1][0] = _mm512_permutex2var_epi16(xIm[0], qam64_third_32, xIm[1]);\
            avxtempCIm[1][1] = _mm512_permutex2var_epi16(xIm[0], qam64_last_64, xIm[1]);\
            _mm512_mask_storeu_epi16(ppSoft+128,nLlrStoreK_inter[2],avxtempCIm[1][1]);\
            \
            avxtempCIm[0][1] = _mm512_permutex2var_epi64(avxtempCIm[0][1],qam64_index_lo_hi,avxtempCIm[1][0]);\
            _mm512_mask_storeu_epi16(ppSoft+64,nLlrStoreK_inter[1],avxtempCIm[0][1]);\
            \
            ppSoft = ppSoft + nRestLen_inter*12;\
            \
            break;\
        }\
        case BBLIB_QAM256:\
        {\
            /*Calculate new mask for storing since we removed DMRS*/\
            nLlrStoreK_inter[0] = (nRestLen_inter>4)?0xffffffffU:(0xffffffffU>>(32-nRestLen_inter*8));\
            nLlrStoreK_inter[1] = (nRestLen_inter>4)?((nRestLen_inter > 8)?0xffffffffU:(0xffffffffU>>(32-(nRestLen_inter-4)*8))):0;\
            nLlrStoreK_inter[2] = (nRestLen_inter>8)?((nRestLen_inter > 12)?0xffffffffU:(0xffffffffU>>(32-(nRestLen_inter-8)*8))):0;\
            nLlrStoreK_inter[3] = (nRestLen_inter>12)?(0xffffffffU>>(32-(nRestLen_inter-12)*8)):0;\
            /*convert the beta and postSNR from flp to fxp*/\
            /*xRe:1/(1-beta), 16Sx*/\
            /*xIm:beta,       16S15 0~0.9999*/\
            xtemp1 = _mm512_cvtps_epi32(ftempPostSINR_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempPostSINR_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            xtemp1 = _mm512_cvtps_epi32(ftempGain_inter[0]);\
            xtemp2 = _mm512_cvtps_epi32(ftempGain_inter[1]);\
            xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);\
            \
            xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);\
            xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);\
            \
            \
            /*Algorithm:\
            LLR0 = InnerCompoundReal0/(1-beta)\
            InnerCompoundReal0_0 = 32/sqrt(170)*(real(In)+InnerFactor0) when abs(real(In))>14*beta/sqrt(170)\
            InnerCompoundReal0_1 = 28/sqrt(170)*(real(In)+InnerFactor1) when 14*beta/sqrt(170)<abs(real(In))<=12*beta/sqrt(170)\
            InnerCompoundReal0_2 = 24/sqrt(170)*(real(In)r+InnerFactor2) when 12*beta/sqrt(170)<abs(real(In))<=10*beta/sqrt(170)\
            InnerCompoundReal0_3 = 20/sqrt(170)*(real(In)r+InnerFactor3) when 10*beta/sqrt(170)<abs(real(In))<=8*beta/sqrt(170)\
            InnerCompoundReal0_4 = 16/sqrt(170)*(real(In)r+InnerFactor4) when 8*beta/sqrt(170)<abs(real(In))<=6*beta/sqrt(170)\
            InnerCompoundReal0_5 = 12/sqrt(170)*(real(In)r+InnerFactor5) when 6*beta/sqrt(170)<abs(real(In))<=4*beta/sqrt(170)\
            InnerCompoundReal0_6 = 8/sqrt(170)*(real(In)r+InnerFactor6) when 4*beta/sqrt(170)<abs(real(In))<=2*beta/sqrt(170)\
            InnerCompoundReal0_7 = 4/sqrt(170)*real(In) when abs(real(In))<=2*beta/sqrt(170)\
            InnerFactor0 = 7*beta/(sqrt(170)); InnerFactor1 = 6*beta/(sqrt(170));InnerFactor2 = 5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = 4*beta/(sqrt(170)); InnerFactor4 = 3*beta/(sqrt(170));InnerFactor5 = 2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = beta/(sqrt(170)), when real(In)<0\
            InnerFactor0 = -7*beta/(sqrt(170)); InnerFactor1 = -6*beta/(sqrt(170));InnerFactor2 = -5*beta/(sqrt(170)),when real(In)<0\
            InnerFactor3 = -4*beta/(sqrt(170)); InnerFactor4 = -3*beta/(sqrt(170));InnerFactor5 = -2*beta/(sqrt(170)),when real(In)<0\
            InnerFactor6 = -beta/(sqrt(170)), when real(In)<0\
            \
            LLR1 = InnerCompoundImag0/(1-beta)\
            \
            \
            LLR2 = InnerCompoundReal1/(1-beta)\
            InnerCompoundReal1_0 = 16/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170)) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal1_1 = 12/sqrt(170)*(-abs(real(r))+10*beta/sqrt(170)) when 12*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal1_2 = 8/sqrt(170)*(-abs(real(r))+9*beta/sqrt(170)) when 10*beta/sqrt(170)<=abs(real(r))<=12*beta/sqrt(170)\
            InnerCompoundReal1_3 = 4/sqrt(170)*(-abs(real(r))+8*beta/sqrt(170)) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal1_4 = 8/sqrt(170)*(-abs(real(r))+7*beta/sqrt(170)) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal1_5 = 12/sqrt(170)*(-abs(real(r))+6*beta/sqrt(170)) when 4*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal1_6 = 16/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170)) when 2*beta/sqrt(170)<=abs(real(r))\
            \
            LLR3 = InnerCompoundImag1/(1-beta)\
            \
            LLR4 = InnerCompoundReal2/(1-beta)\
            InnerCompoundReal2_0 = 8/sqrt(170)*(-abs(real(r))+13*beta/sqrt(170) when abs(real(r))>=14*beta/sqrt(170)\
            InnerCompoundReal2_1 = 4/sqrt(170)*(-abs(real(r))+12*beta/sqrt(170) when 10*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)\
            InnerCompoundReal2_2 = 8/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)\
            InnerCompoundReal2_3 = -8/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)\
            InnerCompoundReal2_4 = -4/sqrt(170)*(-abs(real(r))+4*beta/sqrt(170) when 2*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)\
            InnerCompoundReal2_5 = -8/sqrt(170)*(-abs(real(r))+3*beta/sqrt(170) when 2*beta/sqrt(170)>=abs(real(r))\
            \
            LLR5 = InnerCompoundImag2/(1-beta)\
            \
            LLR6 = 4/sqrt(170)*InnerCompoundReal3/(1-beta)\
            InnerCompoundReal3_0 = -1*abs(real(r))+14*beta/sqrt(170) when abs(real(r))>=12*beta/sqrt(42)\
            InnerCompoundReal3_1 = -1*abs(real(r))+10*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<12*beta/sqrt(42)\
            InnerCompoundReal3_2 = -1*abs(real(r))+6*beta/sqrt(170) when 4*beta/sqrt(170)<=abs(real(r))<8*beta/sqrt(42)\
            InnerCompoundReal3_3 = -1*abs(real(r))+2*beta/sqrt(170) when abs(real(r))<4*beta/sqrt(170)\
            \
            LLR7 = 4/sqrt(170)*InnerCompoundImag3/(1-beta)*/\
            \
            /*1.calculate the thresholds and scaling factors*/\
            /*calculate the threshold 2*beta/sqrt(170)*/\
            /*output 16S15*16S13->16S13*/\
            avxtempARe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_2_d_sqrt170);\
            avxtempARe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_2_d_sqrt170);\
            \
            /*calculate the threshold 4*beta/sqrt(170)*/\
            avxtempAIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_4_d_sqrt170);\
            avxtempAIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_4_d_sqrt170);\
            \
            /*calculate the threshold 6*beta/sqrt(170)*/\
            avxtempBRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_6_d_sqrt170);\
            avxtempBRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_6_d_sqrt170);\
            \
            /*calculate the threshold 8*beta/sqrt(170)*/\
            avxtempBIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_8_d_sqrt170);\
            avxtempBIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_8_d_sqrt170);\
            \
            /*calculate the threshold 10*beta/sqrt(170)*/\
            avxtempCRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_10_d_sqrt170);\
            avxtempCRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_10_d_sqrt170);\
            \
            /*calculate the threshold 12*beta/sqrt(170)*/\
            avxtempCIm[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_12_d_sqrt170);\
            avxtempCIm[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_12_d_sqrt170);\
            \
            /*calculate the threshold 14*beta/sqrt(170)*/\
            avxtempDRe[0][0] = _mm512_mulhrs_epi16(xIm[0],i_p_14_d_sqrt170);\
            avxtempDRe[0][1] = _mm512_mulhrs_epi16(xIm[1],i_p_14_d_sqrt170);\
            \
            /*calculate the common scaling factor 4/sqrt(170)/(1-beta)*/\
            /*output 16S13*16Sx->16S(x-2)*/\
            avxtempZRe[0] = _mm512_mulhrs_epi16(xRe[0],i_p_4_d_sqrt170);\
            avxtempZRe[1] = _mm512_mulhrs_epi16(xRe[1],i_p_4_d_sqrt170);\
            \
            /*2.start to calculate LLR0*/\
            /*abs: 16S13*/\
            xtemp1 = _mm512_abs_epi16(avxxTxSymbol_inter[0]);\
            xtemp2 = _mm512_abs_epi16(avxxTxSymbol_inter[1]);\
            \
            /*InnerCompoundReal0_0 = 32/sqrt(170)*(abs(real(In))-7*beta/sqrt(170))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170));\
            avxtempDIm[0][0] = _mm512_mulhrs_epi16(avxtempDIm[0][0],i_p_32_d_sqrt170);\
            avxtempDIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_7_d_sqrt170));\
            avxtempDIm[0][1] = _mm512_mulhrs_epi16(avxtempDIm[0][1],i_p_32_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_1: 28/sqrt(170)*(abs(real(In))- 6*beta/sqrt(170))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(xtemp1,avxtempBRe[0][0]);\
            avxtempERe[0][0] = _mm512_mulhrs_epi16(avxtempERe[0][0],i_p_28_d_sqrt170);\
            avxtempERe[0][1] = _mm512_subs_epi16(xtemp2,avxtempBRe[0][1]);\
            avxtempERe[0][1] = _mm512_mulhrs_epi16(avxtempERe[0][1],i_p_28_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_2: 24/sqrt(170)*(abs(real(In))-5*beta/sqrt(170))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempEIm[0][0] = _mm512_mulhrs_epi16(avxtempEIm[0][0],i_p_24_d_sqrt170);\
            avxtempEIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt170));\
            avxtempEIm[0][1] = _mm512_mulhrs_epi16(avxtempEIm[0][1],i_p_24_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_3: 20/sqrt(170)*(abs(real(In))-4*beta/sqrt(170))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0][0]);\
            avxtempFRe[0][0] = _mm512_mulhrs_epi16(avxtempFRe[0][0],i_p_20_d_sqrt170);\
            avxtempFRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempAIm[0][1]);\
            avxtempFRe[0][1] = _mm512_mulhrs_epi16(avxtempFRe[0][1],i_p_20_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_4: 16/sqrt(170)*(abs(real(In))-3*beta/sqrt(170))*/\
            avxtempFIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempFIm[0][0] = _mm512_mulhrs_epi16(avxtempFIm[0][0],i_p_16_d_sqrt170);\
            avxtempFIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt170));\
            avxtempFIm[0][1] = _mm512_mulhrs_epi16(avxtempFIm[0][1],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_5: 12/sqrt(170)*(abs(real(In))-2*beta/sqrt(170))*/\
            avxtempGRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempARe[0][0]);\
            avxtempGRe[0][0] = _mm512_mulhrs_epi16(avxtempGRe[0][0],i_p_12_d_sqrt170);\
            avxtempGRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempARe[0][1]);\
            avxtempGRe[0][1] = _mm512_mulhrs_epi16(avxtempGRe[0][1],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal0_6: 8/sqrt(170)*(abs(real(In))-1*beta/sqrt(170))*/\
            avxtempGIm[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_1_d_sqrt170));\
            avxtempGIm[0][0] = _mm512_mulhrs_epi16(avxtempGIm[0][0],i_p_8_d_sqrt170);\
            avxtempGIm[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_1_d_sqrt170));\
            avxtempGIm[0][1] = _mm512_mulhrs_epi16(avxtempGIm[0][1],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempDRe[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempDRe[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal0_2, otherwise return InnerCompoundReal_0_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempCRe[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_0_1*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempCIm[0][0],avxtempARe[1][0],avxtempZIm[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempCIm[0][1],avxtempARe[1][1],avxtempZIm[1]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal0_4, otherwise return InnerCompoundReal_0_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempFIm[0][0],avxtempGRe[0][0]);\
            avxtempHRe[1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempFIm[0][1],avxtempGRe[0][1]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempARe, otherwise return avxtempBRe*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempBIm[0][0],avxtempARe[1][0],avxtempHRe[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempBIm[0][1],avxtempARe[1][1],avxtempHRe[1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal0_6, otherwise return 4/sqrt(170)*abs(real(In))*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempGIm[0][0],_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt170));\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempGIm[0][1],_mm512_mulhrs_epi16(xtemp2,i_p_4_d_sqrt170));\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return avxtempARe, otherwise return avxtempBRe*/\
            avxtempARe[1][0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempARe[1][0],avxtempZIm[0]);\
            avxtempARe[1][1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempARe[1][1],avxtempZIm[1]);\
            \
            /*convert sign of InnferFactor according to the sign of real(In)*/\
            avxtempARe[1][0] = copy_sign_epi16(avxxTxSymbol_inter[0],avxtempARe[1][0]);\
            avxtempARe[1][1] = copy_sign_epi16(avxxTxSymbol_inter[1],avxtempARe[1][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempARe[1][0] = _mm512_mulhrs_epi16(avxtempARe[1][0],xRe[0]);\
            \
            avxtempARe[1][1] = _mm512_mulhrs_epi16(avxtempARe[1][1],xRe[1]);\
            \
            avxtempARe[1][0] = limit_to_saturated_range(avxtempARe[1][0], llr_range_low, llr_range_high);\
            avxtempARe[1][1] = limit_to_saturated_range(avxtempARe[1][1], llr_range_low, llr_range_high);\
            \
            /*3.start to calculate LLR2*/\
            /*calculate the the InnerCompoundReal1_0: 16/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*16S13->16S11*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempDIm[0][0] = _mm512_mulhrs_epi16(avxtempDIm[0][0],i_p_16_d_sqrt170);\
            avxtempDIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_11_d_sqrt170),xtemp2);\
            avxtempDIm[0][1] = _mm512_mulhrs_epi16(avxtempDIm[0][1],i_p_16_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_1: 12/sqrt(170)*(10*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(avxtempCRe[0][0],xtemp1);\
            avxtempERe[0][0] = _mm512_mulhrs_epi16(avxtempERe[0][0],i_p_12_d_sqrt170);\
            avxtempERe[0][1] = _mm512_subs_epi16(avxtempCRe[0][1],xtemp2);\
            avxtempERe[0][1] = _mm512_mulhrs_epi16(avxtempERe[0][1],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_2: 8/sqrt(170)*(9*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_9_d_sqrt170),xtemp1);\
            avxtempEIm[0][0] = _mm512_mulhrs_epi16(avxtempEIm[0][0],i_p_8_d_sqrt170);\
            avxtempEIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_9_d_sqrt170),xtemp2);\
            avxtempEIm[0][1] = _mm512_mulhrs_epi16(avxtempEIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_3: 4/sqrt(170)*(8*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(avxtempBIm[0][0],xtemp1);\
            avxtempFRe[0][0] = _mm512_mulhrs_epi16(avxtempFRe[0][0],i_p_4_d_sqrt170);\
            avxtempFRe[0][1] = _mm512_subs_epi16(avxtempBIm[0][1],xtemp2);\
            avxtempFRe[0][1] = _mm512_mulhrs_epi16(avxtempFRe[0][1],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_4: 8/sqrt(170)*(7*beta/sqrt(170)-abs(real(In)))*/\
            avxtempFIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_7_d_sqrt170),xtemp1);\
            avxtempFIm[0][0] = _mm512_mulhrs_epi16(avxtempFIm[0][0],i_p_8_d_sqrt170);\
            avxtempFIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_7_d_sqrt170),xtemp2);\
            avxtempFIm[0][1] = _mm512_mulhrs_epi16(avxtempFIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 12/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGRe[0][0] = _mm512_subs_epi16(avxtempBRe[0][0],xtemp1);\
            avxtempGRe[0][0] = _mm512_mulhrs_epi16(avxtempGRe[0][0],i_p_12_d_sqrt170);\
            avxtempGRe[0][1] = _mm512_subs_epi16(avxtempBRe[0][1],xtemp2);\
            avxtempGRe[0][1] = _mm512_mulhrs_epi16(avxtempGRe[0][1],i_p_12_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal1_5: 16/sqrt(170)*(5*beta/sqrt(170)-abs(real(In)))*/\
            avxtempGIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170),xtemp1);\
            avxtempGIm[0][0] = _mm512_mulhrs_epi16(avxtempGIm[0][0],i_p_16_d_sqrt170);\
            avxtempGIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt170),xtemp2);\
            avxtempGIm[0][1] = _mm512_mulhrs_epi16(avxtempGIm[0][1],i_p_16_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempDRe[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempDRe[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal1_2, otherwise return InnerCompoundReal_1_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempCRe[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempCRe[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_1_1*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempCIm[0][0],avxtempBRe[1][0],avxtempZIm[0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempCIm[0][1],avxtempBRe[1][1],avxtempZIm[1]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempFIm[0][0],avxtempGRe[0][0]);\
            avxtempHRe[1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempFIm[0][1],avxtempGRe[0][1]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempBRe[1][0],avxtempHRe[0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempBRe[1][1],avxtempHRe[1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal1_5, otherwise return InnerCompoundReal1_6*/\
            avxtempBRe[1][0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempBRe[1][0],avxtempGIm[0][0]);\
            avxtempBRe[1][1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempBRe[1][1],avxtempGIm[0][1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/\
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/\
            avxtempBRe[1][0] = _mm512_mulhrs_epi16(avxtempBRe[1][0],xRe[0]);\
            avxtempBRe[1][0] = limit_to_saturated_range(avxtempBRe[1][0], llr_range_low, llr_range_high);\
            avxtempBRe[1][1] = _mm512_mulhrs_epi16(avxtempBRe[1][1],xRe[1]);\
            avxtempBRe[1][1] = limit_to_saturated_range(avxtempBRe[1][1], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR4*/\
            /*calculate the the InnerCompoundReal2_0: 8/sqrt(170)*(13*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_13_d_sqrt170),xtemp1);\
            avxtempDIm[0][0] = _mm512_mulhrs_epi16(avxtempDIm[0][0],i_p_8_d_sqrt170);\
            avxtempDIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_13_d_sqrt170),xtemp2);\
            avxtempDIm[0][1] = _mm512_mulhrs_epi16(avxtempDIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_1: 4/sqrt(170)*(12*beta/sqrt(170)-abs(real(In)))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(avxtempCIm[0][0],xtemp1);\
            avxtempERe[0][0] = _mm512_mulhrs_epi16(avxtempERe[0][0],i_p_4_d_sqrt170);\
            avxtempERe[0][1] = _mm512_subs_epi16(avxtempCIm[0][1],xtemp2);\
            avxtempERe[0][1] = _mm512_mulhrs_epi16(avxtempERe[0][1],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_2: 8/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[0],i_p_11_d_sqrt170),xtemp1);\
            avxtempEIm[0][0] = _mm512_mulhrs_epi16(avxtempEIm[0][0],i_p_8_d_sqrt170);\
            avxtempEIm[0][1] = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[1],i_p_11_d_sqrt170),xtemp2);\
            avxtempEIm[0][1] = _mm512_mulhrs_epi16(avxtempEIm[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_3: 8/sqrt(170)*(abs(real(In)-5*beta/sqrt(170))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_5_d_sqrt170));\
            avxtempFRe[0][0] = _mm512_mulhrs_epi16(avxtempFRe[0][0],i_p_8_d_sqrt170);\
            avxtempFRe[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_5_d_sqrt170));\
            avxtempFRe[0][1] = _mm512_mulhrs_epi16(avxtempFRe[0][1],i_p_8_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_4: 4/sqrt(170)*(abs(real(In)-4*beta/sqrt(170))*/\
            avxtempFIm[0][0] = _mm512_subs_epi16(xtemp1,avxtempAIm[0][0]);\
            avxtempFIm[0][0] = _mm512_mulhrs_epi16(avxtempFIm[0][0],i_p_4_d_sqrt170);\
            avxtempFIm[0][1] = _mm512_subs_epi16(xtemp2,avxtempAIm[0][1]);\
            avxtempFIm[0][1] = _mm512_mulhrs_epi16(avxtempFIm[0][1],i_p_4_d_sqrt170);\
            \
            /*calculate the the InnerCompoundReal2_5: 8/sqrt(170)*(abs(real(In)-3*beta/sqrt(170))*/\
            avxtempGRe[0][0] = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[0],i_p_3_d_sqrt170));\
            avxtempGRe[0][0] = _mm512_mulhrs_epi16(avxtempGRe[0][0],i_p_8_d_sqrt170);\
            avxtempGRe[0][1] = _mm512_subs_epi16(xtemp2,_mm512_mulhrs_epi16(xIm[1],i_p_3_d_sqrt170));\
            avxtempGRe[0][1] = _mm512_mulhrs_epi16(avxtempGRe[0][1],i_p_8_d_sqrt170);\
            \
            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal2_0, otherwise return InnerCompoundReal2_1*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempDRe[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempDRe[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return InnerCompoundReal2_2, otherwise return InnerCompoundReal_2_1*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempBIm[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempBIm[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>10*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_1_1*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempCRe[0][0],avxtempCRe[1][0],avxtempZIm[0]);\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempCRe[0][1],avxtempCRe[1][1],avxtempZIm[1]);\
            \
            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal2_4, otherwise return InnerCompoundReal_2_5*/\
            avxtempHRe[0] = select_high_low_epi16(xtemp1,avxtempARe[0][0],avxtempFIm[0][0],avxtempGRe[0][0]);\
            avxtempHRe[1] = select_high_low_epi16(xtemp2,avxtempARe[0][1],avxtempFIm[0][1],avxtempGRe[0][1]);\
            \
            /*if abs(real(In))>6*beta/sqrt(170), return avxtempARe, otherwise return InnerCompoundReal_1_1*/\
            avxtempCRe[1][0] = select_high_low_epi16(xtemp1,avxtempBRe[0][0],avxtempCRe[1][0],avxtempHRe[0]);\
            avxtempCRe[1][1] = select_high_low_epi16(xtemp2,avxtempBRe[0][1],avxtempCRe[1][1],avxtempHRe[1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempCRe[1][0] = _mm512_mulhrs_epi16(avxtempCRe[1][0],xRe[0]);\
            avxtempCRe[1][0] = limit_to_saturated_range(avxtempCRe[1][0], llr_range_low, llr_range_high);\
            avxtempCRe[1][1] = _mm512_mulhrs_epi16(avxtempCRe[1][1],xRe[1]);\
            avxtempCRe[1][1] = limit_to_saturated_range(avxtempCRe[1][1], llr_range_low, llr_range_high);\
            \
            /*4.start to calculate LLR6*/\
            /*calculate the the InnerCompoundReal3_0: 4/sqrt(170)*(14*beta/sqrt(170)-abs(real(In)))*/\
            /*output 16S13*/\
            avxtempDIm[0][0] = _mm512_subs_epi16(avxtempDRe[0][0],xtemp1);\
            avxtempDIm[0][1] = _mm512_subs_epi16(avxtempDRe[0][1],xtemp2);\
            \
            /*calculate the the InnerCompoundReal3_1: 4/sqrt(170)*(abs(real(In)-10*beta/sqrt(170))*/\
            avxtempERe[0][0] = _mm512_subs_epi16(xtemp1,avxtempCRe[0][0]);\
            avxtempERe[0][1] = _mm512_subs_epi16(xtemp2,avxtempCRe[0][1]);\
            \
            /*calculate the the InnerCompoundReal3_2: 4/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/\
            avxtempEIm[0][0] = _mm512_subs_epi16(avxtempBRe[0][0],xtemp1);\
            avxtempEIm[0][1] = _mm512_subs_epi16(avxtempBRe[0][1],xtemp2);\
            \
            /*calculate the the InnerCompoundReal3_3: 4/sqrt(170)*(abs(real(In)-2*beta/sqrt(170))*/\
            avxtempFRe[0][0] = _mm512_subs_epi16(xtemp1,avxtempARe[0][0]);\
            avxtempFRe[0][1] = _mm512_subs_epi16(xtemp2,avxtempARe[0][1]);\
            \
            /*if abs(real(In))>12*beta/sqrt(170), return InnerCompoundReal3_0, otherwise return InnerCompoundReal2_1*/\
            avxtempDRe[1][0] = select_high_low_epi16(xtemp1,avxtempCIm[0][0],avxtempDIm[0][0],avxtempERe[0][0]);\
            avxtempDRe[1][1] = select_high_low_epi16(xtemp2,avxtempCIm[0][1],avxtempDIm[0][1],avxtempERe[0][1]);\
            \
            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal3_2, otherwise return InnerCompoundReal_3_4*/\
            avxtempZIm[0] = select_high_low_epi16(xtemp1,avxtempAIm[0][0],avxtempEIm[0][0],avxtempFRe[0][0]);\
            avxtempZIm[1] = select_high_low_epi16(xtemp2,avxtempAIm[0][1],avxtempEIm[0][1],avxtempFRe[0][1]);\
            \
            /*if abs(real(In))>8*beta/sqrt(170), return avxtempDRe, otherwise return avxtempZIm*/\
            avxtempDRe[1][0] = select_high_low_epi16(xtemp1,avxtempBIm[0][0],avxtempDRe[1][0],avxtempZIm[0]);\
            avxtempDRe[1][1] = select_high_low_epi16(xtemp2,avxtempBIm[0][1],avxtempDRe[1][1],avxtempZIm[1]);\
            \
            /*multiply with the scaling factor 1/(1-beta) to get LLR6*/\
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/\
            avxtempDRe[1][0] = _mm512_mulhrs_epi16(avxtempDRe[1][0],avxtempZRe[0]);\
            avxtempDRe[1][0] = limit_to_saturated_range(avxtempDRe[1][0], llr_range_low, llr_range_high);\
            avxtempDRe[1][1] = _mm512_mulhrs_epi16(avxtempDRe[1][1],avxtempZRe[1]);\
            avxtempDRe[1][1] = limit_to_saturated_range(avxtempDRe[1][1], llr_range_low, llr_range_high);\
            \
            /*8.saturated pack to int8*/\
            xRe[0] = _mm512_packs_epi16(avxtempARe[1][0],\
                                        avxtempBRe[1][0]);\
            xRe[1] = _mm512_packs_epi16(avxtempCRe[1][0],\
                                        avxtempDRe[1][0]);\
            xIm[0] = _mm512_packs_epi16(avxtempARe[1][1],\
                                        avxtempBRe[1][1]);\
            xIm[1] = _mm512_packs_epi16(avxtempCRe[1][1],\
                                        avxtempDRe[1][1]);\
            avxtempCIm[0][0] = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xRe[1]);\
            _mm512_mask_storeu_epi16 (ppSoft,nLlrStoreK_inter[0], avxtempCIm[0][0]);\
            \
            avxtempCIm[0][1]= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xRe[1]);\
            _mm512_mask_storeu_epi16 (ppSoft+64,nLlrStoreK_inter[1], avxtempCIm[0][1]);\
            \
            avxtempCIm[1][0] = _mm512_permutex2var_epi16(xIm[0], qam256_first_64, xIm[1]);\
            _mm512_mask_storeu_epi16 (ppSoft+128,nLlrStoreK_inter[2], avxtempCIm[1][0]);\
            \
            avxtempCIm[1][1]= _mm512_permutex2var_epi16(xIm[0], qam256_second_64, xIm[1]);\
            _mm512_mask_storeu_epi16 (ppSoft+192,nLlrStoreK_inter[3], avxtempCIm[1][1]);\
            ppSoft = ppSoft + nRestLen_inter * 16;\
            break;\
        }\
        default:\
        {\
            printf("\nInvalid modulation request, mod order %d\n",modOrder);\
            exit(-1);\
        }\
    }\
}
#endif

#endif
