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
#include "simd_insts.hpp"
#include "phy_pusch_symbol_processing_5gnr.h"
#include "phy_pusch_symbol_processing_5gnr_avx512.h"
#include "phy_tafo_table_gen.h"
#include "matrix.hpp"
//#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
//#include <ipps.h>
//#include <ippcore.h>
//#include <ippvm.h>
//#endif
#ifdef _BBLIB_AVX512_
#define SAVE_GAIN (0)
#define DIG_AGC (1)
#define AGC_OVERFLOW_DET (0)
using namespace PUSCH_SYMBOL_PROCESS;

//using namespace W_SDK;
// static int16_t gKRErefTable[2][4][6] = {{{0,2,1,3,0,0},{2,4,3,5,0,0},{6,8,7,9,0,0},{8,10,9,11,0,0}},{{0,1,2,3,4,5},{1,6,3,8,5,10},{6,7,8,9,10,11},{7,0,9,2,11,4}}};
#define NULL_DATA (0xffffffff)
#ifndef PI
#define PI          (3.14159265358979323846)
#endif

// Original tables are in phy_rx_mimo_mmse_avx512.cpp
extern int16_t numgroups[BBLIB_INTERP_GRANS];
extern int16_t numingroup[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t symnumsA[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t symnumsB[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t wType[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t wType_A[6][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t wType_B[4][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t numgroups_2p2_A[BBLIB_INTERP_GRANS];
extern int16_t numgroups_2p2_B[BBLIB_INTERP_GRANS];
extern int16_t numingroup_2p2_A[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t numingroup_2p2_B[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF];
extern int16_t symnums_2p2_A[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t symnums_2p2_B[BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][6];
extern int16_t wType_2p2[2][BBLIB_INTERP_GRANS][BBLIB_N_SYMB_PER_SF][2];
extern int16_t g_flag_symH_upd_optA[6][BBLIB_INTERP_GRANS];
extern int16_t g_flag_symH_upd_optB[4][BBLIB_INTERP_GRANS];
extern int16_t flag_symH_upd_2p2_opt[2][BBLIB_INTERP_GRANS];
extern float FocPhaseFixTable[2][BBLIB_N_SYMB_PER_SF];
// mmse llr mimo linear interpolation
template<typename T, FO_E fo_flag = FO_E::disable, size_t N_RX = 16, size_t N_TX = 16, uint8_t DMRSTYPE = 1,uint8_t NROFCDMS = 2, INTERP_E interp_flag = INTERP_E::disable>
struct SymbolProcess {
    // type define
    using FloatSimd = typename DataType<T>::FloatSimd;
    using Float = typename DataType<T>::Float;
    using procDataType = typename DataType<T>::procDataType;
    const static auto fp16Int16 = DataType<T>::fp16Int16;

    // method
    inline static void data_dmrs_mux(bblib_pusch_symbol_processing_request *request, uint16_t &dataDmrsMux, uint16_t &nDmrsPortIdx) {
        uint16_t nTotalDmrsSymbol = request->nTotalDmrsSymbol;
        uint16_t * pDmrsSymbolIdx = request->pDmrsSymbolIdx;
        uint16_t nDataDmrsInter = 0;
        uint8_t  * pDmrsPortIdx = request->pDmrsPortIdx[0];
        int16_t nMappingType = request->nMappingType;
        //Check if data/dmrs are interleaved for this group
        if constexpr (((DMRSTYPE == 1) && (NROFCDMS == 2)) || ((DMRSTYPE == 2) && (NROFCDMS == 3))) {
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
        FloatSimd &avxShift, FloatSimd &avxfSigma2, float &llr_postsnr_fxp_dynamic, FloatSimd &avxGainShift,
        int16_t &llr_range_high, int16_t &llr_range_low) {
        constexpr auto mmse_x_left = (Float)(1 << BBLIB_MMSE_X_LEFT_SHIFT);

#ifdef _BBLIB_SPR_
        if constexpr (fp16Int16 == FP16_E::FP16) {
            const Float sigma2 = (Float)(static_cast<float>(request->nSigma2));
            avxShift = FloatSimd(mmse_x_left);
            avxfSigma2 = FloatSimd(sigma2, 0.0);
        }
        else
#endif
        {
            const Float sigma2 = (Float)(request->nSigma2);
            const auto left_shift = (N_TX == 4 ? 1.0 / mmse_x_left : mmse_x_left);
            avxShift = FloatSimd(left_shift);
            avxfSigma2 = FloatSimd(sigma2);
            const auto nFactor = 1.0 / (Float)(1 << BBLIB_MMSE_LEMMA_SCALING);
            avxGainShift = (N_TX == 4 ? FloatSimd(nFactor) : FloatSimd(1.0));
        }
        llr_postsnr_fxp_dynamic = ((float)((int16_t)1 << (POSTSNR_FXP_BASE+request->nLlrFxpPoints)));

        llr_range_high = (1 << (request->nLlrSaturatedBits - 1)) - 1;
        llr_range_low = -1 * llr_range_high;
    }
    template<size_t N_CH_NUM>
    static void init_input_addr(int16_t iChSymb, int16_t nStartSymbIndex, bblib_pusch_symbol_processing_request *request,
        T * pChIn[N_CH_NUM][N_TX][N_RX], T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX]) {
        const int16_t nAlignedTotalSubCarrier = request->nTotalAlignedSubCarrier;

        //load H (nRxAnt X 2)
        for (size_t j = 0; j < N_TX; j ++) {
            for (size_t i = 0; i < N_RX; i++) {
                if constexpr (N_CH_NUM == 2) {
                    auto pTmp = ptr_cast<int32_t *>(request->pChState[i][j]);
                    pChIn[0][j][i] = ptr_cast<T *>(pTmp);
                    pChIn[1][j][i] = ptr_cast<T *>(pTmp + nAlignedTotalSubCarrier);
                } else {
                    pChIn[0][j][i] = ptr_cast<T *>(ptr_cast<int32_t *>(request->pChState[i][j]) +
                        iChSymb * nAlignedTotalSubCarrier);
                }
            }
        }
        // convert rx pointer
        if constexpr (N_CH_NUM == 2) {
            const int16_t nDataSymb = request->nSymb;
            for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                auto nDataSymbIdx = request->pSymbIndex[iSymb];
                for (size_t i = 0; i < N_RX; i++) {
                    pRxIn[nDataSymbIdx][i] = reinterpret_cast<T *>
                        (reinterpret_cast<int32_t *>(request->pRxSignal[i][nDataSymbIdx]));
                    // _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T1);
                }
            }
        } else {
            for (int16_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                const auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                for (size_t i = 0; i < N_RX; i++) {
                    pRxIn[nDataSymbIdx][i] = ptr_cast<T *>(ptr_cast<int32_t *>(request->pRxSignal[i][nDataSymbIdx]));
                    // _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T1);
                }
            }
        }
    }

#ifdef _BBLIB_SPR_
    template<INTERP_E interp = INTERP_E::disable>
    FORCE_INLINE inline
    static void tx_calc(uint16_t nDataSymbIdx, uint16_t nDataNextSymbIdx, FloatSimd avxShift,
        FloatSimd ftempBRe[N_TX][N_TX],
        T ChIn[N_TX][N_RX], CI16vec16 *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX],
        Is16vec32 avxxTxSymbol[N_TX], F16vec32 avxAgcScale, T *FoOffsetTable) {

        FloatSimd ftempZRe[N_TX];

        if constexpr(interp == INTERP_E::disable) {
            CI16vec16 rxIn[N_RX];
            #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                //load Y
                rxIn[i] = *pRxIn[nDataSymbIdx][i]++;
                _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T1);
                _mm_prefetch(pRxIn[nDataSymbIdx][i] + 1, _MM_HINT_T1);
            }

            #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i++) {
                // calculate the real part of z
                // calculate the imag part of z
                ftempZRe[i] = acc_sum<N_RX>(ChIn[i], rxIn, avxAgcScale);
            }

            // 4. x = invA * z
            #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i++) {
                auto tx = dot<T, N_TX>(ftempBRe[i], ftempZRe);
                // tx = tx * avxShift;
                tx = _mm512_mul_ph(tx, avxShift);
                if constexpr (fo_flag == FO_E::enable) {
                    tx = fmulconj(tx, FoOffsetTable[i]);
                }
                avxxTxSymbol[i] = _mm512_cvtph_epi16(tx);
            }

        } else {
            #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i++) {
                // calculate the real part of z
                // calculate the imag part of z
                ftempZRe[i] = acc_sum<N_RX>(ChIn[i], pRxIn[nDataSymbIdx], avxAgcScale);
            }

            #pragma unroll(N_RX)
            for (size_t i = 0; i < N_RX; i++) {
                _mm_prefetch(pRxIn[nDataSymbIdx][i] + 1, _MM_HINT_T1);
                _mm_prefetch(pRxIn[nDataSymbIdx][i] + 2, _MM_HINT_T1);
                pRxIn[nDataSymbIdx][i]++;
            }

            // 4. x = invA * z
            #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i++) {
                auto tx = dot<T, N_TX>(ftempBRe[i], ftempZRe);
                // tx = tx * avxShift;
                tx = _mm512_mul_ph(tx, avxShift);
                if constexpr (fo_flag == FO_E::enable) {
                    tx = fmulconj(tx, FoOffsetTable[i]);
                }
                avxxTxSymbol[i] = _mm512_cvtph_epi16(tx);
            }
        }
    }
#endif

    template<INTERP_E interp = INTERP_E::disable>
    FORCE_INLINE inline
    static void tx_calc(uint16_t nDataSymbIdx,  uint16_t nDataNextSymbIdx,  FloatSimd avxShift,
        FloatSimd finvARe[N_TX][N_TX],  FloatSimd finvAIm[N_TX][N_TX],
        T ChIn[N_TX][N_RX], T ChImNegRe[N_TX][N_RX],  T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX],
        Is16vec32 avxxTxSymbol[N_TX], int16_t nAgcBitShift,uint32_t & nAgcOverflow, T *FoOffsetTable) {
        T rxIn[N_RX];
        FloatSimd ftempZRe[N_TX], ftempZIm[N_TX];
        #pragma unroll(N_RX)
        for (size_t i = 0; i < N_RX; i++) {
            //load Y
            rxIn[i] = *pRxIn[nDataSymbIdx][i];
            //  _mm_prefetch(pRxIn[nDataNextSymbIdx][i], _MM_HINT_T2);
#if DIG_AGC
#if AGC_OVERFLOW_DET
             agc_shift_overflow_detection(rxIn[i], nAgcBitShift, nAgcOverflow);
#else
             agc_shift(rxIn[i], nAgcBitShift);
#endif
#endif
        }

        // 3. Z = H' * y
        #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i++) {
            // calculate the real part of z
            ftempZRe[i] = acc_sum<N_RX>(ChIn[i], rxIn);
            // calculate the imag part of z
            ftempZIm[i] = acc_sum<N_RX>(ChImNegRe[i], rxIn);
        }
        #pragma unroll(N_RX)
        for (size_t i = 0; i < N_RX; i++) {
            _mm_prefetch(pRxIn[nDataSymbIdx][i] + 1, _MM_HINT_T1);
            _mm_prefetch(pRxIn[nDataSymbIdx][i] + 2, _MM_HINT_T1);
            pRxIn[nDataSymbIdx][i]++;
        }

        if constexpr(interp == INTERP_E::disable) {
            for (size_t i = 0; i < N_TX; i++) {
                avxxTxSymbol[i] = txCalc<N_TX>(i, avxShift, finvAIm, finvARe, ftempZRe, ftempZIm);
                if constexpr (fo_flag == FO_E::enable) {
                    avxxTxSymbol[i] = fmulconj(avxxTxSymbol[i], FoOffsetTable[i]);
                }
            }
        }
        else {
            #pragma unroll(N_TX)
            for (size_t i = 0; i < N_TX; i++) {
                avxxTxSymbol[i] = txCalc<N_TX>(i, avxShift, finvAIm, finvARe, ftempZRe, ftempZIm);
                if constexpr (fo_flag == FO_E::enable) {
                    avxxTxSymbol[i] = fmulconj(avxxTxSymbol[i], FoOffsetTable[i]);
                }
            }
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

    static void mimo_mmse_llr_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response) {
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX], *pChIn[1][N_TX][N_RX];
        T ChIn[N_TX][N_RX];
        FloatSimd ftempGain[N_TX];
        FloatSimd ftempPostSINR[N_TX];
        Is16vec32 avxxTxSymbol[N_TX];
        F32vec16  fsumPostSINR[N_TX] = {F32vec16()};

        const int16_t nSubCarrier = request->nSubCarrier;
        const int16_t nTime = nSubCarrier / 16;
        const int16_t nTime0 = (nSubCarrier + 15) / 16;
        const int16_t nRestLen = nSubCarrier - nTime * 16;

#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)

        FloatSimd ftempARe[N_TX][N_TX];
        FloatSimd ftempBRe[N_TX][N_TX];
        FloatSimd finvARe[N_TX][N_TX];
#endif
#if  defined (_BBLIB_AVX512_)
        FloatSimd ftempAIm[N_TX][N_TX];
        FloatSimd ftempBIm[N_TX][N_TX];
        FloatSimd finvAIm[N_TX][N_TX];
        T ChImNegRe[N_TX][N_RX];
#endif
        T FoOffsetTable[BBLIB_N_SYMB_PER_SF][N_TX];

        float llr_postsnr_fxp_dynamic;
        FloatSimd avxShift, avxfSigma2, avxGainShift;
        int16_t llr_range_high, llr_range_low;
        init_factor(request, avxShift, avxfSigma2, llr_postsnr_fxp_dynamic, avxGainShift, llr_range_high, llr_range_low);

        uint16_t nDmrsPortIdx = 0, dataDmrsMux = 0;
        data_dmrs_mux(request, dataDmrsMux, nDmrsPortIdx);
        int16_t nStartSymbIndex = 0;
        const int16_t nChSymb = request->nChSymb;
        int16_t nAgcShiftBit = request->nAgcGain;
#ifdef _BBLIB_SPR_
        float16 fAgcScale = (float16)(1.0 / (request->nAgcGain));
        F16vec32 avxAgcScale = _mm512_set1_ph(fAgcScale);
#endif
        response->nAgcOverflow = 0;

#ifdef SUBMODULE_TICK
        uint64_t nSubModuleTick[PUSCH_SP_SUBMODULE_MAX][3];
#endif
        LOG_TICK_INIT(PUSCH_SP_SUBMODULE_MAX);
        if constexpr (fo_flag == FO_E::enable) {
            init_table_fo(nChSymb, request, FoOffsetTable);
        }
        // loop channel symbol
        #pragma loop_count min(1), max(14)
        for (size_t iChSymb = 0; iChSymb < nChSymb; iChSymb++) {

            init_input_addr<1>(iChSymb, nStartSymbIndex, request, pChIn, pRxIn);
            for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
                int32_t nSc = (nSCIdx + 16 > nSubCarrier) * nRestLen + (nSCIdx + 16 <= nSubCarrier) * 16;

                LOG_TICK_START(PUSCH_SP_H_INTERP);

                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        ChIn[j][i] = *pChIn[0][j][i];
                        if constexpr (fp16Int16 == FP16_E::INT16) {
                            ChImNegRe[j][i] = imagNegReal(ChIn[j][i]);
                        }
                    }
                }
                LOG_TICK_END(PUSCH_SP_H_INTERP);

                if constexpr (fp16Int16 == FP16_E::INT16) {
                    LOG_TICK_START(PUSCH_SP_HxH);
                    //1. A = H' * H + Sigma2
                    HxH ( ftempARe, ftempBRe, ChIn, ftempAIm, ftempBIm, ChImNegRe, avxfSigma2);
                    LOG_TICK_END(PUSCH_SP_HxH);

                    LOG_TICK_START(PUSCH_SP_INV);
                    // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                    matrix_inverse<N_TX>(ftempBRe, ftempBIm, finvARe, finvAIm);
                    LOG_TICK_END(PUSCH_SP_INV);

                    LOG_TICK_START(PUSCH_SP_GAIN);
                    // 3. gain calc
                    if (iChSymb == 0) {
                        gainCalc<N_TX, 1>(ftempGain, ftempPostSINR, fsumPostSINR, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe, llr_postsnr_fxp_dynamic);
                    } else {
                        gainCalc<N_TX, 0>(ftempGain, ftempPostSINR, fsumPostSINR, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe, llr_postsnr_fxp_dynamic);
                    }
                    LOG_TICK_END(PUSCH_SP_GAIN);
                }
#if defined (_BBLIB_SPR_)
                else
                {
                    LOG_TICK_START(PUSCH_SP_HxH);
                    //1. A = H' * H + Sigma2
                    HxH<T, N_TX, N_RX> ( ftempARe, ftempBRe, ChIn, avxfSigma2);
                    // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                    LOG_TICK_END(PUSCH_SP_HxH);

                    LOG_TICK_START(PUSCH_SP_INV);
                    matrix_inverse<T, N_TX>(ftempBRe);
                    LOG_TICK_END(PUSCH_SP_INV);

                    LOG_TICK_START(PUSCH_SP_GAIN);
                    // 3. gain calc
                    if (iChSymb == 0) {
                        gainCalc<N_TX, 1>(ftempGain, ftempPostSINR, fsumPostSINR, ftempBRe, ftempARe, llr_postsnr_fxp_dynamic);
                    } else {
                        gainCalc<N_TX, 0>(ftempGain, ftempPostSINR, fsumPostSINR, ftempBRe, ftempARe, llr_postsnr_fxp_dynamic);
                    }
                    LOG_TICK_END(PUSCH_SP_GAIN);

                }
#endif
                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        pChIn[0][j][i]++;
                        // _mm_prefetch(pChIn[0][j][i], _MM_HINT_T1);
                    }
                }

                xran_decomp(request,  pRxIn, nSCIdx, nSubCarrier, iChSymb);

                for (size_t nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++) {
                    auto nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                    auto nDataNextSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex + 1);
                    auto dataDmrsFlag = dataDmrsMux & ((uint16_t) 1) << nDataSymbIdx;

                    LOG_TICK_START(PUSCH_SP_EQU);
                    // 4. x = invA * z
                    if constexpr (fp16Int16 == FP16_E::INT16) {
                        tx_calc<INTERP_E::disable>(nDataSymbIdx, nDataNextSymbIdx, avxShift, finvARe, finvAIm, ChIn, ChImNegRe, pRxIn, avxxTxSymbol, nAgcShiftBit, response->nAgcOverflow, FoOffsetTable[nDataSymbIdx]);
                    }
#if defined (_BBLIB_SPR_)
                    else
                    {
                        tx_calc<INTERP_E::disable>(nDataSymbIdx, nDataNextSymbIdx, avxShift, ftempBRe, ChIn, (CI16vec16 * (*)[N_RX])pRxIn, avxxTxSymbol, avxAgcScale, FoOffsetTable[nDataSymbIdx]);
                    }
#endif
                    LOG_TICK_END(PUSCH_SP_EQU);

                    LOG_TICK_START(PUSCH_SP_DEMOD);
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
                    LOG_TICK_END(PUSCH_SP_DEMOD);
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
        LOG_TICK_REPORT(response, PUSCH_SP_SUBMODULE_MAX)
    }

    static void init_table_interp(bblib_pusch_symbol_processing_request *request, procDataType wSym[BBLIB_N_SYMB_PER_SF][2], int16_t &flag_symH_upd) {
        int16_t nDmrsChSymb = request->nDmrsChSymb;
        int16_t nDataSymb = request->nSymb;
        int16_t nMappingType = request->nMappingType;
        int16_t nGranularity = request->nGranularity;
        uint8_t nNrOfDMRSSymbols = request->nNrOfDMRSSymbols;
        for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
            auto nSymNum = request->pSymbIndex[iSymb];
            int16_t wSym0 = 0, wSym1 = 1;
            if(nMappingType == 0) {//type A
                uint16_t lut_idx = request->pDmrsSymbolIdx[nNrOfDMRSSymbols] - request->pDmrsSymbolIdx[0] - 4;// map to index of LUT
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
                uint16_t lut_idx = request->pDmrsSymbolIdx[nNrOfDMRSSymbols]/2 -2;// map to index of LUT
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
            wSym[nSymNum][0] = static_cast<procDataType>((float)wSym0 / wcoeff);
            wSym[nSymNum][1] = static_cast<procDataType>((float)wSym1 / wcoeff);
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
    }

    static void mimo_mmse_llr_avx512_interp(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response) {
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX], *pChIn[2][N_TX][N_RX];
        T ChIn[N_TX][N_RX];
        T ChInTemp[2][N_TX][N_RX];
        FloatSimd ftempGain[N_TX];
        FloatSimd ftempPostSINR[N_TX];
        Is16vec32 avxxTxSymbol[N_TX];
        F32vec16  fsumPostSINR[N_TX] = {F32vec16()};

        const int16_t nSubCarrier = request->nSubCarrier;
        const int16_t nTime = nSubCarrier / 16;
        const int16_t nTime0 = (nSubCarrier + 15) / 16;
        const int16_t nRestLen = nSubCarrier - nTime * 16;
        uint8_t nNrOfDMRSSymbols = request->nNrOfDMRSSymbols;
#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
        FloatSimd ftempARe[N_TX][N_TX];
        FloatSimd ftempBRe[N_TX][N_TX];
        FloatSimd finvARe[N_TX][N_TX];
#endif
#if  defined (_BBLIB_AVX512_)
        FloatSimd ftempAIm[N_TX][N_TX];
        FloatSimd ftempBIm[N_TX][N_TX];
        FloatSimd finvAIm[N_TX][N_TX];
        T ChImNegRe[N_TX][N_RX];
#endif
        T FoOffsetTable[BBLIB_N_SYMB_PER_SF][N_TX];

        float llr_postsnr_fxp_dynamic;
        FloatSimd avxShift, avxfSigma2, avxGainShift;
        int16_t llr_range_high, llr_range_low;
        init_factor(request, avxShift, avxfSigma2, llr_postsnr_fxp_dynamic, avxGainShift, llr_range_high, llr_range_low);
        uint16_t nDmrsPortIdx = 0, dataDmrsMux = 0;
        data_dmrs_mux(request, dataDmrsMux, nDmrsPortIdx);
        int16_t nAgcShiftBit = request->nAgcGain;
#ifdef _BBLIB_SPR_
        float16 fAgcScale = (float16)(1.0 / (request->nAgcGain));
        F16vec32 avxAgcScale = _mm512_set1_ph(fAgcScale);
#endif
        const int16_t nChSymb = request->nChSymb;
        response->nAgcOverflow = 0;
        int16_t *pIntrpweights[BBLIB_MAX_TX_LAYER_NUM];

        for (size_t iLayer = 0; iLayer < N_TX; iLayer++) {
            pIntrpweights[iLayer] = request->pIntrpweights[iLayer];
        }

#ifdef SUBMODULE_TICK
        uint64_t nSubModuleTick[PUSCH_SP_SUBMODULE_MAX][3];
#endif
        LOG_TICK_INIT(PUSCH_SP_SUBMODULE_MAX);

        procDataType wSym[BBLIB_N_SYMB_PER_SF][2];
        int16_t flag_symH_upd = 0;
        init_table_interp(request, wSym, flag_symH_upd);
        if constexpr (fo_flag == FO_E::enable) {
            init_table_fo(nChSymb, request, FoOffsetTable);
        }
        init_input_addr<2>(0, 0, request, pChIn, pRxIn);
        auto iDmrsidx1 = request->pDmrsSymbolIdx[nNrOfDMRSSymbols];
        // loop channel symbol
        for (size_t nSCIdx = 0; nSCIdx < nSubCarrier; nSCIdx = nSCIdx + 16) {
            int32_t nSc = (nSCIdx + 16 > nSubCarrier) * nRestLen + (nSCIdx + 16 <= nSubCarrier) * 16;
            LOG_TICK_START(PUSCH_SP_LoadH_P1);
            if constexpr (N_TX < 4)
            {
                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        ChInTemp[0][j][i] = *pChIn[0][j][i]++;
                        ChInTemp[1][j][i] = *pChIn[1][j][i]++;
                        if constexpr (fo_flag == FO_E::enable) {
                            ChInTemp[1][j][i] = fmulconj(ChInTemp[1][j][i], FoOffsetTable[iDmrsidx1][j]);
                        }
                        _mm_prefetch(pChIn[0][j][i], _MM_HINT_T2);
                        _mm_prefetch(pChIn[1][j][i], _MM_HINT_T2);
                    }
                }
            }
            else
            {
                #pragma unroll(N_TX)
                for (size_t j = 0; j < N_TX; j ++) {
                    #pragma unroll(N_RX)
                    for (size_t i = 0; i < N_RX; i++) {
                        ChInTemp[0][j][i] = *pChIn[0][j][i];
                        ChInTemp[1][j][i] = *pChIn[1][j][i];
                        if constexpr (fo_flag == FO_E::enable) {
                            ChInTemp[1][j][i] = fmulconj(ChInTemp[1][j][i], FoOffsetTable[iDmrsidx1][j]);
                        }
                    }
                }
            }
            LOG_TICK_END(PUSCH_SP_LoadH_P1);

            xran_decomp(request, pRxIn, nSCIdx, nSubCarrier, 0);

            int16_t nDataSymb = request->nSymb;
            procDataType w_dmrsSym0, w_dmrsSym1;
            int16_t w_temp_dmrsSym0, w_temp_dmrsSym1;
            for (size_t iSymb = 0; iSymb < nDataSymb; iSymb++) {
                auto nDataSymbIdx = request->pSymbIndex[iSymb];
                auto dataDmrsFlag = dataDmrsMux & ((uint16_t) 1)  << nDataSymbIdx;
                // If it is needed to update interpolated CE, otherwise previous sym parameters reused
                if (flag_symH_upd & (1 << (BBLIB_N_SYMB_PER_SF -  1 - nDataSymbIdx))) {

                    LOG_TICK_START(PUSCH_SP_H_INTERP);
                    #pragma unroll(N_TX)
                    for (size_t j = 0; j < N_TX; j ++) {
                        #pragma unroll(N_RX)
                        for (size_t i = 0; i < N_RX; i++) {

                            if (request->nGranularity < 6) {
                                w_dmrsSym0 = wSym[nDataSymbIdx][0];
                                w_dmrsSym1 = wSym[nDataSymbIdx][1];
                            }
                            else {
                                w_temp_dmrsSym0 = *(pIntrpweights[j] + nDataSymbIdx*4);
                                w_temp_dmrsSym1 = *(pIntrpweights[j] + nDataSymbIdx*4+1);
                                w_dmrsSym0 = static_cast<procDataType>(w_temp_dmrsSym0);
                                w_dmrsSym1 = static_cast<procDataType>(w_temp_dmrsSym1);
                            }

                            if constexpr (fp16Int16 == FP16_E::INT16) {
                            auto temp0 = mulhrs(ChInTemp[0][j][i], T(w_dmrsSym0));
                            auto temp1 = mulhrs(ChInTemp[1][j][i], T(w_dmrsSym1));
                                ChIn[j][i] = (temp0 + temp1) << 1;
                            }
#if defined (_BBLIB_SPR_)
                            else {
                                ChIn[j][i] = mulhrs(ChInTemp[0][j][i], T(w_dmrsSym0));
                                ChIn[j][i] = _mm512_fmadd_ph(ChInTemp[1][j][i], T(w_dmrsSym1), ChIn[j][i]);
                            }
#endif

                            if constexpr (fp16Int16 == FP16_E::INT16) {
                                ChImNegRe[j][i] = imagNegReal(ChIn[j][i]);
                            }
                        }
                    }
                    LOG_TICK_END(PUSCH_SP_H_INTERP);

                    if constexpr (fp16Int16 == FP16_E::INT16) {
                        LOG_TICK_START(PUSCH_SP_HxH);
                        //1. A = H' * H + Sigma2
                        HxH ( ftempARe, ftempBRe, ChIn, ftempAIm, ftempBIm, ChImNegRe, avxfSigma2);
                        LOG_TICK_END(PUSCH_SP_HxH);

                        LOG_TICK_START(PUSCH_SP_INV);
                        // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                        matrix_inverse<N_TX>(ftempBRe, ftempBIm, finvARe, finvAIm);
                        LOG_TICK_END(PUSCH_SP_INV);

                        LOG_TICK_START(PUSCH_SP_GAIN);
                        // 3. gain calc
                        if (iSymb == 0) {
                            gainCalc<N_TX, 1>(ftempGain, ftempPostSINR, fsumPostSINR, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe, llr_postsnr_fxp_dynamic);
                        } else {
                            gainCalc<N_TX, 0>(ftempGain, ftempPostSINR, fsumPostSINR, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe, llr_postsnr_fxp_dynamic);
                        }
                        LOG_TICK_END(PUSCH_SP_GAIN);
                    }
#if defined (_BBLIB_SPR_)
                    else
                    {
                        LOG_TICK_START(PUSCH_SP_HxH);
                        //1. A = H' * H + Sigma2
                        HxH<T, N_TX, N_RX> ( ftempARe, ftempBRe, ChIn, avxfSigma2);
                        LOG_TICK_END(PUSCH_SP_HxH);

                        LOG_TICK_START(PUSCH_SP_INV);
                        // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                        matrix_inverse<T, N_TX>(ftempBRe);
                        LOG_TICK_END(PUSCH_SP_INV);

                        LOG_TICK_START(PUSCH_SP_GAIN);
                        // 3. gain calc
                        if (iSymb == 0) {
                            gainCalc<N_TX, 1>(ftempGain, ftempPostSINR, fsumPostSINR, ftempBRe, ftempARe, llr_postsnr_fxp_dynamic);
                        } else {
                            gainCalc<N_TX, 0>(ftempGain, ftempPostSINR, fsumPostSINR, ftempBRe, ftempARe, llr_postsnr_fxp_dynamic);
                        }
                        LOG_TICK_END(PUSCH_SP_GAIN);
                    }
#endif

                    LOG_TICK_START(PUSCH_SP_LoadH_P2);
                    if constexpr (N_TX >= 4)
                    {
                        if (iSymb == 0)
                        {
                            #pragma unroll(N_TX)
                            for (size_t j = 0; j < N_TX; j ++) {
                                #pragma unroll(N_RX)
                                for (size_t i = 0; i < N_RX; i++) {
                                    pChIn[0][j][i]++;
                                    pChIn[1][j][i]++;
                                    //_mm_prefetch(pChIn[0][j][i], _MM_HINT_T2);
                                    //_mm_prefetch(pChIn[1][j][i], _MM_HINT_T2);
                                }
                            }
                        }
                    }
                    LOG_TICK_END(PUSCH_SP_LoadH_P2);
                }

                LOG_TICK_START(PUSCH_SP_EQU);
                auto nDataNextSymbIdx = request->pSymbIndex[iSymb + 1];
                // 4. x = invA * z
                if constexpr (fp16Int16 == FP16_E::INT16) {
                    tx_calc<INTERP_E::enable>(nDataSymbIdx, nDataNextSymbIdx, avxShift, finvARe, finvAIm, ChIn, ChImNegRe, pRxIn, avxxTxSymbol, nAgcShiftBit, response->nAgcOverflow, FoOffsetTable[nDataSymbIdx]);
                }
#if defined (_BBLIB_SPR_)
                else
                {
                    tx_calc<INTERP_E::enable>(nDataSymbIdx, nDataNextSymbIdx, avxShift, ftempBRe, ChIn, (CI16vec16 * (*)[N_RX])pRxIn, avxxTxSymbol, avxAgcScale, FoOffsetTable[nDataSymbIdx]);
                }
#endif
                LOG_TICK_END(PUSCH_SP_EQU);

                LOG_TICK_START(PUSCH_SP_DEMOD);
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
                LOG_TICK_END(PUSCH_SP_DEMOD);
            }//for iSymb
        }//end the sc loops
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
        LOG_TICK_REPORT(response, PUSCH_SP_SUBMODULE_MAX)
    }
   #if 0
    static void mimo_idft_llr_avx512_fp16int16(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
    {
        if (N_RX == 0) {
            exit(-1);
        }// Only add for klocwork check
        int32_t nSCIdx, nSymbIdx;
        T *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX];        // Pointer to Y
        T* pChIn[1][N_TX][N_RX];                       // Pointer to H
        T avxhChState[N_TX][N_RX];                  // H
        Is16vec32 avxxTempTxSymbol[N_TX];           // X per loop
        Is16vec32 avxxTxSymbol[MAX_TP_SC_TIME];     // X all
        Is16vec32 avxxTxSymbolIdft[MAX_TP_SC_TIME]; // X after IDFT all
        FloatSimd ftempInvARe[MAX_TP_SC_TIME];      // Float invA all
        F32vec16  fsumPostSINR[N_TX] = {F32vec16()};
#if defined (_BBLIB_SPR_) || defined (_BBLIB_AVX512_)
        FloatSimd ftempARe[N_TX][N_TX];             // Float A = H'*H
        FloatSimd ftempBRe[N_TX][N_TX];             // Float B = A + sigma2
        FloatSimd finvARe[N_TX][N_TX];              // invA = reciprocal(B) per loop
#endif
#if  defined (_BBLIB_AVX512_)
        FloatSimd ftempAIm[N_TX][N_TX];
        FloatSimd ftempBIm[N_TX][N_TX];
        FloatSimd finvAIm[N_TX][N_TX];
        T avxhChSwitch[N_TX][N_RX];
        IppsDFTSpec_C_32fc *pDFTSpec=0;
        Ipp8u  *pDFTInitBuf = NULL, *pDFTWorkBuf = NULL;
        int16_t *pIdftInI16 = NULL;
        float *pIdftOutF32 = NULL;
        int sizeDFTSpec,sizeDFTInitBuf,sizeDFTWorkBuf;
        __align(64) Ipp16sc *pSrcI16;
        __align(64) Ipp16s *pTempDstI16Re;
        __align(64) Ipp16s *pTempDstI16Im;
        __align(64) Ipp32f *pTempDstFP32Re;
        __align(64) Ipp32f *pTempDstFP32Im;
        __align(64) Ipp32fc *pTempSrcFP32;
        __align(64) Ipp32fc *pDstFP32;
        FloatSimd fp32xTxSymbolIdft[2*MAX_TP_SC_TIME]; // FP32 X after IDFT all
#endif
#if defined (_BBLIB_SPR_)
        CI16vec16 rxIn[N_RX];                        // Y
        FloatSimd ftempZRe[N_TX];                    // Float Z (Z = H' * Y)
        FloatSimd fp16xTxSymbol[MAX_TP_SC_TIME];     // FP16 X all
        FloatSimd fp16xTxSymbolIdft[MAX_TP_SC_TIME]; // FP16 X after IDFT all
        // Used for ipp Float IDFT function
        Float *pIdftIn = NULL, *pIdftOut = NULL;
        __align(64) Ipp16fc *pSrcFP16;
        __align(64) Ipp16fc *pDstFP16;
#endif
        int16_t nSubCarrier = request->nSubCarrier;
        FloatSimd ftempGain[N_TX];
        FloatSimd ftempPostSINR[N_TX];
        FloatSimd ftempGainAccu[1];
        float llr_postsnr_fxp_dynamic;
        int16_t llr_range_high, llr_range_low;
        FloatSimd avxShift, avxfSigma2, avxGainShift;
        init_factor(request, avxShift, avxfSigma2, llr_postsnr_fxp_dynamic, avxGainShift, llr_range_high, llr_range_low);

        T FoOffsetTable[BBLIB_N_SYMB_PER_SF][N_TX];

        int16_t nStartSC = request->nStartSC;
        int32_t i;
        uint16_t nDataSymbIdx = 0, nDataNextSymbIdx = 0;
        int32_t iChSymb = 0;
        int16_t nChSymb = request->nChSymb;
        int16_t nStartSymbIndex = 0;
        int8_t **ppSoft = NULL;
        int32_t nSc = 0;
#ifdef _BBLIB_SPR_
        float16 fAgcScale = (float16)(1.0 / (request->nAgcGain));
        F16vec32 avxAgcScale = _mm512_set1_ph(fAgcScale);
#endif

        enum bblib_modulation_order modOrder;

        //pre-calculate the remainder length and masks
        int16_t nTime = (nSubCarrier + 15) / 16;
        if(unlikely(nTime <= 0 || nTime > MAX_TP_SC_TIME))
        {
            printf("\nInvalid nSubCarrier %d!!\n", nSubCarrier);
            exit(-1);
        }

#if defined (_BBLIB_SPR_)
        if(nSubCarrier == _DFT_IDFT_1728 || nSubCarrier == _DFT_IDFT_2304 || nSubCarrier == _DFT_IDFT_2592 ||
        nSubCarrier == _DFT_IDFT_2880 || nSubCarrier == _DFT_IDFT_3072)
        {
            printf("\nInvalid nSubCarrier %d for FP16 IDFT!!\n", nSubCarrier);
            exit(-1);
        }
#endif


        int16_t nRestLen = nSubCarrier - (nTime - 1) * 16;
        __mmask16 gainStorMask = 0xffffU >> (16 - nRestLen);

        struct bblib_dft_request idft_request;
        struct bblib_dft_response idft_response;

        if constexpr (fo_flag == FO_E::enable) {
            init_table_fo(nChSymb, request, FoOffsetTable);
        }

        // Initialize IDFT Buffers
        if constexpr (fp16Int16 == FP16_E::INT16)
        {
            ippsDFTGetSize_C_32fc(nSubCarrier, IPP_FFT_NODIV_BY_ANY, 
            ippAlgHintAccurate, &sizeDFTSpec, &sizeDFTInitBuf, &sizeDFTWorkBuf);
            pDFTSpec    = (IppsDFTSpec_C_32fc*)ippsMalloc_8u(sizeDFTSpec);
            pDFTInitBuf = ippsMalloc_8u(sizeDFTInitBuf);
            pDFTWorkBuf = ippsMalloc_8u(sizeDFTWorkBuf);
            ippsDFTInit_C_32fc(nSubCarrier, IPP_FFT_NODIV_BY_ANY, 
            ippAlgHintAccurate, pDFTSpec, pDFTInitBuf);
            if (pDFTInitBuf) ippsFree(pDFTInitBuf);
            pTempDstI16Re = ippsMalloc_16s(nSubCarrier);
            pTempDstI16Im = ippsMalloc_16s(nSubCarrier);
            pTempDstFP32Re = ippsMalloc_32f(nSubCarrier);
            pTempDstFP32Im = ippsMalloc_32f(nSubCarrier);
            pTempSrcFP32 = ippsMalloc_32fc(nSubCarrier);
        }

        // loop channel symbol
        for (iChSymb = 0; iChSymb < nChSymb; iChSymb++)
        {
            ftempGainAccu[0] = FloatSimd(0.0);
            Float postSINR = 0;
            Float gain = 0;
            // one channel symbol only when disable interpolation
            init_input_addr<1>(iChSymb, nStartSymbIndex, request, pChIn, pRxIn);
            for (nSCIdx = 0; nSCIdx < nTime; nSCIdx++)
            {
                //load H (N_RX x 1)
                for (i = 0; i < N_RX; i++)
                {
                    avxhChState[0][i] = *pChIn[0][0][i]++;
                }

                if constexpr (fp16Int16 == FP16_E::INT16)
                {
                    // 1. A = H' * H + Sigma2
                    HxH (ftempARe, ftempBRe, avxhChState, ftempAIm, ftempBIm, avxhChSwitch, avxfSigma2);
                    // 2. invA = inv(H' * H + Sigma2*I), 1x1 matrix inversion
                    ftempInvARe[nSCIdx] = (FloatSimd)_mm512_rcp14_ps((__m512)ftempBRe[0][0]);
                    // 3. calculate the per sc gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
                    ftempGain[0] = _mm512_mul_ps((__m512)ftempInvARe[nSCIdx], (__m512)ftempARe[0][0]);
                    ftempGain[0] = select_low_float(m_gain_threshold, ftempGain[0]);
                    if (nSCIdx == nTime - 1){
                        ftempGainAccu[0] = _mm512_mask_add_ps(ftempGainAccu[0], gainStorMask, ftempGainAccu[0], ftempGain[0]);
                    }
                    else
                        ftempGainAccu[0] = _mm512_add_ps(ftempGainAccu[0], ftempGain[0]);
                }
#if defined (_BBLIB_SPR_)
                else
                {
                    // 1. A = H' * H + Sigma2
                    HxH<T, N_TX, N_RX> (ftempARe, ftempBRe, avxhChState, avxfSigma2);
                    // 2. invA = inv(H' * H + Sigma2*I), 16x16 matrix inversion
                    matrix_inverse<T, N_TX>(ftempBRe);
                    ftempInvARe[nSCIdx] = ftempBRe[0][0];
                    // 3. calculate the per sc gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
                    CF16vec16 gainTmp = CF16vec16((float16)0.0);
                    gainTmp = gainCalc<CF16vec16, 1>(0, ftempBRe, ftempARe);
                    if (nSCIdx == nTime - 1){
                        ftempGainAccu[0] = _mm512_mask_add_ph(ftempGainAccu[0], gainStorMask, ftempGainAccu[0], gainTmp);
                    }
                    else
                        ftempGainAccu[0] = _mm512_add_ph(ftempGainAccu[0], gainTmp);
                }
#endif
            }

            if constexpr (fp16Int16 == FP16_E::INT16)
            {
                // gain = (accumulate per sc gain) / nSubCarrier
                // which is same for all sc in one SC-FDMA symbol
                gain = _mm512_reduce_add_ps(ftempGainAccu[0]);
                gain /= nSubCarrier;

                //calc postSINR=1/(1-gain)
                postSINR = 1.0 / (1.0 - gain);

                if (iChSymb == 0) {
                    response->fPostSINR[0] = gain * postSINR;
                }

                gain *= (1<<15); //gain is 16S15;
                ftempGain[0] = _mm512_set1_ps(gain);

                postSINR *= llr_postsnr_fxp_dynamic;
                ftempPostSINR[0] = _mm512_set1_ps(postSINR);
            }
#if defined (_BBLIB_SPR_)
            else
            {
                gain =  _mm512_reduce_add_ph(ftempGainAccu[0]);
                gain /= nSubCarrier;

                if (iChSymb == 0) {
                    response->fPostSINR[0] = gain / (1.0 - gain);
                }

                CF16vec16 gainTemp = CF16vec16((float16)gain);
                gainTemp = duplicateReal(gainTemp);
                auto llr_beta_fxp_fp16 = CF16vec16(static_cast<float16>(1 << 15));
                ftempGain[0] = _mm512_mul_ph(gainTemp, llr_beta_fxp_fp16);

                CF16vec16 postSinrTemp = _mm512_rcp_ph(CF16vec16((float16)1.0) - gainTemp);
                postSinrTemp = _mm512_mul_ph (postSinrTemp, CF16vec16((float16)llr_postsnr_fxp_dynamic));
                ftempPostSINR[0] = min(postSinrTemp, CF16vec16((float16)32759.0));
            }
#endif
            for (nSymbIdx = 0; nSymbIdx < request->nSymbPerDmrs[iChSymb]; nSymbIdx++)
            {
                nDataSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex);
                nDataNextSymbIdx = *(request->pSymbIndex + nSymbIdx + nStartSymbIndex + 1);
                //due to Klocwork report: memory bound protection
                if(unlikely(nDataSymbIdx<0||nDataSymbIdx>=BBLIB_N_SYMB_PER_SF))
                    nDataSymbIdx = 0;

                // for (i = 0; i < N_RX; i++)
                // pChIn[0][0][i] =
                //   ptr_cast<T *>((int16_t *)request->pChState[i][0] + iChSymb * request->nTotalAlignedSubCarrier * 2);
                for (i = 0; i < N_RX; i++) {
                    pChIn[0][0][i] =
                        ptr_cast<T *>((int16_t *)request->pChState[i][0] + iChSymb * request->nTotalAlignedSubCarrier * 2 + nStartSC * 2);
                }

                //do equalization, no need to do remainder processing, because all of the data are stored in temporary buffer
                for (nSCIdx = 0; nSCIdx < nTime; nSCIdx++)
                {

                    xran_decomp(request, pRxIn, nSCIdx, nSubCarrier, iChSymb);

                    for (i = 0; i < N_RX; i++)
                    {   // load H (N_RX X 1)
                        avxhChState[0][i] = *pChIn[0][0][i]++;


                        if constexpr (fp16Int16 == FP16_E::INT16)
                        {
                            avxhChSwitch[0][i] = imagNegReal(avxhChState[0][i]);
                        }
                        // load Y poniter (N_RX X 1) here and load data in tx_calc
                        // pRxIn[nDataSymbIdx][i] = ptr_cast<T *>((int16_t *)request->pRxSignal[i][nDataSymbIdx] + nStartSC * 2 + nSCIdx * 32);
                    }

                    // 4. x = invA * z
                    if constexpr (fp16Int16 == FP16_E::INT16)
                    {
                        finvARe[0][0] = ftempInvARe[nSCIdx];
                        finvAIm[0][0] = FloatSimd(0.0);
                        tx_calc<INTERP_E::disable>(nDataSymbIdx, nDataNextSymbIdx, avxShift, finvARe, finvAIm, avxhChState,
                                                   avxhChSwitch, pRxIn, avxxTempTxSymbol, request->nAgcGain, response->nAgcOverflow, FoOffsetTable[nDataSymbIdx]);
                        avxxTxSymbol[nSCIdx] = avxxTempTxSymbol[0];
                    }
#if defined (_BBLIB_SPR_)
                    else
                    {
                         ftempBRe[0][0] = ftempInvARe[nSCIdx];
                        #pragma unroll(N_RX)
                        for (size_t i = 0; i < N_RX; i++) {
                            //load Y
                            rxIn[i] = *(CI16vec16*)pRxIn[nDataSymbIdx][i]++;
                            _mm_prefetch(pRxIn[nDataSymbIdx][i], _MM_HINT_T1);
                            _mm_prefetch(pRxIn[nDataSymbIdx][i] + 1, _MM_HINT_T1);
                        }

                        #pragma unroll(N_TX)
                        for (size_t i = 0; i < N_TX; i++) {
                            // calculate the real part and imag part of z
                            ftempZRe[i] = acc_sum<N_RX>(avxhChState[i], rxIn, avxAgcScale);
                        }

                        // 4. x = invA * z
                        #pragma unroll(N_TX)
                        for (size_t i = 0; i < N_TX; i++) {
                            fp16xTxSymbol[nSCIdx] = dot<T, N_TX>(ftempBRe[i], ftempZRe);
                        }
                    }
#endif
                }

                if constexpr (fp16Int16 == FP16_E::INT16)
                {
                    if(nSCIdx==nTime)
                    {
                        if(nSubCarrier <= 1200)
                        {
                            //do IDFT, it is symbol level
                            idft_request.dft_idft_flag = BBLIB_IDFT_TYPE;
                            idft_request.dft_idft_points = nSubCarrier;
                            idft_request.num_input_buffers = 1;
                            idft_request.data_in = (void *)(&avxxTxSymbol[0]);
                            idft_response.data_out = (void *)(&avxxTxSymbolIdft[0]);
                            // perform dft_idft in fixed point
                            bblib_idft_burst_fxp(&idft_request, &idft_response);

                            // To retore idft scaling, data should multiply 2^(scale_out) / N * sqrt(dft_idft_points)
                            float idftScale = (float)(1 << idft_response.scale_out) / (double)(sqrt(nSubCarrier));
                            // mulhrs results 16s(Q0+Q1-Q2), default Q0:13, Q2:15
                            int16_t Q2 = 15;
                            int16_t Q1 = floor(log2(float(1 << Q2) / (double)idftScale));
                            int16_t scaleFactor = idftScale * (1 << Q1);
                            __m512i scale = _mm512_set1_epi16(scaleFactor);
                            for (nSCIdx = 0; nSCIdx < nTime; nSCIdx ++)
                            {
                                avxxTxSymbolIdft[nSCIdx] = _mm512_mulhrs_epi16(avxxTxSymbolIdft[nSCIdx], scale);
                                avxxTxSymbolIdft[nSCIdx] = _mm512_slli_epi16(avxxTxSymbolIdft[nSCIdx], (Q2-Q1));
                            }
                        }
                        else // nSubCarrier > 1200 - IPP based implementation
                        {
                            pIdftInI16 = (int16_t*)(&avxxTxSymbol[0]);
                            pIdftOutF32 = (float*)(&fp32xTxSymbolIdft[0]);
                            pSrcI16 = (Ipp16sc *)pIdftInI16;
                            pDstFP32 = (Ipp32fc *)pIdftOutF32;

                            ippsCplxToReal_16sc(pSrcI16, pTempDstI16Re, pTempDstI16Im, nSubCarrier);
                            ippsConvert_16s32f(pTempDstI16Re, pTempDstFP32Re, nSubCarrier);
                            ippsConvert_16s32f(pTempDstI16Im, pTempDstFP32Im, nSubCarrier);
                            ippsRealToCplx_32f(pTempDstFP32Re, pTempDstFP32Im, pTempSrcFP32, nSubCarrier);
                            ippsDFTInv_CToC_32fc(pTempSrcFP32,pDstFP32,pDFTSpec,pDFTWorkBuf);
                            // restore original X, X_after_Idft / N * sqrt(dft_idft_points)
                            F32vec16 sqrtN = _mm512_rcp14_ps(F32vec16((float)(sqrt(nSubCarrier))));
                            for (nSCIdx = 0; nSCIdx < nTime; nSCIdx++)
                            {
                                fp32xTxSymbolIdft[2*nSCIdx] = _mm512_mul_ps(fp32xTxSymbolIdft[2*nSCIdx], sqrtN);
                                fp32xTxSymbolIdft[2*nSCIdx + 1] = _mm512_mul_ps(fp32xTxSymbolIdft[2*nSCIdx + 1], sqrtN);

                                __m512i temp1_512i = _mm512_cvtps_epi32 (fp32xTxSymbolIdft[2*nSCIdx]);
                                __m256i temp1_256i = _mm512_cvtepi32_epi16 (temp1_512i);
                                __m512i temp2_512i = _mm512_cvtps_epi32 (fp32xTxSymbolIdft[2*nSCIdx + 1]);
                                __m256i temp2_256i = _mm512_cvtepi32_epi16 (temp2_512i);
                                __m512i temp_512i = _mm512_setzero_epi32();
                                temp_512i =  _mm512_inserti32x8 (temp_512i, temp1_256i, 0);
                                temp_512i =  _mm512_inserti32x8 (temp_512i, temp2_256i, 1);
                                avxxTxSymbolIdft[nSCIdx] = temp_512i;
                            }
                        }
                    }
                }
#if defined (_BBLIB_SPR_)
                else
                {
                    if(nSCIdx==nTime)
                    {
                        pIdftIn = (Float*)(&fp16xTxSymbol[0]);
                        pIdftOut = (Float*)(&fp16xTxSymbolIdft[0]);
                        // do Float IDFT
                        pSrcFP16 = (Ipp16fc *)pIdftIn;
                        pDstFP16 = (Ipp16fc *)pIdftOut;
                        ippsDFTInv_Direct_CToC_16fc(pSrcFP16, pDstFP16, nSubCarrier);
                        // restore original X, X_after_Idft / N * sqrt(dft_idft_points)
                        CF16vec16 sqrtN = _mm512_rcp_ph(CF16vec16((float16)(sqrt(nSubCarrier))));
                        for (nSCIdx = 0; nSCIdx < nTime; nSCIdx++)
                        {
                            fp16xTxSymbolIdft[nSCIdx] = _mm512_mul_ph(fp16xTxSymbolIdft[nSCIdx], sqrtN);
                            fp16xTxSymbolIdft[nSCIdx] = _mm512_mul_ph(fp16xTxSymbolIdft[nSCIdx], avxShift);
                            avxxTxSymbolIdft[nSCIdx] = _mm512_cvtph_epi16(fp16xTxSymbolIdft[nSCIdx]);
                        }
                    }
                }
#endif
                // do LLR demapping
                // in current FPGA decoding implementation, LLR=p(b=0|r)/p(b=1|r)
                ppSoft = &(response->pLlr[nDataSymbIdx][0]);
                modOrder = request->eModOrder[0];
                for (nSCIdx = 0; nSCIdx < nTime; nSCIdx++)
                {
                    nSc = (nSCIdx == nTime - 1) ? nRestLen : 16;
                    Is16vec32 *pTx = &avxxTxSymbolIdft[nSCIdx];
                    DEMAPPER<T, DATA_DMRS_MUX_E::disable, DMRSTYPE, NROFCDMS>::demaper_llr(modOrder, ftempPostSINR, ftempGain, pTx,
                                                                                           *ppSoft, llr_range_low, llr_range_high, nSc);
                    *ppSoft += nSc * modOrder;
                }
            }//end the Symbol cycle
            if (iChSymb == 0) {
                if (nTime <= 3) {
                    for (size_t j = 0; j < N_TX; j++)
                    {
                        response->fPostSINR[j] = reduce_add(fsumPostSINR[j]) / nSubCarrier;
                    }
                } else {
                    for (size_t j = 0; j < N_TX; j++)
                    {
                        response->fPostSINR[j] = *(reinterpret_cast<float *>(&fsumPostSINR[j])) / nTime;
                    }
                }
            }
            nStartSymbIndex += request->nSymbPerDmrs[iChSymb];
        }//end the Channel symbol cycle

        // Free IDFT Buffers
        if constexpr (fp16Int16 == FP16_E::INT16)
        {
            ippsFree(pDFTWorkBuf);
            ippsFree(pTempDstI16Re);
            ippsFree(pTempDstI16Im);
            ippsFree(pTempDstFP32Re);
            ippsFree(pTempDstFP32Im);
            ippsFree(pTempSrcFP32);
        }

    }
#endif

    static void mimo_mmse_llr_avx512_common(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response *response) {
        if constexpr (interp_flag == INTERP_E::enable) {
            mimo_mmse_llr_avx512_interp(request,response);}
        else{
            mimo_mmse_llr_avx512(request,response);}
    }
};
/*! \brief PUSCH symbol processing for transform precoding, include MMSE MIMO detection for 1xN, IDFT transformation, LLR demapping
\param [in] request Input request structure for PUSCH symbol processing
\param [out] response Output response structure PUSCH symbol processing
*/
template<size_t N_RX, typename T = Is16vec32>
inline void mimo_idft_llr_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
{
    /*
    * Transform precoding only support below configuration temporarily
    * N_TX=1 (1layer per UE per UE group)
    * DMRSconfigType=1 nrOfCDMs=2 interp_flag:disable
    */
    if (request->nEnableFoComp == 1)
        SymbolProcess<T, FO_E::enable, N_RX, 1, 1, 2, INTERP_E::disable>::mimo_idft_llr_avx512_fp16int16(request, response);
    else
        SymbolProcess<T, FO_E::disable, N_RX, 1, 1, 2, INTERP_E::disable>::mimo_idft_llr_avx512_fp16int16(request, response);

}

template<typename T, FO_E fo_flag = FO_E::disable, size_t N_RX = 16, size_t N_TX = 16, INTERP_E interp_flag = INTERP_E::disable>
inline void mimo_mmse_llr_avx512_interp_fp16int16_dmrstype(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
{
    if ((request->nDMRSType == 1) && (request->nNrOfCDMs == 2))
        SymbolProcess<T, fo_flag, N_RX, N_TX,1,2,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else if((request->nDMRSType == 1) && (request->nNrOfCDMs == 1))
        SymbolProcess<T, fo_flag, N_RX, N_TX,1,1,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else if((request->nDMRSType == 2) && (request->nNrOfCDMs == 3))
        SymbolProcess<T, fo_flag, N_RX, N_TX,2,3,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else if((request->nDMRSType == 2) && (request->nNrOfCDMs == 2))
        SymbolProcess<T, fo_flag, N_RX, N_TX,2,2,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
    else
        SymbolProcess<T, fo_flag, N_RX, N_TX,2,1,interp_flag>::mimo_mmse_llr_avx512_common(request, response);
}

template<typename T = Is16vec32, size_t N_RX = 16, size_t N_TX = 16>
inline void mimo_mmse_llr_avx512_interp_fp16int16(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
{
    if ((request->nLinInterpEnable == 1) &&
        (request->nDmrsChSymb == 2 || request->nDmrsChSymb == 4)) {
        if (request->nEnableFoComp == 1) {
            mimo_mmse_llr_avx512_interp_fp16int16_dmrstype<T, FO_E::enable, N_RX, N_TX, INTERP_E::enable>(request, response);
        } else {
            mimo_mmse_llr_avx512_interp_fp16int16_dmrstype<T, FO_E::disable, N_RX, N_TX, INTERP_E::enable>(request, response);
        }
    } else {
        if (request->nEnableFoComp == 1) {
            mimo_mmse_llr_avx512_interp_fp16int16_dmrstype<T, FO_E::enable, N_RX, N_TX, INTERP_E::disable>(request, response);
        } else {
            mimo_mmse_llr_avx512_interp_fp16int16_dmrstype<T, FO_E::disable, N_RX, N_TX, INTERP_E::disable>(request, response);
        }
    }
}

template<size_t N_RX = 16, size_t N_TX = 16, typename T = Is16vec32>
inline void mimo_mmse_llr_avx512_interp(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response)
{
#if defined (_BBLIB_SPR_)
    if (typeid(T) == typeid(CF16vec16)) {
        mimo_mmse_llr_avx512_interp_fp16int16<CF16vec16, N_RX, N_TX>(request, response);
    }
    else
#endif
    {
        mimo_mmse_llr_avx512_interp_fp16int16<CI16vec16, N_RX, N_TX>(request, response);
    }
}

/*! \brief PUSCH symbol processing, include MMSE MIMO detection,layer demap and LLR demap.
    \param [in] request Input request structure for PUSCH symbol processing.
    \param [out] response Output response structure for PUSCH symbol processing.
    \return 0 for success, and -1 for error
*/
template<typename T = Is16vec32>
inline int32_t bblib_pusch_symbol_processing_detection(
    bblib_pusch_symbol_processing_request *request,
    bblib_pusch_symbol_processing_response *response)
{
    uint16_t nLayer = request->nLayerInGroup;
    uint16_t nRxAnt = request->nRxAnt;
    uint16_t nUeInGroup = request->nUeInGroup;
    int32_t n_return = -1;

    if(unlikely(0==request->nSubCarrier))
    {
        printf("bblib_pusch_symbol_processing_avx512: Error! nSubCarrier == 0\n");
        return n_return;
    }
#if 0
    if (1 == nUeInGroup && 1 == nLayer && request->nTpFlag != 0)
    {
        if ((request->nSubCarrier >= 12) && (request->nSubCarrier <= MAX_IDFT_SIZE))
        {
            if (1 == nRxAnt)
            {
                mimo_idft_llr_avx512<1, T>(request, response);
                n_return = 0;
            }
            else if (2 == nRxAnt)
            {
                mimo_idft_llr_avx512<2, T>(request, response);
                n_return = 0;
            }
            else if (4 == nRxAnt)
            {
                mimo_idft_llr_avx512<4, T>(request, response);
                n_return = 0;
            }
            else if (8 == nRxAnt)
            {
                mimo_idft_llr_avx512<8, T>(request, response);
                n_return = 0;
            }
        }
        else
        {
           printf("Invalid subcarrier number %d for transform precoding! \n", request->nSubCarrier);
           return n_return;
        }
    }
#endif
    if (request->nTpFlag == 0)
    {
#if 0
        if (1 == nLayer)
        {
            if (1 == nRxAnt)
            {
                // mimo_llr_1x1_avx512(request, response);
                mimo_mmse_llr_avx512_interp<1, 1, T>(request, response);
                n_return = 0;
            }
            else if (2 == nRxAnt)
            {
                // mimo_llr_1x2_avx512(request, response); // interp
                mimo_mmse_llr_avx512_interp<2, 1, T>(request, response);
                n_return = 0;
            }
            else if (4 == nRxAnt)
            {
                // mimo_llr_1x4_avx512(request, response);
                mimo_mmse_llr_avx512_interp<4, 1, T>(request, response);
                n_return = 0;
            }
            else if (8 == nRxAnt)
            {
                // mimo_llr_1x4_avx512(request, response);
                mimo_mmse_llr_avx512_interp<8, 1, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                // mimo_llr_1x16_method1_avx512(request, response); // interp
                mimo_mmse_llr_avx512_interp<16, 1, T>(request, response);
                n_return = 0;
            }
        }
#endif
        if(2 == nLayer)
        {
            if (2 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<2, 2, T>(request, response);
                n_return = 0;
            }
            else if (4 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<4, 2, T>(request, response);
                n_return = 0;
            }
            else if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 2, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 2, T>(request, response);
                n_return = 0;
            }
        }
#if 0
        else if(3 == nLayer)
        {
            if (4 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<4, 3, T>(request, response);
                n_return = 0;
            }
            else if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 3, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 3, T>(request, response);
                n_return = 0;
            }
        }
#endif
        else if(4 == nLayer)
        {
            if (4 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<4, 4, T>(request, response);
                n_return = 0;
            }
            else if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 4, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 4, T>(request, response);
                n_return = 0;
            }
        }
#if 0
        else if(5 == nLayer)
        {
            if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 5, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 5, T>(request, response);
                n_return = 0;
            }
        }
        else if(6 == nLayer)
        {
            if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 6, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 6, T>(request, response);
                n_return = 0;
            }
        }
        else if(7 == nLayer)
        {
            if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 7, T>(request, response);
                n_return = 0;
            }
            else if (16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 7, T>(request, response);
                n_return = 0;
            }
        }
        else if(8 == nLayer)
        {
            if (8 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<8, 8, T>(request, response);
                n_return = 0;
            }
            else if(16 == nRxAnt)
            {
                mimo_mmse_llr_avx512_interp<16, 8, T>(request, response);
                n_return = 0;
            }
        }
#endif
    }
    if (n_return)
    {
        printf("bblib_pusch_symbol_processing_avx512: Error! don't support this combination -> nLayer = %d, nRxAnt = %d, nUeInGroup = %d, nTpFlag = %d\n",
            nLayer, nRxAnt, request->nUeInGroup, request->nTpFlag);
    }

    return (n_return);
}

/*! \brief PUSCH symbol processing detection, with LLR soft bits calculation. With AVX512 intrinsics.
    \param [in] request Input request structure for PUSCH symbol processing.
    \param [out] response Output response structure for PUSCH symbol processing..
    \return 0 for success, and -1 for error
*/
int32_t bblib_pusch_symbol_processing_avx512(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response) {
    LOG_TICK_INSTANCE(response);
    return bblib_pusch_symbol_processing_detection<CI16vec16>(request, response);

}
#ifdef _BBLIB_SPR_
int32_t bblib_pusch_symbol_processing_avx512_5gisa(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response* response) {
    LOG_TICK_INSTANCE(response);
    return bblib_pusch_symbol_processing_detection<CF16vec16>(request, response);
}
#endif


/*! \brief layer and llr demapping processing for 2 layers.
    \param [in] request Input request structure for layer and LLR demapping processing
    \param [out] response Output response structure for layer and LLR demapping processing
*/
template<size_t N_TX = 2 ,typename T = Is16vec32>
void layer_llr_avx512(bblib_layer_llr_demap_request *request, bblib_layer_llr_demap_response* response)
{
    using FloatSimd = typename DataType<T>::FloatSimd;

    Is16vec32 avxxTxSymbol[N_TX];
    FloatSimd ftempGain[N_TX];
    FloatSimd ftempPostSINR[N_TX];

    int8_t *pSoft[BBLIB_MAX_MU] = {NULL};
    int8_t *ppSoft = NULL;
    pSoft[0]=response->pLlr;
    enum bblib_modulation_order modOrder;

#ifndef _BBLIB_SPR_
    const int16_t llr_range_high = (1 << (request->nLlrSaturatedBits - 1)) - 1;
    const int16_t llr_range_low = -1 * llr_range_high;
#else
    // const T llr_range_high = _mm512_set1_ph(((1 << (request->nLlrSaturatedBits - 1))-1)/FLOAT16_SCALE_SQRT);
    // const T llr_range_low = _mm512_set1_ph((-1.0) * ((1 << (request->nLlrSaturatedBits - 1))-1)/FLOAT16_SCALE_SQRT);
    const int16_t llr_range_high = ((1 << (request->nLlrSaturatedBits - 1))-1)/FLOAT16_SCALE_SQRT;
    const int16_t llr_range_low = -1.0 * llr_range_high;
#endif

    ppSoft = (pSoft[0]);
    for (size_t nSCIdx = 0; nSCIdx < request->nLen; nSCIdx = nSCIdx + 16)
    {
        // do LLR demapping
        for (size_t i = 0; i < N_TX; i++) {
            #ifdef _BBLIB_SPR_
            ftempGain[i] = _mm512_loadu_ph((void const *)((float *)request->pMmseGain[i] + nSCIdx));
            ftempPostSINR[i] = _mm512_loadu_ph((void const*)((float *)request->pPostSINR[i] + nSCIdx));
            #elif _BBLIB_AVX512_
            ftempGain[i] = _mm512_loadu_ps((void const *)((float *)request->pMmseGain[i] + nSCIdx));
            ftempPostSINR[i] = _mm512_loadu_ps((void const*)((float *)request->pPostSINR[i] + nSCIdx));
            #endif
            auto pTempX = reinterpret_cast<T *>(request->pEqualOut[i] + nSCIdx*2);
            #ifdef _BBLIB_SPR_
            auto tmp = loadu(pTempX);
            avxxTxSymbol[i] = _mm512_cvtph_epi16(tmp);
            #else
            auto tmp = loadu(reinterpret_cast<M512 *>(pTempX));
            avxxTxSymbol[i] = tmp;
            #endif
        }
        // in current FPGA decoding implementation, LLR=p(b=0|r)/p(b=1|r)
        modOrder = request->eModOrder;
        auto pPostSINR = &ftempPostSINR[0];
        auto pGain = &ftempGain[0];
        int32_t nSc = (nSCIdx + 16 > request->nLen) * (request->nLen - (request->nLen)&0xfffffff0) + (nSCIdx + 16 <= request->nLen) * 16;
        auto pTx = avxxTxSymbol;
        if (modOrder == BBLIB_QPSK) {
            LLR<N_TX,FloatSimd>::llr_demap_layer_qpsk(pPostSINR,pGain,pTx,ppSoft,llr_range_low,llr_range_high, nSc);
        } else if (modOrder == BBLIB_QAM16) {
            LLR<N_TX ,FloatSimd>::llr_demap_layer_qam16(pPostSINR,pGain,pTx,ppSoft,llr_range_low,llr_range_high, nSc);
        } else if (modOrder == BBLIB_QAM64) {
            LLR<N_TX ,FloatSimd>::llr_demap_layer_qam64(pPostSINR,pGain,pTx,ppSoft,llr_range_low,llr_range_high, nSc);
        } else if (modOrder == BBLIB_QAM256) {
            LLR<N_TX ,FloatSimd>::llr_demap_layer_qam256(pPostSINR,pGain,pTx,ppSoft,llr_range_low,llr_range_high, nSc);
        }
	ppSoft = ppSoft + 16 * modOrder;
    }
}

/*! \brief layer and llr demapping processing.
    \param [in] request Input request structure for layer and LLR demapping processing
    \param [out] response Output response structure for layer and LLR demapping processing
*/
int32_t bblib_layer_llr_demap_processing_avx512(
    bblib_layer_llr_demap_request *request,
    bblib_layer_llr_demap_response* response)
{

    if(request->nLlrSaturatedBits < 2 || request->nLlrSaturatedBits > 8) {
        printf("Error! Not support nLlrSaturatedBits %d in pusch symbol process, valid range 2~8\n",
                request->nLlrSaturatedBits);
        return -1;
    }

    int32_t n_return = 0;
    switch(request->nLayer)
    {
        case 1:
     #ifndef _BBLIB_SPR_
            layer_llr_avx512<1 ,CI16vec16>(request,response);
     #else
            layer_llr_avx512<1 ,CF16vec16>(request,response);
     #endif
            break;
        case 2:
     #ifndef _BBLIB_SPR_
            layer_llr_avx512<2 ,CI16vec16>(request,response);
     #else
            layer_llr_avx512<2 ,CF16vec16>(request,response);
     #endif
            break;
        default:
           printf("\nError! Current layer demapping processing doesn't support the number of layer %d!\n",request->nLayer);
           n_return = -1;
    }
    return n_return;
}

#endif
