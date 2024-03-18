/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/*!
    \file   phy_matrix_inv_lemma.cpp
    \brief  matrix inversion lemma based matrix inversion.
*/


/*******************************************************************************
* Include private header files
*******************************************************************************/
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "common_typedef_sdk.h"
#include "phy_rx_mimo_mmse.h"

#define N_RX_4 4 // 4 RX antenna
#ifdef _BBLIB_AVX512_
const static __m512 allfZero = _mm512_setzero_ps();


#define GET_ADJ_AND_DET_AVX512(matInRe, matInIm, matOutRe, matOutIm, det) \
{ \
    matOutRe[0][0] = matInRe[1][1]; \
    matOutRe[0][1] = _mm512_sub_ps(allfZero, matInRe[0][1]); \
    matOutRe[1][0] = _mm512_sub_ps(allfZero, matInRe[1][0]); \
    matOutRe[1][1] = matInRe[0][0]; \
    \
    matOutIm[0][0] = allfZero; \
    matOutIm[0][1] = _mm512_sub_ps(allfZero, matInIm[0][1]); \
    matOutIm[1][0] = _mm512_sub_ps(allfZero, matInIm[1][0]); \
    matOutIm[1][1] = allfZero; \
    \
    det = _mm512_mul_ps(matInRe[0][0], matInRe[1][1]); \
    det = _mm512_sub_ps(det, _mm512_mul_ps(matInRe[0][1], matInRe[0][1])); \
    det = _mm512_sub_ps(det, _mm512_mul_ps(matInIm[0][1], matInIm[0][1])); \
}

#define GET_M2X2_MUL_M2X2_AVX512(matARe, matAIm, matBRe, matBIm, matOutRe, matOutIm) \
{ \
    matOutRe[0][0] = _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[0][0], matBRe[0][0]),  \
                     _mm512_mul_ps(matAIm[0][0], matBIm[0][0])); \
    matOutRe[0][0] = _mm512_add_ps(matOutRe[0][0], _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[0][1], matBRe[1][0]),  \
                     _mm512_mul_ps(matAIm[0][1], matBIm[1][0]))); \
    \
    matOutRe[0][1] = _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[0][0], matBRe[0][1]),  \
                     _mm512_mul_ps(matAIm[0][0], matBIm[0][1])); \
    matOutRe[0][1] = _mm512_add_ps(matOutRe[0][1], _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[0][1], matBRe[1][1]),  \
                     _mm512_mul_ps(matAIm[0][1], matBIm[1][1]))); \
    \
    matOutRe[1][0] = _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[1][0], matBRe[0][0]),  \
                     _mm512_mul_ps(matAIm[1][0], matBIm[0][0])); \
    matOutRe[1][0] = _mm512_add_ps(matOutRe[1][0], _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[1][1], matBRe[1][0]),  \
                     _mm512_mul_ps(matAIm[1][1], matBIm[1][0]))); \
    \
    matOutRe[1][1] = _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[1][0], matBRe[0][1]),  \
                     _mm512_mul_ps(matAIm[1][0], matBIm[0][1])); \
    matOutRe[1][1] = _mm512_add_ps(matOutRe[1][1], _mm512_sub_ps( \
                     _mm512_mul_ps(matARe[1][1], matBRe[1][1]),  \
                     _mm512_mul_ps(matAIm[1][1], matBIm[1][1]))); \
    \
    matOutIm[0][0] = _mm512_add_ps( \
                     _mm512_mul_ps(matARe[0][0], matBIm[0][0]),  \
                     _mm512_mul_ps(matAIm[0][0], matBRe[0][0])); \
    matOutIm[0][0] = _mm512_add_ps(matOutIm[0][0], _mm512_add_ps( \
                     _mm512_mul_ps(matARe[0][1], matBIm[1][0]),  \
                     _mm512_mul_ps(matAIm[0][1], matBRe[1][0]))); \
    \
    matOutIm[0][1] = _mm512_add_ps( \
                     _mm512_mul_ps(matARe[0][0], matBIm[0][1]),  \
                     _mm512_mul_ps(matAIm[0][0], matBRe[0][1])); \
    matOutIm[0][1] = _mm512_add_ps(matOutIm[0][1], _mm512_add_ps( \
                     _mm512_mul_ps(matARe[0][1], matBIm[1][1]),  \
                     _mm512_mul_ps(matAIm[0][1], matBRe[1][1]))); \
    \
    matOutIm[1][0] = _mm512_add_ps( \
                     _mm512_mul_ps(matARe[1][0], matBIm[0][0]),  \
                     _mm512_mul_ps(matAIm[1][0], matBRe[0][0])); \
    matOutIm[1][0] = _mm512_add_ps(matOutIm[1][0], _mm512_add_ps( \
                     _mm512_mul_ps(matARe[1][1], matBIm[1][0]),  \
                     _mm512_mul_ps(matAIm[1][1], matBRe[1][0]))); \
    \
    matOutIm[1][1] = _mm512_add_ps( \
                     _mm512_mul_ps(matARe[1][0], matBIm[0][1]),  \
                     _mm512_mul_ps(matAIm[1][0], matBRe[0][1])); \
    matOutIm[1][1] = _mm512_add_ps(matOutIm[1][1], _mm512_add_ps( \
                     _mm512_mul_ps(matARe[1][1], matBIm[1][1]),  \
                     _mm512_mul_ps(matAIm[1][1], matBRe[1][1]))); \
}

int32_t matrix_inv_lemma_4x4(__m512 ftempARe[4][4], __m512 ftempAIm[4][4],
    __m512 ftempInvARe[4][4], __m512 ftempInvAIm[4][4], int16_t nFixedBits)
{
    // temp matrix and variables for matrix inversion
    __m512 matARe[2][2], matBRe[2][2], matCRe[2][2], matDRe[2][2];
    __m512 matAIm[2][2], matBIm[2][2], matCIm[2][2], matDIm[2][2];
    __m512 matAoRe[2][2], matDoRe[2][2], matPoRe[2][2], matQoRe[2][2];
    __m512 matAoIm[2][2], matDoIm[2][2], matPoIm[2][2], matQoIm[2][2];
    __m512 matDoCRe[2][2], matBDoCRe[2][2], matAoBRe[2][2], matCAoBRe[2][2];
    __m512 matDoCIm[2][2], matBDoCIm[2][2], matAoBIm[2][2], matCAoBIm[2][2];
    __m512 matAoBQoRe[2][2], matAoBQoIm[2][2];
    __m512 aDet, dDet, pDet, qDet;

    float nFactor = (float)(1 << nFixedBits);
    __m512 avxRightShift = _mm512_set1_ps(1 / nFactor);

    // load mA, mB, mC, mD
    matARe[0][0] = _mm512_mul_ps(ftempARe[0][0], avxRightShift); matAIm[0][0] = _mm512_mul_ps(ftempAIm[0][0], avxRightShift);
    matARe[0][1] = _mm512_mul_ps(ftempARe[0][1], avxRightShift); matAIm[0][1] = _mm512_mul_ps(ftempAIm[0][1], avxRightShift);
    matARe[1][0] = _mm512_mul_ps(ftempARe[1][0], avxRightShift); matAIm[1][0] = _mm512_mul_ps(ftempAIm[1][0], avxRightShift);
    matARe[1][1] = _mm512_mul_ps(ftempARe[1][1], avxRightShift); matAIm[1][1] = _mm512_mul_ps(ftempAIm[1][1], avxRightShift);

    matBRe[0][0] = _mm512_mul_ps(ftempARe[0][2], avxRightShift); matBIm[0][0] = _mm512_mul_ps(ftempAIm[0][2], avxRightShift);
    matBRe[0][1] = _mm512_mul_ps(ftempARe[0][3], avxRightShift); matBIm[0][1] = _mm512_mul_ps(ftempAIm[0][3], avxRightShift);
    matBRe[1][0] = _mm512_mul_ps(ftempARe[1][2], avxRightShift); matBIm[1][0] = _mm512_mul_ps(ftempAIm[1][2], avxRightShift);
    matBRe[1][1] = _mm512_mul_ps(ftempARe[1][3], avxRightShift); matBIm[1][1] = _mm512_mul_ps(ftempAIm[1][3], avxRightShift);

    matCRe[0][0] = _mm512_mul_ps(ftempARe[2][0], avxRightShift); matCIm[0][0] = _mm512_mul_ps(ftempAIm[2][0], avxRightShift);
    matCRe[0][1] = _mm512_mul_ps(ftempARe[2][1], avxRightShift); matCIm[0][1] = _mm512_mul_ps(ftempAIm[2][1], avxRightShift);
    matCRe[1][0] = _mm512_mul_ps(ftempARe[3][0], avxRightShift); matCIm[1][0] = _mm512_mul_ps(ftempAIm[3][0], avxRightShift);
    matCRe[1][1] = _mm512_mul_ps(ftempARe[3][1], avxRightShift); matCIm[1][1] = _mm512_mul_ps(ftempAIm[3][1], avxRightShift);

    matDRe[0][0] = _mm512_mul_ps(ftempARe[2][2], avxRightShift); matDIm[0][0] = _mm512_mul_ps(ftempAIm[2][2], avxRightShift);
    matDRe[0][1] = _mm512_mul_ps(ftempARe[2][3], avxRightShift); matDIm[0][1] = _mm512_mul_ps(ftempAIm[2][3], avxRightShift);
    matDRe[1][0] = _mm512_mul_ps(ftempARe[3][2], avxRightShift); matDIm[1][0] = _mm512_mul_ps(ftempAIm[3][2], avxRightShift);
    matDRe[1][1] = _mm512_mul_ps(ftempARe[3][3], avxRightShift); matDIm[1][1] = _mm512_mul_ps(ftempAIm[3][3], avxRightShift);

    // 1) get adjoint and determinant of mA (Hermite matrix)
    GET_ADJ_AND_DET_AVX512(matARe, matAIm, matAoRe, matAoIm, aDet);

    // 2) get adjoint and determinant of mD (Hermite matrix)
    GET_ADJ_AND_DET_AVX512(matDRe, matDIm, matDoRe, matDoIm, dDet);

    // 3) get matrix mP
    // 3.1 dA = d * A (A -> dA) (Hermite matrix)
    matARe[0][0] = _mm512_mul_ps(matARe[0][0], dDet);
    matARe[0][1] = _mm512_mul_ps(matARe[0][1], dDet);
    matARe[1][0] = _mm512_mul_ps(matARe[1][0], dDet);
    matARe[1][1] = _mm512_mul_ps(matARe[1][1], dDet);

    matAIm[0][1] = _mm512_mul_ps(matAIm[0][1], dDet);
    matAIm[1][0] = _mm512_mul_ps(matAIm[1][0], dDet);

    // 3.2 DoC = Do * C
    GET_M2X2_MUL_M2X2_AVX512(matDoRe, matDoIm, matCRe, matCIm,
                  matDoCRe, matDoCIm);

    // 3.3 BDoC = B * DoC
    GET_M2X2_MUL_M2X2_AVX512(matBRe, matBIm, matDoCRe, matDoCIm,
                  matBDoCRe, matBDoCIm);

    // 3.4 P = dA - BDoC (A -> dA -> P) (Hermite matrix)
    matARe[0][0] = _mm512_sub_ps(matARe[0][0], matBDoCRe[0][0]);
    matARe[0][1] = _mm512_sub_ps(matARe[0][1], matBDoCRe[0][1]);
    matARe[1][0] = _mm512_sub_ps(matARe[1][0], matBDoCRe[1][0]);
    matARe[1][1] = _mm512_sub_ps(matARe[1][1], matBDoCRe[1][1]);

    matAIm[0][1] = _mm512_sub_ps(matAIm[0][1], matBDoCIm[0][1]);
    matAIm[1][0] = _mm512_sub_ps(matAIm[1][0], matBDoCIm[1][0]);

    // 4) get matrix mQ
    // 4.1 aD = a * D (D -> aD) (Hermite matrix)
    matDRe[0][0] = _mm512_mul_ps(matDRe[0][0], aDet);
    matDRe[0][1] = _mm512_mul_ps(matDRe[0][1], aDet);
    matDRe[1][0] = _mm512_mul_ps(matDRe[1][0], aDet);
    matDRe[1][1] = _mm512_mul_ps(matDRe[1][1], aDet);

    matDIm[0][1] = _mm512_mul_ps(matDIm[0][1], aDet);
    matDIm[1][0] = _mm512_mul_ps(matDIm[1][0], aDet);

    // 4.2 AoB = Ao * B
    GET_M2X2_MUL_M2X2_AVX512(matAoRe, matAoIm, matBRe, matBIm,
                  matAoBRe, matAoBIm);

    // 4.3 CAoB = C * AoB
    GET_M2X2_MUL_M2X2_AVX512(matCRe, matCIm, matAoBRe, matAoBIm,
                  matCAoBRe, matCAoBIm);

    // 4.4 Q = aD - CAoB (D -> aD -> Q) (Hermite matrix)
    matDRe[0][0] = _mm512_sub_ps(matDRe[0][0], matCAoBRe[0][0]);
    matDRe[0][1] = _mm512_sub_ps(matDRe[0][1], matCAoBRe[0][1]);
    matDRe[1][0] = _mm512_sub_ps(matDRe[1][0], matCAoBRe[1][0]);
    matDRe[1][1] = _mm512_sub_ps(matDRe[1][1], matCAoBRe[1][1]);

    matDIm[0][1] = _mm512_sub_ps(matDIm[0][1], matCAoBIm[0][1]);
    matDIm[1][0] = _mm512_sub_ps(matDIm[1][0], matCAoBIm[1][0]);

    // 5) get adjoint and determinant of mP (P <- A) (Hermite matrix)
    GET_ADJ_AND_DET_AVX512(matARe, matAIm, matPoRe, matPoIm, pDet);

    // p = 1/p (p -> 1/p)
    pDet = _mm512_rcp14_ps(pDet);

    // 6) get adjoint and determinant of mQ (Q <- D) (Hermite matrix)
    GET_ADJ_AND_DET_AVX512(matDRe, matDIm, matQoRe, matQoIm, qDet);

    // q = 1/q (q -> 1/q)
    qDet = _mm512_rcp14_ps(qDet);

    // 7) get matrix R (R <- tempA)
    // 7.1 R(1:2; 1:2) = dPo (Hermite matrix)
    ftempInvARe[0][0] = _mm512_mul_ps(matPoRe[0][0], dDet);
    ftempInvARe[0][1] = _mm512_mul_ps(matPoRe[0][1], dDet);
    ftempInvARe[1][0] = _mm512_mul_ps(matPoRe[1][0], dDet);
    ftempInvARe[1][1] = _mm512_mul_ps(matPoRe[1][1], dDet);

    ftempInvAIm[0][0] = allfZero;
    ftempInvAIm[0][1] = _mm512_mul_ps(matPoIm[0][1], dDet);
    ftempInvAIm[1][0] = _mm512_mul_ps(matPoIm[1][0], dDet);
    ftempInvAIm[1][1] = allfZero;

    // 7.2 R(1:2; 3:4) = -AoB * Qo
    matAoBRe[0][0] = _mm512_sub_ps(allfZero, matAoBRe[0][0]);
    matAoBRe[0][1] = _mm512_sub_ps(allfZero, matAoBRe[0][1]);
    matAoBRe[1][0] = _mm512_sub_ps(allfZero, matAoBRe[1][0]);
    matAoBRe[1][1] = _mm512_sub_ps(allfZero, matAoBRe[1][1]);

    matAoBIm[0][0] = _mm512_sub_ps(allfZero, matAoBIm[0][0]);
    matAoBIm[0][1] = _mm512_sub_ps(allfZero, matAoBIm[0][1]);
    matAoBIm[1][0] = _mm512_sub_ps(allfZero, matAoBIm[1][0]);
    matAoBIm[1][1] = _mm512_sub_ps(allfZero, matAoBIm[1][1]);

    GET_M2X2_MUL_M2X2_AVX512(matAoBRe, matAoBIm, matQoRe, matQoIm,
                      matAoBQoRe, matAoBQoIm);

    // 7.3 R(3:4; 1:2) = -DoC * Po
    // delete this part, because invA is a Hermite matrix

    // 7.4 R(3:4; 3:4) = aQo (Hermite matrix)
    ftempInvARe[2][2] = _mm512_mul_ps(matQoRe[0][0], aDet);
    ftempInvARe[2][3] = _mm512_mul_ps(matQoRe[0][1], aDet);
    ftempInvARe[3][2] = _mm512_mul_ps(matQoRe[1][0], aDet);
    ftempInvARe[3][3] = _mm512_mul_ps(matQoRe[1][1], aDet);

    ftempInvAIm[2][2] = allfZero;
    ftempInvAIm[2][3] = _mm512_mul_ps(matQoIm[0][1], aDet);
    ftempInvAIm[3][2] = _mm512_mul_ps(matQoIm[1][0], aDet);
    ftempInvAIm[3][3] = allfZero;


    // 8) get inversion of tempA : invA = inv(dH' * dH + sigma2 * I)
    // 8.1 R(:, 1:2) = p * tempA(:, 1:2)
    // 8.2 R(:, 3:4) = q * tempA(:, 3:4)

    ftempInvARe[0][0] = _mm512_mul_ps(ftempInvARe[0][0], pDet);
    ftempInvARe[0][1] = _mm512_mul_ps(ftempInvARe[0][1], pDet);
    ftempInvARe[0][2] = _mm512_mul_ps(matAoBQoRe[0][0], qDet);
    ftempInvARe[0][3] = _mm512_mul_ps(matAoBQoRe[0][1], qDet);
    ftempInvARe[1][1] = _mm512_mul_ps(ftempInvARe[1][1], pDet);
    ftempInvARe[1][2] = _mm512_mul_ps(matAoBQoRe[1][0], qDet);
    ftempInvARe[1][3] = _mm512_mul_ps(matAoBQoRe[1][1], qDet);
    ftempInvARe[2][2] = _mm512_mul_ps(ftempInvARe[2][2], qDet);
    ftempInvARe[2][3] = _mm512_mul_ps(ftempInvARe[2][3], qDet);
    ftempInvARe[3][3] = _mm512_mul_ps(ftempInvARe[3][3], qDet);

    ftempInvARe[1][0] = ftempInvARe[0][1];
    ftempInvARe[2][0] = ftempInvARe[0][2];
    ftempInvARe[2][1] = ftempInvARe[1][2];
    ftempInvARe[3][0] = ftempInvARe[0][3];
    ftempInvARe[3][1] = ftempInvARe[1][3];
    ftempInvARe[3][2] = ftempInvARe[2][3];

    ftempInvAIm[0][1] = _mm512_mul_ps(ftempInvAIm[0][1], pDet);
    ftempInvAIm[0][2] = _mm512_mul_ps(matAoBQoIm[0][0], qDet);
    ftempInvAIm[0][3] = _mm512_mul_ps(matAoBQoIm[0][1], qDet);
    ftempInvAIm[1][2] = _mm512_mul_ps(matAoBQoIm[1][0], qDet);
    ftempInvAIm[1][3] = _mm512_mul_ps(matAoBQoIm[1][1], qDet);
    ftempInvAIm[2][3] = _mm512_mul_ps(ftempInvAIm[2][3], qDet);

    ftempInvAIm[1][0] = _mm512_sub_ps(allfZero, ftempInvAIm[0][1]);
    ftempInvAIm[2][0] = _mm512_sub_ps(allfZero, ftempInvAIm[0][2]);
    ftempInvAIm[2][1] = _mm512_sub_ps(allfZero, ftempInvAIm[1][2]);
    ftempInvAIm[3][0] = _mm512_sub_ps(allfZero, ftempInvAIm[0][3]);
    ftempInvAIm[3][1] = _mm512_sub_ps(allfZero, ftempInvAIm[1][3]);
    ftempInvAIm[3][2] = _mm512_sub_ps(allfZero, ftempInvAIm[2][3]);

    return 0;
}
#endif
