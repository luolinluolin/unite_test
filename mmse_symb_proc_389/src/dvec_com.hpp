/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/**
 * @file dvec_comm.hpp
 * @brief This header file used to define dvec atrubute and alias name
 * used global.
 */
#ifndef _WIN32

    #ifdef __ICC
        #if __ICC >= 1910
            #include <dvec.h>
        #else
            #include "dvec_icc.h"
        #endif
    #elif defined(__GNUC__) && !defined(__clang__)
        #include "dvec_fvec_ivec_gcc.h"
    #else
        #include <dvec.h>
    #endif

#else
    #include "dvec_icc.h"
#endif

#pragma once
namespace W_SDK {
/*! \brief common to define compiler option
 *
 * compiler option declare
 */
#ifdef _WIN32
#define FORCE_INLINE
#else
#define FORCE_INLINE  __attribute__((always_inline))
#endif

/*! \brief common function to define mask
 *
 * mask declare
 */
using Mask64 = __mmask64;
using Mask32 = __mmask32;
using Mask16 = __mmask16;
using Mask8  = __mmask8;
using M512 = M512vec;
// using M512 = __m512i;

}
