/*******************************************************************************
*
* <COPYRIGHT_TAG>
*
*******************************************************************************/

/**
 * @file dvec_inc.h
 * @brief This header file used to differentiate between different compilers and include appropriate dvec header file.
 * used global.
 */

#ifndef _DVEC_INC_H_
#define _DVEC_INC_H_

#include <immintrin.h>

#ifndef _WIN32
    #if defined (__ICC)
        #include <dvec.h>
    #else
        #include <dvec.h>
        #ifdef _BBLIB_SPR_
        #include "dvec_fp16.hpp"
        #endif
    #endif
#else
    #include <dvec.h>
#endif

#endif /* #ifndef _DVEC_INC_H_*/

