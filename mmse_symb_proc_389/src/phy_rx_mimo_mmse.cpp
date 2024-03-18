/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/*
* @file   phy_rx_mimo_mmse.cpp
* @brief  Source code of External API for MMSE MIMO with post SINR, for 5GNR
*/

#include <stdio.h>
#include <stdint.h>

#include "sdk_version.h"
#include "phy_rx_mimo_mmse.h"

#if defined ( _WIN64 )
#define __func__ __FUNCTION__
#endif

int16_t
bblib_mimo_mmse_detection_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts  by the
    *       jobs building the library and/or preparing the release packages.
    *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_mimo_mmse_detection_5gnr version #DIRTY#";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}

struct bblib_mimo_mmse_detection_init
{
    bblib_mimo_mmse_detection_init(){
        /* Print library version number */
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_mimo_mmse_detection_version(version, sizeof(version));
        printf("%s\n", version);
    }
};

bblib_mimo_mmse_detection_init do_constructor_equalization;

int32_t
bblib_mimo_mmse_detection(bblib_mmse_mimo_request *request, bblib_mmse_mimo_response *response)
{
#ifdef _BBLIB_AVX512_
    return ( bblib_mimo_mmse_detection_avx512(request, response) );
#else
    printf("__func__ cannot run with this CPU type, needs AVX512\n");
    return (-1);
#endif
}

