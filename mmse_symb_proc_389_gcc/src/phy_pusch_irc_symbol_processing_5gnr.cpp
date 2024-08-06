/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/

/*
* @file   phy_pusch_irc_symbol_processing_5gnr.cpp
* @brief  Source code of External API for 5GNR PUSCH symbol processing.
*/

#include <stdio.h>
#include <stdint.h>

#include "phy_pusch_irc_symbol_processing_5gnr.h"

#if defined ( _WIN64 )
#define __func__ __FUNCTION__
#endif

int16_t
bblib_pusch_irc_symbol_processing_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts  by the
    *       jobs building the library and/or preparing the release packages.
    *       Do not edit the version string manually */
  //  const char *msg = "FlexRAN SDK bblib_pusch_irc_symbol_processing_5gnr version #DIRTY#";

//    return(bblib_sdk_version(&version, &msg, buffer_size));
}

struct bblib_pusch_irc_symbol_processing_init
{
    bblib_pusch_irc_symbol_processing_init(){
        /* Print library version number */
        //char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
       // bblib_pusch_irc_symbol_processing_version(version, sizeof(version));
       // printf("%s\n", version);
    }
};

bblib_pusch_irc_symbol_processing_init do_constructor_pusch_irc_symbol_processing;

int32_t
bblib_pusch_irc_symbol_processing(bblib_pusch_symbol_processing_request *request, bblib_pusch_symbol_processing_response *response)
{
#ifdef _BBLIB_SPR_
    return ( bblib_pusch_irc_symbol_processing_avx512_5gisa(request, response) );
#elif _BBLIB_AVX512_
    return ( bblib_pusch_irc_symbol_processing_avx512(request, response) );
#else
    printf("__func__ cannot run with this CPU type, needs AVX512\n");
    return (-1);
#endif
}

