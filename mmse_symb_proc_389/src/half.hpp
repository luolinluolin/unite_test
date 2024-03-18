/*******************************************************************************
 *
 * INTEL CONFIDENTIAL
 * Copyright 2009-2020 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to the
 * source code ("Material") are owned by Intel Corporation or its suppliers or
 * licensors. Title to the Material remains with Intel Corporation or its
 * suppliers and licensors. The Material may contain trade secrets and proprietary
 * and confidential information of Intel Corporation and its suppliers and
 * licensors, and is protected by worldwide copyright and trade secret laws and
 * treaty provisions. No part of the Material may be used, copied, reproduced,
 * modified, published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be
 * express and approved by Intel in writing.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter this
 * notice or any other notice embedded in Materials by Intel or Intel's suppliers
 * or licensors in any way.
 *
 *  version: RefPHY-22.11
 *
 *******************************************************************************/

#pragma once

#include <complex>
#include <iostream>
#include <math.h>

/// Some code uses half-precision floating point. Try to use the compiler support where possible,
/// but otherwise fall back to an emulation library.

#ifdef __llvm__
using half = _Float16;
#else
using half = short float;
#endif

// Implement half rcp/rsqrt using float instead of half. This is very slightly more accurate
// (1ULP difference) but is so close that it is unnoticeable for most wireless kernels.
static inline half rcp(half value)
{
return half(1.0) / value;
}

static inline half rsqrt(half value)
{
return half(1.0) / half(sqrt(float(value)));
}


/*! \brief common to define data type
 *
 * data type declare
 */
#ifdef __llvm__
using float16 = half;
#endif
static inline std::ostream& operator<<(std::ostream& stream, half v)
{
  stream << (float)v;
  return stream;
}

static inline std::ostream& operator<<(std::ostream& stream, std::complex<half> v)
{
  // Force the use of the normal std::complex, so that it is consistent with other types.
  // stream << std::complex<float>(float(v.real()), float(v.imag()));
  stream << float(v.real()) << "+" << float(v.imag()) << "j";
  return stream;
}

// Overload some common operators.
static inline half abs(half a)
{
  union
  {
    half h;
    uint16_t ui;
  };

  h = a;
  ui &= 0x7FFF;

  return h;
}

static inline half sqrt(half in)
{
  return (half)sqrt((float)in);
}

static inline half min(half a, half b)
{
  if (a > b)
    return b;
  else
    return a;
}

static inline half max(half a, half b)
{
  if (a > b)
    return a;
  else
    return b;
}
