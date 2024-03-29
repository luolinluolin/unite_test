/**********************************************************************
*
* <COPYRIGHT_TAG>
*
**********************************************************************/
/*!
    \file   common_typedef_simd.hpp
    \brief  common function definitions used throughout the SDK
*/

#ifndef _COMMON_TYPEDEF_SIMD_HPP_
#define _COMMON_TYPEDEF_SIMD_HPP_
#include <immintrin.h>
#include "gcc_inc.h"

/*!
* \brief Class definition of type I32vec8
* Defined here as this is not yet supported in dvec.h
*/

#define GCC_COMPILER (defined(__GNUC__) && !defined(__clang__))

#if defined(__ICC) || defined(_WIN32) || defined(__llvm__) || defined(GCC_COMPILER)
namespace COM{
class I32vec8
{
public:

    static constexpr unsigned k_numElements = 32;

    I32vec8() { simd = _mm256_setzero_si256(); }

    //! Simple copy constructor.
    I32vec8(__m256i v) : simd(v) { }

    //! Allow implicit conversion to the underlying type.
    operator __m256i() const { return simd; }

    // Basic broadcast of scalar value.
    I32vec8(int value)
    {
        simd = _mm256_set1_epi32(value);
    }

    friend I32vec8 operator +(const I32vec8 &a, const I32vec8 &b)
    {
        return (I32vec8)_mm256_add_epi32(a, b);
    }

    I32vec8& operator += (I32vec8 rhs)
    {
        simd = _mm256_add_epi32(simd, rhs.simd);
        return *this;
    }

    __m256i simd;
};

/*!
* \brief Class definition of type I8vec32
* Defined here as this is not yet supported in dvec.h
*/
class I8vec32
{
public:

    static constexpr unsigned k_numElements = 32;

    I8vec32() { simd = _mm256_setzero_si256(); }

    //! Simple copy constructor.
    I8vec32(__m256i v) : simd(v) { }

    //! Allow implicit conversion to the underlying type.
    operator __m256i() const { return simd; }

    //! Set the first 32 values
    I8vec32(char src0, char src1, char src2, char src3, char src4, char src5, char src6, char src7,
        char src8, char src9, char src10, char src11, char src12, char src13, char src14, char src15,
        char src16, char src17, char src18, char src19, char src20, char src21, char src22, char src23,
        char src24, char src25, char src26, char src27, char src28, char src29, char src30, char src31)
    {
        simd = _mm256_setr_epi8(src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10,
            src11, src12, src13, src14, src15, src16, src17, src18, src19, src20,
            src21, src22, src23, src24, src25, src26, src27, src28, src29, src30,
            src31);
    }

    __m256i simd;
};

/*!
 * \brief Class definition of type I16vec16
 * Defined here as this is not yet supported in dvec.h
 */
class I16vec16
{
public:
    static constexpr unsigned k_numElements = 16;

    I16vec16() { simd = _mm256_setzero_si256(); }

    //! Simple copy constructor.
    I16vec16(__m256i v) : simd(v) { }

    // Allow implicit conversion to the underlying type.
    operator __m256i() const { return simd; }


    //! Basic broadcast of scalar value.
    I16vec16(short value)
    {
      simd = _mm256_set1_epi16(value);
    }

    //! Set the first 16 values
    I16vec16(short src0, short src1, short src2, short src3,
             short src4, short src5, short src6, short src7,
             short src8, short src9, short src10, short src11,
             short src12, short src13, short src14, short src15)
    {
      simd =  _mm256_setr_epi16(src0, src1, src2, src3, src4, src5, src6, src7,
                                src8, src9, src10, src11, src12, src13, src14, src15);
    }

    //! Zero out every element.
    void set_zero() {
      simd = _mm256_setzero_si256();
    }

    I16vec16 operator-() const {
      I16vec16 result;
      result.set_zero();
      result -= *this;
      return result;
    }

    friend I16vec16 operator+ (const I16vec16 &a, const I16vec16 &b)
    {
        return (I16vec16)_mm256_add_epi16(a, b);
    }

    friend I16vec16 operator- (const I16vec16 &a, const I16vec16 &b)
    {
      return (I16vec16)_mm256_sub_epi16(a, b);
    }

    friend I16vec16 operator* (const I16vec16 &a, const I16vec16 &b)
    {
      return (I16vec16)_mm256_mullo_epi16(a, b);
    }

    friend I16vec16 operator>> (const I16vec16 &a, int count)
    {
      return (I16vec16)_mm256_srli_epi16(a, count);
    }

    friend I16vec16 operator<< (const I16vec16 &a, int count)
    {
      return (I16vec16)_mm256_slli_epi16(a, count);
    }

    I16vec16& operator -= (const I16vec16& rhs)
    {
      simd = _mm256_sub_epi16(simd, rhs.simd);
      return *this;
    }

    I16vec16& operator += (I16vec16 rhs)
    {
      simd = _mm256_add_epi16(simd, rhs.simd);
      return *this;
    }

    I16vec16& operator *= (const I16vec16& rhs)
    {
      simd = _mm256_mulhrs_epi16(simd, rhs.simd);
      return *this;
    }

#if !(defined(__GNUC__) && !defined(__clang__) && !defined(__ICC))
    I16vec16& operator /= (I16vec16 rhs)
    {
      simd = _mm256_div_epi16(simd, rhs.simd);
      return *this;
    }
#endif

    short operator[](int index) const
    {
      const short* hp = (const short*)&simd;
      return hp[index];
    }

    short& operator[](int index)
    {
      short* hp = (short*)&simd;
      return hp[index];
    }

    __m256i simd;
};
}
#endif

#endif /* _COMMON_TYPEDEF_SIMD_HPP_ */

