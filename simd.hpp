/*
 *  * @Description:  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *   * @Version: 1.0
 *    * @Autor: like
 *     * @Date: 2023-05-08 19:05:34
 *      * @LastEditors: like
 *       * @LastEditTime: 2023-05-08 20:54:09
 *        */
#pragma once
#include <immintrin.h>
#include "matrix.hpp"
#include <algorithm>
#include <iostream>
#include <typeinfo>

#ifdef _MSC_VER_ 
#   define FORCE_INLINE __forceinline
#elif defined __GNUC__ 
#   define FORCE_INLINE __inline__ __attribute__((always_inline))
#else
#   error "unsupport plantform"
#endif

/**
 *  * @brief SIMD family
 *   * 
 *    */
#define SIMD_FAMILY_MMX 64 /* unsupport type "SIMD_FAMILY_MMX" , it's too old. */ 
#define SIMD_FAMILY_SSE 128
#define SIMD_FAMILY_AVX 256
/**
 *  * @brief my cpu unsupport type 
 *   * 
 *    */
#define SIMD_FAMILY_AVX_512 512
/**
 *  * @brief default SIMD family 
 *   * 
 *    */
#define DEFAULT_SIMD_FAMILY SIMD_FAMILY_AVX

/**
 *  * @brief  SIMD family -> Intrinsics type
 *   * 
 *    */
#define _DECLARE_SIMD_TYPE_TRAITS(family)\
    template<> struct simd_traits<double, family>{ using simd_type = __m##family##d;using simd_type_in = double; };\
    template<> struct simd_traits<float, family>{ using simd_type = __m##family ;using simd_type_in = float; };\
    template<class T> struct simd_traits<T, family>{ using simd_type = __m##family##i; using simd_type_in = simd_type; };
#define DECLARE_SIMD_TYPE_TRAITS(family) _DECLARE_SIMD_TYPE_TRAITS(family)

#define REGIST_SIMD_AVX(Family) _REGIST_SIMD_AVX(Family)
#define _REGIST_SIMD_AVX(Family)\
    REGIST_SIMD(float, _ps, _ps, Family, Family##_)\
    REGIST_SIMD(double, _pd, _pd, Family, Family##_)\
    REGIST_SIMD(int8_t, _epi8, _si##Family, Family, Family##_)\
    REGIST_SIMD(int16_t, _epi16, _si##Family, Family, Family##_)\
    REGIST_SIMD(int32_t, _epi32, _si##Family, Family, Family##_)\
    REGIST_SIMD(int64_t, _epi64, _si##Family, Family, Family##_) \
    REGIST_SIMD(uint8_t, s_epu8, _si##Family, Family, Family##_)\
    REGIST_SIMD(uint16_t, s_epu16, _si##Family, Family, Family##_)

#define REGIST_SIMD_SSE(Family) _REGIST_SIMD_SSE(Family)
#define _REGIST_SIMD_SSE(Family)\
    REGIST_SIMD(float, _ps, _ps, Family, _)\
    REGIST_SIMD(double, _pd, _pd, Family, _)\
    REGIST_SIMD(int8_t, _epi8, _si##Family, Family, _)\
    REGIST_SIMD(int16_t, _epi16, _si##Family, Family, _)\
    REGIST_SIMD(int32_t, _epi32, _si##Family, Family, _)\
    REGIST_SIMD(int64_t, _epi64, _si##Family, Family, _)\
    REGIST_SIMD(uint8_t, s_epu8, _si##Family, Family, _)\
    REGIST_SIMD(uint16_t, s_epu16, _si##Family, Family, _)
/**
 *  * @brief SIMD api wrapper 
 *   * struct simd_cal_op<T, Family>{
 *    *      _mm"bit_count"_"operator"_"suffix"
 *     * }
 *      * 
 *       */
#define REGIST_SIMD(T, suffix, suffix1, Family, family)\
template<> struct simd_cal_op<T, Family>{\
    FORCE_INLINE static auto load(T const * mem_addr){return _mm##family##load##suffix1(reinterpret_cast<simd_type_in<T, Family> const*>(mem_addr));}\
    FORCE_INLINE static auto add(simd_type_out<T, Family> a, simd_type_out<T, Family> b ){return _mm##family##add##suffix(a, b);}\
    FORCE_INLINE static auto abs(simd_type_out<T, Family> a ){return _mm##family##abs##suffix(a);}\
};

namespace simd_cpp
{
    /**
 *      * @brief SIMD allocator, align memory by simd_family 
 *           * 
 *                * @tparam T 
 *                     * @tparam simd_family 
 *                          * @tparam Alloc 
 *                               */
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY, class Alloc = std::allocator<T>>
    struct simd_alloc
    {
        constexpr  static int simd_byte_align = static_cast<int>(simd_family) / 8;
        static_assert( 0 < simd_family && 0 == (simd_byte_align % sizeof(T)),  "unsupport simd family" );
        constexpr static int simd_data_align = simd_byte_align / sizeof(T); 

        static T* allocate(const size_t n)
        {
            Alloc a;
            return a.allocate(align(n));
        }
        static void deallocate(T* p,const  size_t n)
        {
            Alloc a;
            return a.deallocate(p, align(n));
        }
    private: 
        static int align(size_t n)
        {
            return  (n + simd_data_align - 1)  / simd_data_align * simd_data_align;
        }
    };
    /**
 *      * @brief get data type by simd family 
 *           * 
 *                * @tparam T 
 *                     * @tparam family 
 *                          */
    template<class T, int family = DEFAULT_SIMD_FAMILY> struct simd_traits;
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_SSE);
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_AVX);
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_AVX_512);
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_out = typename simd_traits<T, simd_family>::simd_type;
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_in = typename simd_traits<T, simd_family>::simd_type_in;
    /**
 *      * @brief SIMD API wrapper
 *           * 
 *                * @tparam T 
 *                     * @tparam simd_family 
 *                          */
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY> struct simd_cal_op;
    REGIST_SIMD_SSE(SIMD_FAMILY_SSE);
    REGIST_SIMD_AVX(SIMD_FAMILY_AVX);
}


template<class test_type = double>
void simd_test()
{
    using namespace simd_cpp;
    constexpr int buf_len = simd_alloc<test_type>::simd_data_align;

    printf("begin simd test : %s, %d\n", typeid(test_type).name(), buf_len);
    test_type buf[buf_len] = {0};
    for(int i = 0; i < buf_len; i++)
    {
        buf[i] = i;
    }
    mkl_cpp::matrix<test_type, 2> m(buf, {buf_len, 1});

    simd_type_out<test_type> a = simd_cal_op<test_type>::load(m.p);
    *reinterpret_cast<simd_type_out<test_type>*>(m.p) = simd_cal_op<test_type>::add(a, a);

    for (auto n : m)
    {
        std::cout << static_cast<double>(n) << " ";
    }
    printf("\nend simd test\n\n\n");
}
