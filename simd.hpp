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
#include <type_traits>

#ifdef _MSC_VER
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
    REGIST_SIMD_LOAD(float, _ps, Family, Family##_)\
    REGIST_SIMD_LOAD(double, _pd, Family, Family##_)\
    REGIST_SIMD_LOAD(int8_t, _si##Family, Family, Family##_)\
    REGIST_SIMD_LOAD(int16_t, _si##Family, Family, Family##_)\
    REGIST_SIMD_LOAD(int32_t, _si##Family, Family, Family##_)\
    REGIST_SIMD_LOAD(int64_t, _si##Family, Family, Family##_) \
    REGIST_SIMD_LOAD(uint8_t, _si##Family, Family, Family##_)\
    REGIST_SIMD_LOAD(uint16_t, _si##Family, Family, Family##_)\
    REGIST_SIMD_ADD(float, _ps, Family, Family##_)\
    REGIST_SIMD_ADD(double, _pd, Family, Family##_)

#define REGIST_SIMD_SSE(Family) _REGIST_SIMD_SSE(Family)
#define _REGIST_SIMD_SSE(Family)\
    REGIST_SIMD_LOAD(float, _ps, Family, _)\
    REGIST_SIMD_LOAD(double, _pd, Family, _)\
    REGIST_SIMD_LOAD(int8_t, _si##Family, Family, _)\
    REGIST_SIMD_LOAD(int16_t, _si##Family, Family, _)\
    REGIST_SIMD_LOAD(int32_t, _si##Family, Family, _)\
    REGIST_SIMD_LOAD(int64_t, _si##Family, Family, _)\
    REGIST_SIMD_LOAD(uint8_t, _si##Family, Family, _)\
    REGIST_SIMD_LOAD(uint16_t, _si##Family, Family, _)\
    REGIST_SIMD_ADD(float, _ps, Family, _)\
    REGIST_SIMD_ADD(double, _pd, Family, _)

/* regist simd load */
#define REGIST_SIMD_LOAD(T, suffix, Family, family) template<> struct simd_load<T, Family>{\
    FORCE_INLINE auto operator()(T const * mem_addr){return _mm##family##load##suffix(reinterpret_cast<simd_type_in<T, Family> const*>(mem_addr));}};
#define REGIST_SIMD_ADD(T, suffix, Family, family) template<> struct simd_add<T, Family>{\
    FORCE_INLINE auto operator()(simd_type_out<T, Family> a, simd_type_out<T, Family> b){return _mm##family##add##suffix(a, b);}};

//FORCE_INLINE static auto abs(simd_type_out<T, Family> a) { return _mm##family##abs##suffix(a); }
namespace simd_cpp
{
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

    template<class T, int family = DEFAULT_SIMD_FAMILY> struct simd_traits;
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_SSE);
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_AVX);
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_AVX_512);
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_out = typename simd_traits<T, simd_family>::simd_type;
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_in = typename simd_traits<T, simd_family>::simd_type_in;

    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_load;
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_add;
    REGIST_SIMD_SSE(SIMD_FAMILY_SSE);
    REGIST_SIMD_AVX(SIMD_FAMILY_AVX);
}
//template<class T, class = decltype(&T::operator())>
//struct is_api_registed;
//template<class T>
//struct is_api_registed<T, decltype(&T::operator())> : std::false_type {};
//template<class T>
//struct is_api_registed<T, void> : std::true_type {};



//// Âú×ãÅÐ¶ÏÆ¥Åä
//template<typename F, typename... Args,
//    typename = decltype(std::declval<F>()(std::declval<Args>()...))> // ·µ»ØÖµÅÐ¶Ï
//constexpr std::true_type IsValidImpl(std::nullptr_t) {
//    return std::true_type();
//}
//
//// ²»Âú×ãÅÐ¶ÏÆ¥Åä
//template<typename F, typename... Args>
//constexpr std::false_type IsValidImpl(...) {
//    return std::false_type();
//}
//template<class T>
//constexpr auto is_api_registed = []() {
//    return [](auto&&... Args) {
//        // µ÷ÓÃ IsValidImpl ½øÐÐÅÐ¶Ï
//        return IsValidImpl<T>(nullptr);
//    };
//};


//using is_api_registed = decltype(IsValidImpl<T>());
template<class test_type = double>
auto simd_load_test()
{
    using namespace simd_cpp;
    constexpr int buf_len = simd_alloc<test_type>::simd_data_align;

    printf("begin simd load test : %s, %d\n", typeid(test_type).name(), buf_len);
    std::array<test_type, buf_len> buf = { 0 };
    for (int i = 0; i < buf_len; i++)
    {
        buf[i] = static_cast<test_type>(i);
    }
    simd_type_out<test_type> a = simd_load<test_type>()(buf.data());
    assert(0 == memcmp(&a, buf.data(), sizeof(a)));
    return buf;
}


template<class test_type = double>
void simd_test()
{
    using namespace simd_cpp;
    
    auto buf = simd_load_test<test_type>();
    mkl_cpp::matrix<test_type, 2> m(buf.data(), {std::size(buf), 1});

    //std::enable_if_t<>
    //if constexpr(std::is_same<void,>)
    simd_type_out<test_type> a = simd_load<test_type>()(m.p);
    //*reinterpret_cast<simd_type_out<test_type>*>(m.p) = simd_add<test_type>()(a, a);
    //printf("%s\n", typeid(decltype(simd_add<test_type>())).name());
    //printf("simd_add : %s\n", is_api_registed<simd_add<test_type>>::value ? "registed" : "unregisted");
    for (auto n : m)
    {
        std::cout << static_cast<double>(n) << " ";
    }
    printf("\nend simd test\n\n\n");
}
