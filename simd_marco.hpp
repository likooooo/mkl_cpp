#pragma once

#include <type_traits>

#ifdef _MSC_VER
#   define FORCE_INLINE __forceinline
#elif defined __GNUC__ 
#   define FORCE_INLINE __inline__ __attribute__((always_inline))
#else
#   error "unsupport plantform"
#endif

#define SIMD_FAMILY_SSE 128
#define SIMD_FAMILY_AVX 256
#define SIMD_FAMILY_AVX_512 512

#define DEFAULT_SIMD_FAMILY SIMD_FAMILY_AVX

#define _DECLARE_SIMD_TYPE_TRAITS(family)\
    template<> struct simd_traits<double, family>{\
        using simd_type = __m##family##d;\
        using simd_type_in = double;\
    };\
    template<> struct simd_traits<float, family>{\
        using simd_type = __m##family ;\
        using simd_type_in = float;\
    };\
    template<class T> struct simd_traits<T, family>{\
        using simd_type = __m##family##i;\
        using simd_type_in = simd_type;\
    };
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
    REGIST_SIMD_LOAD(uint32_t, _si##Family, Family, Family##_)\
    \
    REGIST_SIMD_SET(float, _ps, Family, Family##_)\
    REGIST_SIMD_SET(double, _pd, Family, Family##_)\
    REGIST_SIMD_SET(int8_t, _epi8, Family, Family##_)\
    REGIST_SIMD_SET(int16_t, _epi16, Family, Family##_)\
    REGIST_SIMD_SET(int32_t, _epi32, Family, Family##_)\
    \
    REGIST_SIMD_GATHER(float, _ps, Family, Family##_)\
    REGIST_SIMD_GATHER(double, _pd, Family, Family##_)\
    REGIST_SIMD_GATHER(int32_t, _epi32, Family, Family##_)\
    \
    REGIST_SIMD_ADD(float, _ps, Family, Family##_)\
    REGIST_SIMD_ADD(double, _pd, Family, Family##_)\
    REGIST_SIMD_ADD(int8_t, _epi8, Family, Family##_)\
    REGIST_SIMD_ADD(int16_t, _epi16, Family, Family##_)\
    REGIST_SIMD_ADD(int32_t, _epi32, Family, Family##_)\
    REGIST_SIMD_ADD(uint8_t, s_epu8, Family, Family##_)\
    REGIST_SIMD_ADD(uint16_t, s_epu16, Family, Family##_)\
    \
    REGIST_SIMD_SUB(float, _ps, Family, Family##_)\
    REGIST_SIMD_SUB(double, _pd, Family, Family##_)\
    REGIST_SIMD_SUB(int8_t, _epi8, Family, Family##_)\
    REGIST_SIMD_SUB(int16_t, _epi16, Family, Family##_)\
    REGIST_SIMD_SUB(int32_t, _epi32, Family, Family##_)\
    REGIST_SIMD_SUB(int64_t, _epi64, Family, Family##_)\
    REGIST_SIMD_SUB(uint8_t, s_epu8, Family, Family##_)\
    REGIST_SIMD_SUB(uint16_t, s_epu16, Family, Family##_)\
    \
    REGIST_SIMD_MUL(float, _ps, Family, Family##_)\
    REGIST_SIMD_MUL(double, _pd, Family, Family##_)\
    REGIST_SIMD_MUL(int32_t, _epi32, Family, Family##_)\
    REGIST_SIMD_MUL(uint32_t, _epu32, Family, Family##_)\
    \
    REGIST_SIMD_DIV(float, _ps, Family, Family##_)\
    REGIST_SIMD_DIV(double, _pd, Family, Family##_)

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
    REGIST_SIMD_LOAD(uint32_t, _si##Family, Family, _)\
    \
    REGIST_SIMD_SET(float, _ps, Family, _)\
    REGIST_SIMD_SET(double, _pd, Family, _)\
    REGIST_SIMD_SET(int8_t, _epi8, Family, _)\
    REGIST_SIMD_SET(int16_t, _epi16, Family, _)\
    REGIST_SIMD_SET(int32_t, _epi32, Family, _)\
    \
    REGIST_SIMD_GATHER(float, _ps, Family, _)\
    REGIST_SIMD_GATHER(double, _pd, Family, _)\
    REGIST_SIMD_GATHER(int32_t, _epi32, Family, _)\
    \
    REGIST_SIMD_ADD(float, _ps, Family, _)\
    REGIST_SIMD_ADD(double, _pd, Family, _)\
    REGIST_SIMD_ADD(int8_t, _epi8, Family, _)\
    REGIST_SIMD_ADD(int16_t, _epi16, Family, _)\
    REGIST_SIMD_ADD(int32_t, _epi32, Family, _)\
    REGIST_SIMD_ADD(uint8_t, s_epu8, Family, _)\
    REGIST_SIMD_ADD(uint16_t, s_epu16, Family, _)\
    \
    REGIST_SIMD_SUB(float, _ps, Family, _)\
    REGIST_SIMD_SUB(double, _pd, Family, _)\
    REGIST_SIMD_SUB(int8_t, _epi8, Family, _)\
    REGIST_SIMD_SUB(int16_t, _epi16, Family, _)\
    REGIST_SIMD_SUB(int32_t, _epi32, Family, _)\
    REGIST_SIMD_SUB(int64_t, _epi64, Family, _)\
    REGIST_SIMD_SUB(uint8_t, s_epu8, Family, _)\
    REGIST_SIMD_SUB(uint16_t, s_epu16, Family, _)\
    \
    REGIST_SIMD_MUL(float, _ps, Family, _)\
    REGIST_SIMD_MUL(double, _pd, Family, _)\
    REGIST_SIMD_MUL(int32_t, _epi32, Family, _)\
    REGIST_SIMD_MUL(uint32_t, _epu32, Family, _)\
    \
    REGIST_SIMD_DIV(float, _ps, Family, _)\
    REGIST_SIMD_DIV(double, _pd, Family, _)

#define SIMD_API_NAME (prefix, suffix, api) _mm##prefix##api##suffix

#define REGIST_SIMD_LOAD(T, suffix, Family, family) template<> struct simd_load<T, Family> : public std::true_type{\
    FORCE_INLINE auto operator()(T const * mem_addr){return _mm##family##load##suffix(reinterpret_cast<simd_type_in<T, Family> const*>(mem_addr));}};

#define REGIST_SIMD_SET(T, suffix, Family, family) template<> struct simd_set<T, Family> : public std::true_type{\
    template<class ...TPack>FORCE_INLINE auto operator()(const T arg1, const T arg2, const  TPack... args){return _mm##family##set##suffix(arg1, arg2, args...);}\
    FORCE_INLINE auto operator()(const T arg){return _mm##family##set1##suffix(arg);}\
};

/* be careful : "_mm512_i32gather_?" input args is weird*/
#define REGIST_SIMD_GATHER(T, suffix, Family, family)\
    template<> struct simd_gather<T, Family> : public std::true_type{\
    template<class T1, class T2>FORCE_INLINE auto operator()(T1 arg1, T2 arg2, const int scala){\
        static_assert((std::is_same<typename std::remove_cv<T2>::type, typename std::remove_cv<T*>::type>::value && Family == SIMD_FAMILY_AVX_512) ||\
                            (std::is_same<typename std::remove_cv<T1>::type, typename std::remove_cv<T*>::type>::value && Family < SIMD_FAMILY_AVX_512), "support intel ugly api !");\
        return _mm##family##i32gather##suffix(arg1, arg2, scala);}\
};

#define REGIST_SIMD_ADD(T, suffix, Family, family) template<> struct simd_add<T, Family> : public std::true_type{\
    FORCE_INLINE auto operator()(simd_type_out<T, Family> a, simd_type_out<T, Family> b){return _mm##family##add##suffix(a, b);}};

#define REGIST_SIMD_SUB(T, suffix, Family, family) template<> struct simd_sub<T, Family> : public std::true_type{\
    FORCE_INLINE auto operator()(simd_type_out<T, Family> a, simd_type_out<T, Family> b){return _mm##family##sub##suffix(a, b);}};

#define REGIST_SIMD_MUL(T, suffix, Family, family) template<> struct simd_mul<T, Family> : public std::true_type{\
    FORCE_INLINE auto operator()(simd_type_out<T, Family> a, simd_type_out<T, Family> b){return _mm##family##mul##suffix(a, b);}};

#define REGIST_SIMD_DIV(T, suffix, Family, family) template<> struct simd_div<T, Family> : public std::true_type{\
    FORCE_INLINE auto operator()(simd_type_out<T, Family> a, simd_type_out<T, Family> b){return _mm##family##div##suffix(a, b);}};

namespace simd_cpp
{
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_load : public std::false_type{};
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_set : public std::false_type{};
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_gather : public std::false_type{};
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_scatter : public std::false_type{};

    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_add : public std::false_type{};
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_sub : public std::false_type{};
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_mul : public std::false_type{};
    template<class T, int Family = DEFAULT_SIMD_FAMILY> struct simd_div : public std::false_type{};
}



