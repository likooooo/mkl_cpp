#pragma once
#include "simd_marco.hpp"
#include <immintrin.h>
#include <x86intrin.h>
#include <chrono>
#include "matrix.hpp"
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <type_traits>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <random>
#include <assert.h>

namespace simd_cpp
{
    /* no use for user */
    namespace details
    {
        template<int simd_data_align>
        consteval auto get_mask_type()
        {
            static_assert( 64 >= simd_data_align, "SIMD unsupport align size" );

            if constexpr(8 >= simd_data_align) return __mmask8();
            else if constexpr(16 >= simd_data_align) return __mmask16();
            else if constexpr(32 >= simd_data_align) return __mmask32();
            else return __mmask64();
        }
        template<class index_type, int simd_data_align>
        consteval auto get_index_type()
        {
            constexpr int index_bit_width = sizeof(index_type) * 8;
            static_assert( 512 / index_bit_width >= simd_data_align, "SIMD unsupport align size" );

            if constexpr((128 / index_bit_width)  >= simd_data_align) return __m128i();
            else if constexpr(256 / index_bit_width >= simd_data_align) return __m256i();
            else return __m512i();
        }
    }
  
    /* allocator for SIMD memory align */
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    struct simd_alloc
    {
        constexpr  static int simd_byte_align = static_cast<int>(simd_family) / 8;
        static_assert( 0 < simd_family && 0 == (simd_byte_align % sizeof(T)),  "unsupport simd family" );
        constexpr static int simd_data_align = simd_byte_align / sizeof(T); 

        static void deallocate(T* p)
        {
            _mm_free(p);
        }

        using unique_ptr_t =  std::unique_ptr<T, decltype([](T* p){deallocate(p);})>;

        static T* allocate(const size_t n)
        {
            return  (T*)_mm_malloc(sizeof(T) * element_count_with_padding(n), simd_byte_align);
        }
        static unique_ptr_t allocate_unique(const size_t n)
        {
            return unique_ptr_t(allocate((n)));
        }
    private: 
        static size_t element_count_with_padding(size_t n){return (n + simd_data_align - 1)  / simd_data_align * simd_data_align;}
    };
    /* mask type */
    template<class T, int family = DEFAULT_SIMD_FAMILY>
    using simd_mask_type = decltype(details::get_mask_type< simd_alloc<T, family>::simd_data_align >());
    /* index type */
    template<class index_type, class T, int family = DEFAULT_SIMD_FAMILY>
    using simd_index_type = decltype(details::get_index_type< index_type, simd_alloc<T, family>::simd_data_align >());
  
    /* input & output type */
    template<class T, int family = DEFAULT_SIMD_FAMILY> struct simd_traits;
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_SSE);
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_AVX);
    DECLARE_SIMD_TYPE_TRAITS(SIMD_FAMILY_AVX_512);
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_out = typename simd_traits<T, simd_family>::simd_type;
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_in = typename simd_traits<T, simd_family>::simd_type_in;
  
    /* SIMD api */
    REGIST_SIMD_SSE(SIMD_FAMILY_SSE);
    REGIST_SIMD_AVX(SIMD_FAMILY_AVX);
    REGIST_SIMD_AVX(SIMD_FAMILY_AVX_512);
    template<class simd_api>
    constexpr bool is_simd_support = simd_api::value;

    /* make index */
    template<int index_count>
    FORCE_INLINE auto make_index(const std::array<int32_t, index_count> indexs)
    {
        static_assert(4 == index_count || 8 == index_count || 16 == index_count);
        if constexpr (4 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_SSE>()(indexs[3], indexs[2], indexs[1], indexs[0]);
        }
        else  if constexpr (8 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_AVX>()(indexs[7], indexs[6], indexs[5], indexs[4], indexs[3], indexs[2], indexs[1], indexs[0]);
        }
        else  if constexpr (16 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_AVX_512>()(indexs[15], indexs[14], indexs[13], indexs[12], indexs[11], indexs[10], indexs[9], indexs[8], indexs[7], indexs[6], indexs[5], indexs[4], indexs[3], indexs[2], indexs[1], indexs[0]);
        }
    }
    template<int index_count>
    FORCE_INLINE auto make_index(const int32_t n)
    {
        static_assert(4 == index_count || 8 == index_count );
        if constexpr (4 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_SSE>()(n);
        }
        else  if constexpr (8 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_AVX>()(n);
        }
    }
}

