/*
 *  *  * @Description:  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *   *   * @Version: 1.0
 *    *    * @Autor: like
 *     *     * @Date: 2023-05-08 19:05:34
 *      *      * @LastEditors: like
 *       *       * @LastEditTime: 2023-05-08 20:54:09
 *        *        */
#pragma once
#include "simd_marco.hpp"
#include <immintrin.h>
#include "matrix.hpp"
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <type_traits>
#include <cstring>
#include <cstdlib>

namespace simd_cpp
{
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

        static T* allocate(const size_t n){return  (T*)_mm_malloc(simd_byte_align, sizeof(T) * element_count_with_padding(n));}
        static void deallocate(T* p){_mm_free(p);}
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
    template<class simd_api>
    constexpr bool is_simd_support = simd_api::value;
}


#define SIMD_TEST_MSG(simd_api)\
    constexpr bool support_##simd_api = is_simd_support<simd_api<test_type, Family>>;\
    printf("    * " #simd_api " ( %s )\n", support_##simd_api?"support" : "unsupport");\
    if constexpr( support_##simd_api )

template<class test_type = double,  int Family = DEFAULT_SIMD_FAMILY>
auto simd_load_test()
{
    using namespace simd_cpp;
    constexpr int buf_len = simd_alloc<test_type, Family>::simd_data_align;// - 1;
    static_assert(is_simd_support<simd_load<test_type, Family>>, "unsupport simd api");
    
    test_type* buf = simd_alloc<test_type, Family>::allocate(buf_len);
    for (int i = 0; i < buf_len; i++)
    {
        buf[i] = static_cast<test_type>(i + 1);
    }
    SIMD_TEST_MSG(simd_load)
    {
        simd_type_out<test_type, Family> a = simd_load<test_type, Family>()(buf);
        assert(0 == std::memcmp(&a, buf, sizeof(test_type) * buf_len));
    }

    SIMD_TEST_MSG(simd_gather)
    {
        int index_buf[simd_alloc<test_type, Family>::simd_data_align];
        for(int i = 0; i < std::size(index_buf); i++)
        {
            index_buf[i] = i;
        }
        simd_index_type<int, test_type, Family> vindex = {0};
        std::memcpy(&vindex, index_buf, sizeof(vindex));
        auto gatherd_val = simd_gather<test_type, Family>()(buf, vindex, 1);
        assert(0 == std::memcmp(&gatherd_val, buf, sizeof(test_type) * buf_len));
    }

    return std::tuple{buf, buf_len};
}

template<class test_type = double,  int Family = DEFAULT_SIMD_FAMILY>
void simd_test()
{
    using namespace simd_cpp;
    printf("begin simd test\n    simd family : %d\n    value type : %s\n    value len : %d\n", Family, typeid(test_type).name(),  simd_alloc<test_type, Family>::simd_data_align);
    
    auto [buf, buf_len] = simd_load_test<test_type, Family>();
    mkl_cpp::matrix<test_type, 2> m(buf, {buf_len, 1});
    simd_type_out<test_type, Family> a = simd_load<test_type, Family>()(m.p);

    auto print_vector = [&m](){for (auto n : m) std::cout << static_cast<double>(n) << " ";std::cout <<"\n";};
    SIMD_TEST_MSG(simd_add)
    {
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_add<test_type, Family>()(a, a);

        print_vector();
    }
    SIMD_TEST_MSG(simd_sub)
    {
        simd_type_out<test_type, Family> b = simd_load<test_type, Family>()(m.p);
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_sub<test_type, Family>()(b, a);

        print_vector();
    }
    SIMD_TEST_MSG(simd_mul)
    {
        if constexpr (std::is_integral<test_type>::value && 4 == sizeof(test_type))
        {
            const int n = buf_len / 2;
            int64_t temp[n] = {0};
            for(int i = 0; i < n; i++)
            {
                temp[i] = m.p[i];
            }
            memcpy(m.p, temp, m.buffer_size() * sizeof(test_type));
        }
        simd_type_out<test_type, Family> a = simd_load<test_type, Family>()(m.p);
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_mul<test_type, Family>()(a, a);
        if constexpr (std::is_integral<test_type>::value && 4 == sizeof(test_type))
        {
            const int n = buf_len / 2;
            int64_t temp[n] = {0};
            memcpy(temp, m.p, m.buffer_size() * sizeof(test_type));
            for(int i = 0; i < n; i++)
            {
                m.p[i] = temp[i];
            }
            for(int i = n; i < buf_len; i++)
            {
                m.p[i] = 0;
            }
        }

        print_vector();
    }
    SIMD_TEST_MSG(simd_div)
    {
        simd_type_out<test_type, Family> a = simd_load<test_type, Family>()(m.p);
        test_type div[buf_len];
        std::for_each(div, div + buf_len,[](test_type& d){d = static_cast<test_type>(2);});
        simd_type_out<test_type, Family> b = simd_load<test_type, Family>()(div);
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_div<test_type, Family>()(a, b);

        print_vector();
    }

    printf("\nend simd test\n\n");

}

