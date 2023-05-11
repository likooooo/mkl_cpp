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
#include <memory>

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
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_out = typename simd_traits<T, simd_family>::simd_type;
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_in = typename simd_traits<T, simd_family>::simd_type_in;
  
    /* SIMD api */
    REGIST_SIMD_SSE(SIMD_FAMILY_SSE);
    REGIST_SIMD_AVX(SIMD_FAMILY_AVX);
    /* REGIST_SIMD_AVX(SIMD_FAMILY_AVX_512); */
    template<class simd_api>
    constexpr bool is_simd_support = simd_api::value;
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
#include <memory>

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
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_out = typename simd_traits<T, simd_family>::simd_type;
    template<class T, int simd_family = DEFAULT_SIMD_FAMILY>
    using simd_type_in = typename simd_traits<T, simd_family>::simd_type_in;
  
    /* SIMD api */
    REGIST_SIMD_SSE(SIMD_FAMILY_SSE);
    REGIST_SIMD_AVX(SIMD_FAMILY_AVX);
    /* REGIST_SIMD_AVX(SIMD_FAMILY_AVX_512); */
    template<class simd_api>
    constexpr bool is_simd_support = simd_api::value;

    /* make index */
    template<int index_count>
    FORCE_INLINE auto make_index(const std::array<int32_t, index_count> indexs)
    {
        static_assert(4 == index_count || 8 == index_count);
        if constexpr (4 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_SSE>()(indexs[3], indexs[2], indexs[1], indexs[0]);
        }
        else  if constexpr (8 == index_count)
        {
            return simd_set<int32_t, SIMD_FAMILY_AVX>()(indexs[7], indexs[6], indexs[5], indexs[4], indexs[3], indexs[2], indexs[1], indexs[0]);
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


#define SIMD_TEST_MSG(simd_api)\
    constexpr bool support_##simd_api = is_simd_support<simd_api<test_type, Family>>;\
    if constexpr ( support_##simd_api )     printf("    * " #simd_api " (\033[1;32;40m support \033[0m)\n");\
    else     printf("    * " #simd_api " (\033[1;31;40m unsupport \033[0m)\n");\
    if constexpr( support_##simd_api )

namespace simd_cpp::details
{
    template<class T>
    struct type_name;
    template<>struct type_name<int8_t>{constexpr static const char* str = "int8";};
    template<>struct type_name<int16_t>{constexpr static const char* str = "int16";};
    template<>struct type_name<int32_t>{constexpr static const char* str = "int32";};
    template<>struct type_name<int64_t>{constexpr static const char* str = "int64";};
    template<>struct type_name<uint8_t>{constexpr static const char* str = "uint8";};
    template<>struct type_name<uint16_t>{constexpr static const char* str = "uint16";};
    template<>struct type_name<uint32_t>{constexpr static const char* str = "uint32";};
    template<>struct type_name<float>{constexpr static const char* str = "float";};
    template<>struct type_name<double>{constexpr static const char* str = "double";};

    template<class test_type = double,  int Family = DEFAULT_SIMD_FAMILY>
    struct simd_test_meta_info
    {
        simd_test_meta_info()
        {
            printf("begin simd test\n    simd family : %d\n    value type : %s\n    value len : %d\n", Family, type_name<test_type>::str,  simd_alloc<test_type, Family>::simd_data_align);
        }
        ~simd_test_meta_info(){printf("\nend simd test\n------------------------------------\n\n");}
    };
}


template<class test_type = double,  int Family = DEFAULT_SIMD_FAMILY>
auto simd_load_test()
{
    using namespace simd_cpp;
    constexpr int buf_len = simd_alloc<test_type, Family>::simd_data_align;
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
    SIMD_TEST_MSG(simd_set)
    {
        {
            auto vindex = simd_set<test_type, Family>()((test_type)1);
            test_type index[simd_alloc<test_type, Family>::simd_data_align];
            std::for_each(std::begin(index), std::end(index), [](test_type& d){ d = static_cast<test_type>(1);});
            assert(0 == std::memcmp(&vindex, index, sizeof(index)));
        }
        {
            constexpr int len = simd_alloc<int, Family>::simd_data_align;
            std::array<int, len> index;
            for(int i = 0; i < len; i++)
            {
                index[i] = i;
            }
            if constexpr (16 > len )
            {
                auto vindex = make_index<len>(index);
                assert(0 == std::memcmp(&vindex, index.data(), sizeof(index)));
            }
        }
    }

    SIMD_TEST_MSG(simd_gather)
    {
        constexpr int len =  (simd_alloc<test_type, Family>::simd_data_align + 3) / 4 * 4;
        std::array<int, len> index;
        for(int i = 0; i < len; i++)
        {
            index[i] = i;
        }
        auto vindex = make_index<len>(index);
        simd_type_out<test_type, Family> a = simd_gather<test_type, Family>()(
            buf, 
            vindex, sizeof(test_type)
        );
        assert(0 == std::memcmp(&a, buf, sizeof(a)));
    }
    return std::tuple{std::unique_ptr<test_type, decltype([](test_type* p){simd_alloc<test_type, Family>::deallocate(p);})>(buf), buf_len};
}

template<class test_type = double,  int Family = DEFAULT_SIMD_FAMILY>
void simd_test()
{
    using namespace simd_cpp;
    details::simd_test_meta_info<test_type, Family> msg;

    auto [buf, buf_len] = simd_load_test<test_type, Family>();
    mkl_cpp::matrix<test_type, 2> m(buf.get(), {buf_len, 1});
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
        simd_type_out<test_type, Family> a;
        if constexpr (std::is_integral<test_type>::value) /* prepare overflow */
        {
            constexpr int len = simd_alloc<test_type, Family>::simd_data_align;
            std::array<int, len> index = {0};
            for(int i = 2; i < len; i += 2) /* 1,2,3,4 -> 1,padding, 2, padding */
            {
                index[i] = i / 2;
            }
            static_assert(sizeof(int) == sizeof(test_type));
            auto vindex = make_index<len>(index);
            a = simd_gather<int, Family>()((int*)m.p, vindex/* index of pointer */, sizeof(test_type) /* byte offset */);
        }
        else
        {
            a = simd_load<test_type, Family>()(m.p);
        }
        const auto result = simd_mul<test_type, Family>()(a, a);
        if constexpr (std::is_integral<test_type>::value)
        {
            /* TODO : handle overflow */
            *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = result ;
        }
        else
        {
            *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = result ;
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
}

