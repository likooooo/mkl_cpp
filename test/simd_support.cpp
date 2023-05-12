#include "../simd.hpp"

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

    template<class test_type,  int Family>
    struct simd_test_meta_info
    {
        simd_test_meta_info()
        {
            printf("begin simd test\n    simd family : %d\n    value type : %s\n    value len : %d\n", Family, type_name<test_type>::str,  simd_alloc<test_type, Family>::simd_data_align);
            
        }
        ~simd_test_meta_info()
        {
            printf("\nend simd test ()\n------------------------------------\n\n");
        }
    };
}

template<class test_type = double,  int Family = DEFAULT_SIMD_FAMILY>
auto simd_load_test()
{
    using namespace simd_cpp;
    constexpr int buf_len = simd_alloc<test_type, Family>::simd_data_align;
    static_assert(is_simd_support<simd_load<test_type, Family>>, "unsupport simd api");
    test_type* buf = simd_alloc<test_type, Family>::allocate(buf_len);
    assert(buf);
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
    auto print_result = [&m](){
        const int buf_size = m.buffer_size();
        for(int i = 0; i < std::min(16, buf_size); i++)
        {
            std::cout << +m.p[i] << " ";
        }
        if(buf_size > 16) std::cout << "...";
        std::cout <<  std::endl;
    };

    simd_type_out<test_type, Family> a = simd_load<test_type, Family>()(m.p);
    SIMD_TEST_MSG(simd_add)
    {
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_add<test_type, Family>()(a, a);

        print_result();
    }
    SIMD_TEST_MSG(simd_sub)
    {
        simd_type_out<test_type, Family> b = simd_load<test_type, Family>()(m.p);
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_sub<test_type, Family>()(b, a);

        print_result();
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

        print_result();
    }
    SIMD_TEST_MSG(simd_div)
    {
        simd_type_out<test_type, Family> a = simd_load<test_type, Family>()(m.p);
        test_type div[buf_len];
        std::for_each(div, div + buf_len,[](test_type& d){d = static_cast<test_type>(2);});
        simd_type_out<test_type, Family> b = simd_load<test_type, Family>()(div);
        *reinterpret_cast<simd_type_out<test_type, Family>*>(m.p) = simd_div<test_type, Family>()(a, b);

        print_result();
    }
}

int main()
{
    /*simd_test<int8_t, SIMD_FAMILY_AVX_512>();
 *     simd_test<int16_t, SIMD_FAMILY_AVX_512>();
 *         simd_test<int32_t, SIMD_FAMILY_AVX_512>();
 *             simd_test<int64_t, SIMD_FAMILY_AVX_512>();
 *                 simd_test<uint8_t, SIMD_FAMILY_AVX_512>();
 *                     simd_test<uint16_t, SIMD_FAMILY_AVX_512>();
 *                         simd_test<uint32_t, SIMD_FAMILY_AVX_512>();
 *                             simd_test<float, SIMD_FAMILY_AVX_512>();
 *                                 simd_test<double, SIMD_FAMILY_AVX_512>();*/
    printf("##########################\n    * AVX_512 -> AVX\n##########################\n\n");
    simd_test<int8_t, SIMD_FAMILY_AVX>();
    simd_test<int16_t, SIMD_FAMILY_AVX>();
    simd_test<int32_t, SIMD_FAMILY_AVX>();
    simd_test<int64_t, SIMD_FAMILY_AVX>();
    simd_test<uint8_t, SIMD_FAMILY_AVX>();
    simd_test<uint16_t, SIMD_FAMILY_AVX>();
    simd_test<uint32_t, SIMD_FAMILY_AVX>();
    simd_test<float, SIMD_FAMILY_AVX>();
    simd_test<double, SIMD_FAMILY_AVX>();
    printf("##########################\n    * AVX -> SSE\n##########################\n\n");
    simd_test<int8_t, SIMD_FAMILY_SSE>();
    simd_test<int16_t, SIMD_FAMILY_SSE>();
    simd_test<int32_t, SIMD_FAMILY_SSE>();
    simd_test<int64_t, SIMD_FAMILY_SSE>();
    simd_test<uint8_t, SIMD_FAMILY_SSE>();
    simd_test<uint16_t, SIMD_FAMILY_SSE>();
    simd_test<uint32_t, SIMD_FAMILY_SSE>();
    simd_test<float, SIMD_FAMILY_SSE>();
    simd_test<double, SIMD_FAMILY_SSE>();
}
 
