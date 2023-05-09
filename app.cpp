#include "mkl.hpp"
#include "simd.hpp"
#include <memory>
#include <algorithm>
#include <tuple>
#include <iostream>
using namespace mkl_cpp;

template<class T>
constexpr auto get_dims()
{
    const int n = T::dim_count;

    typename T::DimContainer dims;
    for(int i = 0; i < n; i++)dims[i] = n - i + 1;
    return dims;
}
template<class T>
constexpr auto get_buffer()
{
    constexpr auto dims = get_dims<T>();
    constexpr int buffer_size = std::accumulate(dims.begin(), dims.end(), 1, [](const auto temp, const auto dim){return temp * dim;});
    std::array<typename T::value_type, buffer_size> buf;
    std::for_each(buf.begin(), buf.end(),
        [i = 0](auto& n) mutable{
            n = i;
            i++;
        }
    );  
    return buf;
}
template<class T>
struct context_constexpr_construct_test1
{
    constexpr static auto dims = get_dims<T>();
    constexpr static auto buf = get_buffer<T>();
    constexpr static auto m = matrix<const typename T::value_type, T::dim_count>(buf.data(), dims);

    static_assert(std::equal(m.begin(), m.end(), buf.begin()));
};
template<class T>
void constexpr_construct_test1()
{
    using ctx = context_constexpr_construct_test1<T>;
    {
        constexpr T m;
        static_assert(nullptr == m.begin());
        static_assert(nullptr == m.end());
        static_assert(0 == std::accumulate(m.begin(), m.end(), 0.0));
        static_assert(0 == m.buffer_size());
    }
     {
        printf("dims : \n    ");
        std::for_each(ctx::dims.begin(), ctx::dims.end(), [](const int dim){printf("%d ", dim);});
        printf("\n");
        printf("buf : \n    ");
        std::for_each(ctx::buf.begin(), ctx::buf.end(), [](const auto n){std::cout << n << " ";});
        printf("\n");
        printf("matrix : \n    ");
        std::for_each(ctx::m.begin(), ctx::m.end(), [](const auto n){std::cout << n << " ";});
        printf("\n");
    }
    if constexpr( 1 < T::dim_count )
    {
        constexpr auto slice_of_last_dim =  ctx::m.slice( ctx::m.dims.back() - 1);
        printf("slice_of_last_dim : \n    ");
        std::for_each(slice_of_last_dim.begin(), slice_of_last_dim.end(), [](const auto n){std::cout << n << " ";});
        printf("\n");
    }

    printf("end constexpr_construct_test1...\n\n\n");
}
int main()
{
    simd_test<int8_t>();
    simd_test<int16_t>();
    simd_test<int32_t>();
    simd_test<int64_t>();
    simd_test<uint8_t>();
    simd_test<uint16_t>();
    simd_test<float>();
    simd_test<double>();
    constexpr_construct_test1<matrix<int, 3>>();
    constexpr_construct_test1<matrix1d>();
    constexpr_construct_test1<matrix2d>();
    constexpr_construct_test1<matrix3d>();
    return 0;
}

