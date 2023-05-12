#include "mkl.hpp"
#include "simd.hpp"
#include <memory>
#include <algorithm>
#include <tuple>
#include <iostream>
using namespace mkl_cpp;


struct time_with_tsh
{
    uint64_t operator()(){return __rdtsc();}
};
struct time_with_chrono
{
    uint64_t operator()()
    {
        using namespace std::chrono;
        return std::chrono::duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }
};
template<class recorder = time_with_tsh>
struct recorder_span
{
    recorder_span() : nanosecond( recorder{}()){}
    ~recorder_span() 
    {
        nanosecond = recorder{}() - nanosecond; 
        printf("record time span : %zd ( ns )\n", nanosecond);
    }
    uint64_t nanosecond;
};

template<int simd_family>
void simd_time_test(int size)
{
    using namespace simd_cpp;
    using namespace simd_cpp::details;

    int step = 0;

    auto ptr1 = simd_alloc<double, simd_family>::allocate_unique(size);
    double* p1 = ptr1.get();
    {
        std::random_device rd; 
        std::mt19937 gen(rd()); 

        constexpr int var_limit = 1000;
        constexpr double var_scala = 0.1 / var_limit;
        std::uniform_int_distribution<> distrib( -var_limit, var_limit);
        for(int i = 0; i < size; i++)
        {
            p1[i] =  var_scala * distrib(gen) + distrib(gen);
        }
    }
    auto ptr2 = simd_alloc<double, simd_family>::allocate_unique(size);
    double* p2 = ptr2.get();
    {
        std::random_device rd; 
        std::mt19937 gen(rd()); 

        constexpr int var_limit = 1000;
        constexpr double var_scala = 0.1 / var_limit;
        std::uniform_int_distribution<> distrib( -var_limit, var_limit);
        for(int i = 0; i < size; i++)
        {
            p2[i] =  var_scala * distrib(gen) + distrib(gen);
        }
    }

    {
        recorder_span<time_with_tsh> _;
        for(size_t i = 0; i < size; i++)
        {
            p1[i] *= p1[i];
        }
    }
    {
        recorder_span<time_with_tsh> _;
        constexpr int simd_flow_aligin = simd_alloc<double, simd_family>::simd_data_align;
        for(size_t i = 0; i < size ; i+= simd_flow_aligin)
        {
            auto a = simd_load<double, simd_family>()(p2 + i);
            *reinterpret_cast<simd_type_out<double, simd_family>*>(p2 + i) = simd_mul<double, simd_family>()(a, a);
        }
    }

    for(size_t i = 0; i < size; i++)
    {
        if(p1[i] != p2[i])
        {
            printf("test error at %d, %lf, %lf\n", i, p1[i], p2[i]);
            break;
        }
        assert(p1[i] == p2[i]);
    }
}

int main()
{
    simd_time_test<SIMD_FAMILY_AVX>(1024);
}

