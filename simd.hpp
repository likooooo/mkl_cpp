/*
 * @Description: 
 * @Version: 1.0
 * @Autor: like
 * @Date: 2023-05-08 19:05:34
 * @LastEditors: like
 * @LastEditTime: 2023-05-08 20:54:09
 */
#pragma once
#include <immintrin.h>
#include "matrix.hpp"
#include <algorithm>
void test()
{
    double buf[] = {1, 2, 3, 4, 5,6,7,8,9};
    mkl_cpp::matrix2d m(buf, {3, 3});
    
    *reinterpret_cast<__m256d*>(m.p) = _mm256_add_pd(*reinterpret_cast<__m256d*>(m.p), *reinterpret_cast<__m256d*>(m.p));
    for (auto n : m)
    {
        printf("%f ", n);
    }
    vdAdd(4, m.p, m.p, m.p);
    for (auto n : m)
    {
        printf("%f ", n);
    }
}