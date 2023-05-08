#pragma once
#include <array>
#include <vector>
#include <type_traits>
#include <numeric>
#include <assert.h>

#define CONSTEXPR constexpr

#define DECLARE_MATRIX(T, Suffix, n)                                                                \
    static_assert(n <= mkl_cpp::details::__dim_count, "unsupport matrix declare"); \
    using matrix##n##Suffix = matrix<T, n>
#define DECLARE_MATRIX_ALL_SIZE(T, Suffix)  \
    DECLARE_MATRIX(T, Suffix, 1);\
    DECLARE_MATRIX(T, Suffix, 2);\
    DECLARE_MATRIX(T, Suffix, 3);

namespace mkl_cpp
{
    namespace details
    {
        /**
 *          * @brief if "DimCount <= __dim_count" more constexpr functioin can be used 
 *                   * 
 *                            */
        constexpr int __dim_count = sizeof(std::vector<int>) / sizeof(int); 
        /**
 *          * @brief matrix memory layout
 *                   * 
 *                            */
        template <class T, int DimCount, class SFINAE = void>
        struct __matrix;
        template <class T, int DimCount >
        struct __matrix<T, DimCount,  std::enable_if_t<(__dim_count >= DimCount)>>
        {
            using DimContainer = std::array<int, DimCount>; 

            T* p{nullptr};
            const DimContainer dims{0};
        };
        template <class T, int DimCount >
        struct __matrix<T, DimCount,  std::enable_if_t<(__dim_count < DimCount)>>
        {
            using DimContainer = std::vector<int>;

            T* p{nullptr};
            const DimContainer dims{std::vector<int>(DimCount)};
        };
    } 

    template <class T, int DimCount>
    struct matrix : public details::__matrix<T, DimCount>
    {
        static constexpr  int dim_count = DimCount;    
        using base_type = details::__matrix<T, DimCount>;
        using DimContainer = typename base_type::DimContainer; 
        using value_type = T;
        using iterator = value_type*;
        using const_iterator = const value_type*;

        constexpr matrix() = default;
        constexpr matrix(value_type* _p, const DimContainer& _dims) : base_type({_p, _dims}) {}
        constexpr matrix(const matrix&) = default;
        constexpr matrix(matrix&&) = default;

        CONSTEXPR iterator begin()
        {
            return this->p;
        }
        CONSTEXPR iterator end()
        {
            return this->p + buffer_size();
        }
        CONSTEXPR const_iterator begin()const
        {
            return this->p;
        }
        CONSTEXPR const_iterator end()const
        {
            return this->p + buffer_size();
        }
        CONSTEXPR int buffer_size() const
        {
            return std::accumulate(this->dims.begin(), this->dims.end(), 1, [](const auto result, const auto dim){return result * dim;});
        } 
        CONSTEXPR auto slice(const int index_of_last_dim)const
        {
            assert(index_of_last_dim < this->dims.back());
            if constexpr (dim_count > 1)
            {
                std::array<int , dim_count - 1> slice_dims;
                std::copy(this->dims.begin(), this->dims.end() - 1, slice_dims.begin());
                matrix<T, dim_count - 1> slice(nullptr , slice_dims);
                slice.p = this->p + (index_of_last_dim * slice.buffer_size());
                return slice;
            }
            else
            {
                value_type& ref = this->p[index_of_last_dim]; 
                return ref;
            }
        }
        template<int _DimCount, class _T = value_type>
        CONSTEXPR matrix<_T, _DimCount> reshape(const typename matrix<_T, _DimCount>::DimContainer & _dims)
        {
            matrix<_T, _DimCount> m(nullptr, _dims);
            assert(m.buffer_size() <= this->buffer_size());

            if constexpr (std::is_same<T, _T>::value)
                m.p = this->p;
            else
                m.p = reinterpret_cast<_T*>(this->p);
            return m;
        }
    };

    DECLARE_MATRIX_ALL_SIZE(int, n);
    DECLARE_MATRIX_ALL_SIZE(double, d);
}
