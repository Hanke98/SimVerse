#pragma once

#include "Vector.h"
#include "Array/Array.h"
#include "Field.h"

namespace dyno
{
    template<typename TDataType>
    struct EnvironmentInfos
    {
        using Real = typename TDataType::Real;
        int num_envs = 0;
        int max_constraints = 512;
        DArray<Vec3f> gravities;
        DArray<Real> timesteps;
    };

    template<typename T, size_t Dim>
    class VecXt
    {
    public:
        DYN_FUNC VecXt();
        DYN_FUNC VecXt(const VecXt<T, Dim>&);
        DYN_FUNC ~VecXt() = default;

        DYN_FUNC static int dims() { return Dim; }

        DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;
        
        DYN_FUNC const VecXt<T, Dim> operator+ (const VecXt<T, Dim>&) const;
		DYN_FUNC VecXt<T, Dim>& operator+= (const VecXt<T, Dim>&);
		DYN_FUNC const VecXt<T, Dim> operator- (const VecXt<T, Dim>&) const;
		DYN_FUNC VecXt<T, Dim>& operator-= (const VecXt<T, Dim>&);
		DYN_FUNC const VecXt<T, Dim> operator* (const VecXt<T, Dim>&) const;
		DYN_FUNC VecXt<T, Dim>& operator*= (const VecXt<T, Dim>&);
		DYN_FUNC const VecXt<T, Dim> operator/ (const VecXt<T, Dim>&) const;
		DYN_FUNC VecXt<T, Dim>& operator/= (const VecXt<T, Dim>&);

        DYN_FUNC VecXt<T, Dim>& operator= (const VecXt<T, Dim> &);

		DYN_FUNC const VecXt<T, Dim> operator* (T) const;
		DYN_FUNC const VecXt<T, Dim> operator- (T) const;
		DYN_FUNC const VecXt<T, Dim> operator+ (T) const;
		DYN_FUNC const VecXt<T, Dim> operator/ (T) const;

		DYN_FUNC VecXt<T, Dim>& operator+= (T);
		DYN_FUNC VecXt<T, Dim>& operator-= (T);
		DYN_FUNC VecXt<T, Dim>& operator*= (T);
		DYN_FUNC VecXt<T, Dim>& operator/= (T);

        DYN_FUNC T Norm() const;
        DYN_FUNC T Dot(const VecXt<T, Dim>&) const;

        std::ostream& operator<<(std::ostream& os) const
        {
            os << "Vec<" << Dim << ">: ";
            for (size_t i = 0; i < Dim; i++)
            {
                os << data[i];
                if (i < Dim - 1)
                    os << ", ";
            }
            return os;
        }
    protected:
        T data[Dim];
    };
}

#ifndef DYNO_VECXT_STANDALONE_PARSE
#define DYNO_VECXT_FROM_TYPE_H 1
#include "VecXt.tpp"
#undef DYNO_VECXT_FROM_TYPE_H
#endif