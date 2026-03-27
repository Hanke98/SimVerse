#include <Platform.h>

#ifndef DYNO_VECXT_FROM_TYPE_H
#define DYNO_VECXT_STANDALONE_PARSE 1
#include "type.h"
#undef DYNO_VECXT_STANDALONE_PARSE
#endif

namespace dyno
{   
    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>::VecXt()
    {
        for (size_t i = 0; i < Dim; i++)
            data[i] = T(0);
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>::VecXt(const VecXt<T, Dim>& other)
    {
        for (size_t i = 0; i < Dim; i++)
            data[i] = other.data[i];
    }

    template<typename T, size_t Dim>
    DYN_FUNC T& VecXt<T, Dim>::operator[](unsigned int index)
    {
        return data[index];
    }

    template<typename T, size_t Dim>
    DYN_FUNC const T& VecXt<T, Dim>::operator[](unsigned int index) const
    {
        return data[index];
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator+(const VecXt<T, Dim>& other) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] + other.data[i];
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator+=(const VecXt<T, Dim>& other)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] += other.data[i];
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator-(const VecXt<T, Dim>& other) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)            
            result.data[i] = this->data[i] - other.data[i];
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator-=(const VecXt<T, Dim>& other)
    {        
        for (size_t i = 0; i < Dim; i++)
            this->data[i] -= other.data[i];
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator*(const VecXt<T, Dim>& other) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] * other.data[i];
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator*=(const VecXt<T, Dim>& other)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] *= other.data[i];
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator/(const VecXt<T, Dim>& other) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] / other.data[i];
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator/=(const VecXt<T, Dim>& other)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] /= other.data[i];
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator=(const VecXt<T, Dim>& other)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] = other.data[i];
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator*(T scalar) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] * scalar;
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator-(T scalar) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] - scalar;
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator+(T scalar) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] + scalar;
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC const VecXt<T, Dim> VecXt<T, Dim>::operator/(T scalar) const
    {
        VecXt<T, Dim> result;
        for (size_t i = 0; i < Dim; i++)
            result.data[i] = this->data[i] / scalar;
        return result;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator+=(T scalar)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] += scalar;
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator-=(T scalar)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] -= scalar;
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator*=(T scalar)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] *= scalar;
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC VecXt<T, Dim>& VecXt<T, Dim>::operator/=(T scalar)
    {
        for (size_t i = 0; i < Dim; i++)
            this->data[i] /= scalar;
        return *this;
    }

    template<typename T, size_t Dim>
    DYN_FUNC T VecXt<T, Dim>::Norm() const
    {
        T sum = T(0);
        for (size_t i = 0; i < Dim; i++)
            sum += data[i] * data[i];
        return sqrt(sum);
    }

    template<typename T, size_t Dim>
    DYN_FUNC T VecXt<T, Dim>::Dot(const VecXt<T, Dim>& other) const
    {
        T sum = T(0);
        for (size_t i = 0; i < Dim; i++)
            sum += this->data[i] * other.data[i];
        return sum;
    }

}