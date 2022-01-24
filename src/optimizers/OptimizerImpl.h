#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace nn
{
    template <typename Dtype, int Dims>
    class OptimizerImpl
    {
    public:
        virtual Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &weights) = 0;
    };
}

