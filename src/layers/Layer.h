#ifndef NN_CPP_LAYER_H
#define NN_CPP_LAYER_H

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "optimizers/Optimizers.h"

namespace nn
{
    template <typename Dtype = float, int Dims = 2>
    class Layer
    {
    public:
        virtual const std::string &getName() = 0;

        virtual Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input) = 0;

        virtual Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &output) = 0;

        virtual void step() = 0;

        virtual void registerOptimizer(std::shared_ptr<StochasticGradientDescent<Dtype>> optimizer) = 0;

        virtual void registerOptimizer(std::shared_ptr<Adam<Dtype>> optimizer) = 0;
    };
}

#endif //NN_CPP_LAYER_H
