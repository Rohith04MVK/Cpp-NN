#ifndef NN_CPP_OPTIMIZERIMPL_H
#define NN_CPP_OPTIMIZERIMPL_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace nn {
    template <typename Dtype, int Dims>
    class OptimizerImpl {
    public:
        virtual Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &weights) = 0;
    };
}

#endif //NN_CPP_OPTIMIZERIMPL_H
