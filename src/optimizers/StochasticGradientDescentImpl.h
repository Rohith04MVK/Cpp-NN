#ifndef NN_CPP_STOCHASTICGRADIENTDESCENT_IMPL_H
#define NN_CPP_STOCHASTICGRADIENTDESCENT_IMPL_H

#include "OptimizerImpl.h"

namespace nn {
    namespace internal {
        template<typename Dtype, int Dims>
        class StochasticGradientDescentImpl : public OptimizerImpl<Dtype, Dims> {
        public:

            explicit StochasticGradientDescentImpl(Dtype learningRate):
                    m_learningRate(learningRate) {}

            Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &gradWeights) {
                return gradWeights * gradWeights.constant(m_learningRate);
            };

        private:
            Dtype m_learningRate;
        };
    }
}

#endif //NN_CPP_STOCHASTICGRADIENTDESCENT_IMPL_H
