#pragma once

#include "layers/Layer.h"

namespace nn
{
    template <typename Dtype = float, int Dims = 2>
    class Relu : public Layer<Dtype, Dims>
    {
    public:
        Relu() = default;

        const std::string &getName()
        {
            const static std::string name = "Relu";
            return name;
        }

        Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input);

        Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &accumulatedGrad);

        void step() {}

        void registerOptimizer(std::shared_ptr<StochasticGradientDescent<Dtype>> optimizer) {}

        void registerOptimizer(std::shared_ptr<Adam<Dtype>> optimizer) {}

    private:
        Eigen::Tensor<Dtype, Dims> m_output;
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Relu<Dtype, Dims>::forward(const Eigen::Tensor<Dtype, Dims> &input)
    {
        m_output = input.cwiseMax(static_cast<Dtype>(0));
        return m_output;
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Relu<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &accumulatedGrad)
    {
        auto inputPositive = m_output > static_cast<Dtype>(0);
        return inputPositive.select(accumulatedGrad, accumulatedGrad.constant(0.0));
    }
}

