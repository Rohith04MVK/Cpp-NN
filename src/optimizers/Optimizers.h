#pragma once

#include "StochasticGradientDescentImpl.h"
#include "AdamImpl.h"
#include <memory>

namespace nn
{
    template <typename Dtype>
    class StochasticGradientDescent
    {
    public:
        explicit StochasticGradientDescent(Dtype learningRate) : m_learningRate(learningRate) {}

        template <int Dims>
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> createOptimizer() const
        {
            return std::unique_ptr<OptimizerImpl<Dtype, Dims>>(new internal::StochasticGradientDescentImpl<Dtype, Dims>(m_learningRate));
        }

    private:
        Dtype m_learningRate;
    };

    template <typename Dtype>
    class Adam
    {
    public:
        explicit Adam(Dtype learningRate, Dtype beta1 = 0.9, Dtype beta2 = 0.999, Dtype epsilon = 1e-8) : m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
        {
        }

        template <int Dims>
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> createOptimizer() const
        {
            return std::unique_ptr<OptimizerImpl<Dtype, Dims>>(new internal::AdamImpl<Dtype, Dims>(m_learningRate, m_beta1, m_beta2, m_epsilon));
        };

    private:
        Dtype m_learningRate;
        Dtype m_beta1;
        Dtype m_beta2;
        Dtype m_epsilon;
    };

}

