#pragma once

#include "OptimizerImpl.h"

namespace nn
{
    namespace internal
    {
        template <typename Dtype, int Dims>
        class AdamImpl : public OptimizerImpl<Dtype, Dims>
        {
        public:
            explicit AdamImpl(Dtype learningRate, Dtype beta1, Dtype beta2, Dtype epsilon) : m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon),
                                                                                             m_isInitialized(false), m_currentTimestep(1)
            {
            }

            Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &gradWeights)
            {
                if (!m_isInitialized)
                {
                    m_firstMoment = Eigen::Tensor<Dtype, Dims>(gradWeights.dimensions());
                    m_firstMoment.setZero();

                    m_secondMoment = Eigen::Tensor<Dtype, Dims>(gradWeights.dimensions());
                    m_secondMoment.setZero();
                    m_isInitialized = true;
                }

                m_firstMoment = m_firstMoment.constant(m_beta1) * m_firstMoment +
                                gradWeights.constant(1 - m_beta1) * gradWeights;

                m_secondMoment = m_secondMoment.constant(m_beta2) * m_secondMoment +
                                 gradWeights.constant(1 - m_beta2) * gradWeights.square();

                auto biasCorrectedFirstMoment = m_firstMoment / m_firstMoment.constant(1 - pow(m_beta1, m_currentTimestep));
                auto biasCorrectedSecondMoment = m_secondMoment / m_secondMoment.constant(1 - pow(m_beta2, m_currentTimestep));

                m_currentTimestep++;
                return biasCorrectedFirstMoment * ((gradWeights.constant(m_learningRate) /
                                                    (biasCorrectedSecondMoment.sqrt() + gradWeights.constant(m_epsilon))));
            };

        private:
            Dtype m_learningRate;
            Dtype m_beta1;
            Dtype m_beta2;
            Dtype m_epsilon;

            bool m_isInitialized;
            size_t m_currentTimestep;

            Eigen::Tensor<Dtype, Dims> m_firstMoment;
            Eigen::Tensor<Dtype, Dims> m_secondMoment;
        };
    }
}

