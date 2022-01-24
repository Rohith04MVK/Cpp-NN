#pragma once

#include "layers/Layers.h"
#include "loss/Losses.h"
#include "optimizers/Optimizers.h"

#include <vector>
#include <memory>

namespace nn
{

    template <typename Dtype = float>
    class Net
    {
    public:
        Net() = default;

        template <int inputDim, int outputDim>
        Eigen::Tensor<Dtype, outputDim> forward(Eigen::Tensor<Dtype, inputDim> input)
        {
            if (m_layers.empty())
            {
                std::cerr << "No layers specified" << std::endl;
                return {};
            }

            auto currentInput = input;
            for (const auto &layer : m_layers)
            {
                currentInput = layer->forward(currentInput);
            }
            return currentInput;
        }

        template <int labelDims>
        void backward(Eigen::Tensor<Dtype, labelDims> input)
        {
            if (!m_hasOptimizer)
            {
                std::cerr << "No registered optimizer" << std::endl;
                return;
            }

            if (m_layers.empty())
            {
                std::cerr << "No layers specified" << std::endl;
                return;
            }

            auto accumulatedGrad = input;
            for (auto rit = m_layers.rbegin(); rit != m_layers.rend(); ++rit)
            {
                accumulatedGrad = (*rit)->backward(accumulatedGrad);
            }
        }

        void registerOptimizer(nn::StochasticGradientDescent<Dtype> *optimizer)
        {
            m_hasOptimizer = true;
            std::shared_ptr<nn::StochasticGradientDescent<Dtype>> optimizerPtr(optimizer);
            for (auto &layer : m_layers)
            {
                layer->registerOptimizer(optimizerPtr);
            }
        }

        void registerOptimizer(nn::Adam<Dtype> *optimizer)
        {
            m_hasOptimizer = true;
            std::shared_ptr<nn::Adam<Dtype>> optimizerPtr(optimizer);
            for (auto &layer : m_layers)
            {
                layer->registerOptimizer(optimizerPtr);
            }
        }

        void step()
        {
            for (auto &layer : m_layers)
            {
                layer->step();
            }
        }

        template <int Dims>
        Net<Dtype> &add(std::unique_ptr<Layer<Dtype, Dims>> layer)
        {
            m_layers.push_back(layer);
            return *this;
        }

        template <int Dims>
        Net<Dtype> &add(Dense<Dtype, Dims> *denseLayer)
        {
            // Do shape checks here
            m_layers.push_back(std::unique_ptr<Layer<Dtype, Dims>>(denseLayer));
            return *this;
        }

        template <int Dims>
        Net<Dtype> &add(Relu<Dtype, Dims> *reluLayer)
        {
            m_layers.push_back(std::unique_ptr<Layer<Dtype, Dims>>(reluLayer));
            return *this;
        }

        template <int Dims>
        Net<Dtype> &add(Softmax<Dtype, Dims> *softmaxLayer)
        {
            m_layers.push_back(std::unique_ptr<Layer<Dtype, Dims>>(softmaxLayer));
            return *this;
        }

    private:
        std::vector<std::unique_ptr<Layer<Dtype>>> m_layers;
        bool m_hasOptimizer;
    };
}

