#pragma once

#include "layers/Layer.h"
#include "utils/WeightInitializers.h"

namespace nn
{

    template <typename Dtype = float, int Dims = 2>
    class Dense : public Layer<Dtype, Dims>
    {
    public:
        explicit Dense(int batchSize, int inputDimension, int outputDimension, bool useBias,
                       InitializationScheme weightInitializer = InitializationScheme::GlorotUniform);

        const std::string &getName()
        {
            const static std::string name = "Dense";
            return name;
        }

        Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input);

        Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &accumulatedGrad);

        Eigen::array<Eigen::Index, Dims> getOutputShape()
        {
            return m_outputShape;
        };

        void step();

        void registerOptimizer(std::shared_ptr<StochasticGradientDescent<Dtype>> optimizer);

        void registerOptimizer(std::shared_ptr<Adam<Dtype>> optimizer);

    private:
        Eigen::array<Eigen::Index, Dims> m_outputShape; ///< The output shape of this layer
        Eigen::Tensor<Dtype, Dims> m_inputCache;        ///< Cache the input to calculate gradient
        Eigen::Tensor<Dtype, Dims> m_weights;           ///< Our weights of the layer
        Eigen::Tensor<Dtype, Dims> m_bias;              ///< The bias weights if specified

        // Gradients
        Eigen::Tensor<Dtype, Dims> m_weightsGrad;                      ///< The gradient of the weights
        Eigen::Tensor<Dtype, Dims> m_biasGrad;                         ///< The gradient of the bias
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> m_weightOptimizer; ///< The optimizer of our weights
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> m_biasOptimizer;   ///< The optimizer of our bias

        bool m_useBias; ///< Whether we use the bias
    };

    template <typename Dtype, int Dims>
    Dense<Dtype, Dims>::Dense(int batchSize, int inputDimension, int outputDimension, bool useBias,
                              InitializationScheme weightInitializer) : m_outputShape({batchSize, outputDimension}),
                                                                        m_useBias(useBias)
    {
        m_weights = getRandomWeights<Dtype>(inputDimension, outputDimension, weightInitializer);

        m_weightsGrad = Eigen::Tensor<Dtype, Dims>(inputDimension, outputDimension);
        m_weightsGrad.setZero();

        if (useBias)
        {
            m_bias = getRandomWeights<Dtype>(1, outputDimension, weightInitializer);

            m_biasGrad = Eigen::Tensor<Dtype, Dims>(1, outputDimension);
            m_biasGrad.setZero();
        }
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Dense<Dtype, Dims>::forward(const Eigen::Tensor<Dtype, Dims> &input)
    {
        assert(input.dimensions()[1] == m_weights.dimensions()[0] &&
               "Dense::forward dimensions of input and weights do not match");
        m_inputCache = input;

        Eigen::array<Eigen::IndexPair<int>, 1> productDims = {Eigen::IndexPair<int>(1, 0)};
        auto result = input.contract(m_weights, productDims);

        if (m_useBias)
        {
            return result + m_bias.broadcast(Eigen::array<Eigen::Index, 2>{input.dimensions()[0], 1});
        }
        else
        {
            return result;
        }
    }

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Dense<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &accumulatedGrad)
    {
        assert(accumulatedGrad.dimensions()[0] == m_inputCache.dimensions()[0] &&
               "Dense::backward dimensions of accumulatedGrad and inputCache do not match");
        // m_inputCache is of shape (batchSize, inputDimension)
        // accumulatedGrad is of shape (batchSize, outputDimension)
        // So we want to contract along dimensions (0, 0), aka m_inputCache.T * accumulatedGrad
        // Where dimensions would be (inputDimension, batchSize) * (batchSize, outputDimension)
        static const Eigen::array<Eigen::IndexPair<int>, 1> transposeInput = {Eigen::IndexPair<int>(0, 0)};

        m_weightsGrad = m_inputCache.contract(accumulatedGrad, transposeInput);
        if (m_useBias)
        {
            m_biasGrad = accumulatedGrad.sum(Eigen::array<int, 1>{0}).eval().reshape(Eigen::array<Eigen::Index, 2>{1, m_outputShape[1]});
        }

        static const Eigen::array<Eigen::IndexPair<int>, 1> transposeWeights = {Eigen::IndexPair<int>(1, 1)};
        return accumulatedGrad.contract(m_weights, transposeWeights);
    }

    template <typename Dtype, int Dims>
    void Dense<Dtype, Dims>::step()
    {
        m_weights -= m_weightOptimizer->weightUpdate(m_weightsGrad);

        if (m_useBias)
        {
            m_bias -= m_biasOptimizer->weightUpdate(m_biasGrad);
        }
    }

    template <typename Dtype, int Dims>
    void Dense<Dtype, Dims>::registerOptimizer(std::shared_ptr<StochasticGradientDescent<Dtype>> optimizer)
    {
        m_weightOptimizer = std::move(optimizer->template createOptimizer<Dims>());

        if (m_useBias)
        {
            m_biasOptimizer = std::move(optimizer->template createOptimizer<Dims>());
        }
    }

    template <typename Dtype, int Dims>
    void Dense<Dtype, Dims>::registerOptimizer(std::shared_ptr<Adam<Dtype>> optimizer)
    {
        m_weightOptimizer = std::move(optimizer->template createOptimizer<Dims>());

        if (m_useBias)
        {
            m_biasOptimizer = std::move(optimizer->template createOptimizer<Dims>());
        }
    }
}
