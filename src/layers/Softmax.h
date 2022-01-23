#ifndef NN_CPP_SOFTMAX_H
#define NN_CPP_SOFTMAX_H

#include "layers/Layer.h"

namespace nn
{
    template <typename Dtype = float, int Dims = 2>
    class Softmax : public Layer<Dtype, Dims>
    {
    public:
        Softmax() = default;

        const std::string &getName()
        {
            const static std::string name = "Softmax";
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
    Eigen::Tensor<Dtype, Dims> Softmax<Dtype, Dims>::forward(const Eigen::Tensor<Dtype, Dims> &input)
    {
        int batchSize = input.dimensions()[0];
        int classDims = input.dimensions()[1];
        auto shiftedInput = input - input.maximum(Eigen::array<int, 1>{1})
                                        .eval()
                                        .reshape(Eigen::array<int, 2>{batchSize, 1})
                                        .broadcast(Eigen::array<int, 2>{1, classDims});

        auto exponentiated = shiftedInput.exp();
        m_output = exponentiated * exponentiated.sum(Eigen::array<int, 1>{1})
                                       .inverse()
                                       .eval()
                                       .reshape(Eigen::array<int, 2>({batchSize, 1}))
                                       .broadcast(Eigen::array<int, 2>({1, classDims}));
        return m_output;
    }

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Softmax<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &accumulatedGrad)
    {
        const int batchSize = accumulatedGrad.dimensions()[0];
        assert(batchSize == m_output.dimensions()[0] && "Dimensions of number of batches does not match");
        return accumulatedGrad / accumulatedGrad.constant(batchSize);
    }
}

#endif //NN_CPP_SOFTMAX_H
