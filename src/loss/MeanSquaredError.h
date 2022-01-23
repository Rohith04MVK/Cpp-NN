#ifndef NN_CPP_MEANSQUAREDERROR_H
#define NN_CPP_MEANSQUAREDERROR_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace nn
{
    template <typename Dtype, int Dims>
    class MeanSquaredError
    {
    public:
        MeanSquaredError() = default;

        Dtype loss(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);

        Eigen::Tensor<Dtype, Dims>
        backward(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);
    };

    template <typename Dtype, int Dims>
    Dtype MeanSquaredError<Dtype, Dims>::loss(const Eigen::Tensor<Dtype, Dims> &predictions,
                                              const Eigen::Tensor<Dtype, Dims> &labels)
    {
        assert(predictions.dimensions()[0] == labels.dimensions()[0] &&
               "MeanSquaredError::loss dimensions don't match");
        assert(predictions.dimensions()[1] == labels.dimensions()[1] &&
               "MeanSquaredError::loss dimensions don't match");

        int batchSize = predictions.dimensions()[0];

        Eigen::Tensor<Dtype, 0> squaredSum = (predictions - labels).square().sum();
        return squaredSum(0) / batchSize;
    }

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> MeanSquaredError<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &predictions,
                                                                       const Eigen::Tensor<Dtype, Dims> &labels)
    {
        return predictions - labels;
    }
}

#endif //NN_CPP_MEANSQUAREDERROR_H
