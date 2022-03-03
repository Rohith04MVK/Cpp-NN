/**
*Cross-entropy loss is used when adjusting model weights during training
*/
#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

namespace nn
{
    template <typename Dtype, int Dims>
    class CrossEntropyLoss
    {
    public:
        CrossEntropyLoss() = default;

        Dtype loss(const Eigen::Tensor<Dtype, Dims> &probabilities, const Eigen::Tensor<Dtype, Dims> &labels);

        Dtype accuracy(const Eigen::Tensor<Dtype, Dims> &probabilities, const Eigen::Tensor<Dtype, Dims> &labels);

        Eigen::Tensor<Dtype, Dims>
        backward(const Eigen::Tensor<Dtype, Dims> &probabilities, const Eigen::Tensor<Dtype, Dims> &labels);
    };

    template <typename Dtype, int Dims>
    Dtype CrossEntropyLoss<Dtype, Dims>::loss(const Eigen::Tensor<Dtype, Dims> &probabilities,
                                              const Eigen::Tensor<Dtype, Dims> &labels)
    {
        int batchSize = probabilities.dimensions()[0];

        static const Dtype stabilizingVal = 0.0001;
        Eigen::Tensor<Dtype, 0> summedLoss = (labels *
                                              (probabilities.constant(stabilizingVal) + probabilities).log())
                                                 .sum();
        return (-1.0 / batchSize) * summedLoss(0);
    }

    template <typename Dtype, int Dims>
    Dtype CrossEntropyLoss<Dtype, Dims>::accuracy(const Eigen::Tensor<Dtype, Dims> &probabilities,
                                                  const Eigen::Tensor<Dtype, Dims> &labels)
    {
        assert(probabilities.dimensions()[0] == labels.dimensions()[0] &&
               "CrossEntropy::accuracy dimensions did not match");
        assert(probabilities.dimensions()[1] == labels.dimensions()[1] &&
               "CrossEntropy::accuracy dimensions did not match");

        auto batchSize = static_cast<Dtype>(labels.dimensions()[0]);

        Eigen::Tensor<bool, 1> ifTensor = probabilities.argmax(1) == labels.argmax(1);
        Eigen::Tensor<Dtype, 1> thenTensor(batchSize);
        auto result = ifTensor.select(thenTensor.constant(1.0), thenTensor.constant(0));
        Eigen::Tensor<Dtype, 0> count = result.sum();
        return static_cast<Dtype>(count(0)) / batchSize;
    }

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> CrossEntropyLoss<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &probabilities,
                                                                       const Eigen::Tensor<Dtype, Dims> &labels)
    {
        return probabilities - labels;
    }
}
