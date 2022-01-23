#ifndef NN_CPP_HUBERLOSS_H
#define NN_CPP_HUBERLOSS_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace nn {
    template<typename Dtype, int Dims>
    class HuberLoss {
    public:

        explicit HuberLoss(Dtype threshold = 1.0): m_threshold(threshold) {}

        Dtype loss(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);

        Eigen::Tensor<Dtype, Dims>
        backward(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);

    private:
        Dtype m_threshold;                                
        Eigen::Tensor<bool, Dims> m_cachedSwitchResults;
    };

    template<typename Dtype, int Dims>
    Dtype HuberLoss<Dtype, Dims>::loss(const Eigen::Tensor<Dtype, Dims> &predictions,
                                              const Eigen::Tensor<Dtype, Dims> &labels) {
        assert(predictions.dimensions()[0] == labels.dimensions()[0] &&
               "HuberLoss::loss dimensions don't match");
        assert(predictions.dimensions()[1] == labels.dimensions()[1] &&
               "HuberLoss::loss dimensions don't match");
        int batchSize = predictions.dimensions()[0];

        auto error = predictions - labels;
        auto absoluteError = error.abs();

        m_cachedSwitchResults = absoluteError <= m_threshold;

        auto lessThanThreshold = error.constant(0.5) * error.square();
        auto moreThanThreshold = error.constant(m_threshold) * absoluteError - error.constant(0.5 * pow(m_threshold, 2));


        auto perItemLoss = m_cachedSwitchResults.select(
                lessThanThreshold,
                moreThanThreshold);

        Eigen::Tensor<Dtype, 0> sum = perItemLoss.sum();
        return sum(0) / batchSize;
    }

    template<typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> HuberLoss<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &predictions,
                                                                       const Eigen::Tensor<Dtype, Dims> &labels) {

        auto error = predictions - labels;

        auto errorPositiveOrZero = error >= static_cast<Dtype>(0);
        auto absoluteErrorGrad = errorPositiveOrZero.select(error.constant(m_threshold), error.constant(-m_threshold));
        return m_cachedSwitchResults.select(error, absoluteErrorGrad);
    }
}

#endif //NN_CPP_SMOOTHL1LOSS_H
