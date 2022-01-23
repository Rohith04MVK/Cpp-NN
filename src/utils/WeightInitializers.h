#ifndef NN_CPP_WEIGHTINITIALIZERS_H
#define NN_CPP_WEIGHTINITIALIZERS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <memory>

namespace nn {
    enum class InitializationScheme {
        GlorotUniform,
        GlorotNormal
    };

    template <typename Dtype>
    class WeightDistribution {
    public:

        explicit WeightDistribution(InitializationScheme scheme, int fanIn, int fanOut):
                m_scheme(scheme),
                m_randomNumberGenerator(std::random_device()())
        {
            if (m_scheme == InitializationScheme::GlorotUniform) {
                Dtype limit = std::sqrt(6.0 / (fanIn + fanOut));
                m_uniformDist.reset(new std::uniform_real_distribution<Dtype>(-limit, limit));
            } else if (m_scheme == InitializationScheme::GlorotNormal) {
                Dtype std = std::sqrt(2.0 / (fanIn + fanOut));
                m_normalDist.reset(new std::normal_distribution<Dtype>(0, std));
            }
        }

        Dtype get() {
            if (m_scheme == InitializationScheme::GlorotUniform) {
                return (*m_uniformDist)(m_randomNumberGenerator);
            } else if (m_scheme == InitializationScheme::GlorotNormal) {
                return (*m_normalDist)(m_randomNumberGenerator);
            } else {
                std::cerr << "Tried to draw from distribution that is uninitialized" << std::endl;
                exit(-1);
            }
        }

    private:
        InitializationScheme m_scheme;                                        
        std::mt19937 m_randomNumberGenerator;                                
        std::unique_ptr<std::uniform_real_distribution<Dtype>> m_uniformDist; 
        std::unique_ptr<std::normal_distribution<Dtype>> m_normalDist;        
    };


    template <typename Dtype>
    Eigen::Tensor<Dtype, 2> getRandomWeights(int inputDimensions, int outputDimensions,
                                             InitializationScheme scheme = InitializationScheme::GlorotUniform) {
        Eigen::Tensor<Dtype, 2> weights(inputDimensions, outputDimensions);
        weights.setZero();

        auto distribution = WeightDistribution<Dtype>(scheme, inputDimensions, outputDimensions);
        for (unsigned int ii = 0; ii < inputDimensions; ++ii) {
            for (unsigned int jj = 0; jj < outputDimensions; ++jj) {
                weights(ii, jj) = distribution.get();
            }
        }
        return weights;
    };
}

#endif //NN_CPP_WEIGHTINITIALIZERS_H
