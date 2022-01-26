#include "../src/Net.h"
#include "../src/loss/CrossEntropy.h"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <map>
#include <iomanip>

const std::map<std::string, int> IRIS_TYPE_TO_INT{
    {"Iris-setosa", 0},
    {"Iris-versicolor", 1},
    {"Iris-virginica", 2}};

struct IrisDataset
{
    std::vector<std::array<float, 4>> data;
    std::vector<int> labels;
};

IrisDataset loadIrisDataset(const std::string &path = "../examples/data/iris_data.csv")
{
    IrisDataset dataset;

    std::ifstream irisFile(path);
    std::string line;
    while (std::getline(irisFile, line, '\n'))
    {
        std::vector<std::string> values;

        boost::split(values, line, [](char c)
                     { return c == ','; });

        if (values.size() < 5)
        {
            std::cout << "Found line with less than five elements, skipping" << std::endl;
            continue;
        }

        float sepalLength = std::stof(values[0]);
        float sepalWidth = std::stof(values[1]);
        float petalLength = std::stof(values[2]);
        float petalWidth = std::stof(values[3]);
        std::string labelName = values[4];

        auto labelIter = IRIS_TYPE_TO_INT.find(labelName);

        if (labelIter == IRIS_TYPE_TO_INT.end())
        {
            std::cerr << "Unknown Iris type of: " << labelName << " please check dataset." << std::endl;
            exit(-1);
        }
        int labelInt = labelIter->second;
        dataset.data.push_back({sepalLength, sepalWidth, petalLength, petalWidth});
        dataset.labels.push_back(labelInt);
    }

    return dataset;
}

int main()
{
    auto dataset = loadIrisDataset();

    int batchSize = dataset.labels.size();
    int numFeatures = dataset.data[0].size();
    int numClasses = *std::max_element(dataset.labels.begin(), dataset.labels.end()) + 1;

    Eigen::Tensor<float, 2> input(batchSize, numFeatures);
    Eigen::Tensor<float, 2> labels(batchSize, numClasses);
    input.setZero();
    labels.setZero();

    for (unsigned int ii = 0; ii < batchSize; ++ii)
    {
        for (unsigned int feature = 0; feature < numFeatures; ++feature)
        {
            input(ii, feature) = dataset.data[ii][feature];
        }

        labels(ii, dataset.labels[ii]) = 1.0;
    }

    int numHiddenNodes = 20;
    bool useBias = true;

    nn::Net<float> net;
    net.add(new nn::Dense<>(batchSize, numFeatures, numHiddenNodes, useBias));
    net.add(new nn::Relu<>());
    net.add(new nn::Dense<>(batchSize, numHiddenNodes, numHiddenNodes, useBias));
    net.add(new nn::Relu<>());
    net.add(new nn::Dense<>(batchSize, numHiddenNodes, numClasses, useBias));
    net.add(new nn::Softmax<>());

    nn::CrossEntropyLoss<float, 2> lossFunc;
    net.registerOptimizer(new nn::Adam<float>(0.01));

    int numEpoch = 250;
    float loss_t, accuracy_t;
    for (unsigned int ii = 0; ii < numEpoch; ++ii)
    {
        auto result = net.forward<2, 2>(input);

        float loss = lossFunc.loss(result, labels);
        float accuracy = lossFunc.accuracy(result, labels);
        std::cout << std::setprecision(5);
        std::cout << "Epoch: " << ii << " loss: " << loss << " accuracy: " << accuracy << std::endl;

        auto lossBack = lossFunc.backward(result, labels);
        net.backward(lossBack);
        net.step();
        loss_t = loss;
        accuracy_t = accuracy;
    }
    std::cout << "Final Loss: " << loss_t << std::endl;
    std::cout << "Final Accuracy: " << accuracy_t << std::endl;

    return 0;
}
