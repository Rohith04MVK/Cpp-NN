<h1 align="center">C++ NN 🧠</h1>
<h3 align="center">A simple Neural Network library written in C++</h3>

## Installation 🚀

#### Clone the repo

```sh
git clone https://github.com/Rohith04MVK/Cpp-NN
```

#### Run the examples

```sh
# TODO add tests
mkdir build
cd build
cmake -D CPP_NN_BUILD_EXAMPLE=ON ..
make
```

## The Structure of Networks
```cpp
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
```
## Training!
```cpp
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
