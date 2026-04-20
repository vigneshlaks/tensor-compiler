#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/frontend.h"
#include "../include/passes.h"
#include "../include/optimizers.h"
#include "../include/mnist.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    auto trainData = loadMNIST("data/MNIST/raw/train-images-idx3-ubyte",
                               "data/MNIST/raw/train-labels-idx1-ubyte");
    auto testData = loadMNIST("data/MNIST/raw/t10k-images-idx3-ubyte",
                              "data/MNIST/raw/t10k-labels-idx1-ubyte");

    std::cout << "Train images: " << trainData.images.size() << std::endl;
    std::cout << "Train labels: " << trainData.labels.size() << std::endl;

    std::string filename = "irs/mnist/mnist.json";

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open " << filename << std::endl;
        return 0;
    }
    json inputIR;
    file >> inputIR;

    Metadata meta = parseMetaData(inputIR);
    LinkedList list = parseJSON(inputIR);

    PassManager pm(&list, meta.passes);
    pm.runGlobal();

    SGD sgd = SGD(0.01f, &list);
    int numEpochs = 3;
    size_t numSamples = 1000;

    #ifdef CUDA_FOUND
    std::cout << "Running on GPU" << std::endl;
    BackendPass gpuPass(GPU);
    gpuPass.globalApply(&list);
    sgd.initDevice();
    #endif

    #ifdef METAL_FOUND
    std::cout << "Running on Metal (M1)" << std::endl;
    BackendPass metalPass(METAL);
    metalPass.globalApply(&list);
    sgd.initDevice();
    #endif

    #if !defined(CUDA_FOUND) && !defined(METAL_FOUND)
    std::cout << "Running on CPU" << std::endl;
    #endif

    

    std::cout << "\n--- Training ---" << std::endl;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        for (size_t i = 0; i < numSamples; i++) {
            sgd.zeroGrad();
            sgd.forward(trainData.images[i], trainData.labels[i]);

            list.tail->output->toHost();
            epochLoss += list.tail->output->storage[0];

            sgd.backward();
            sgd.descentStep();
        }

        std::cout << "Epoch " << epoch << " | Avg Loss: " << epochLoss / numSamples << std::endl;
    }

    sgd.syncToHost();

    return 0;
}