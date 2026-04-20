#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nlohmann/json.hpp>
#include <iostream>

#include "../include/frontend.h"
#include "../include/passes.h"
#include "../include/optimizers.h"
#include "../include/mnist.h"

namespace nb = nanobind;
using json = nlohmann::json;

struct CompiledModel {
    LinkedList list;

    std::vector<float> train(const std::string& imgPath,
                             const std::string& lblPath,
                             float lr       = 0.01f,
                             int   epochs   = 3,
                             int   nSamples = 1000) {
        auto data = loadMNIST(imgPath, lblPath);
        std::cout << "Loaded " << data.images.size() << " images\n";
        SGD sgd(lr, &list);

    #ifdef METAL_FOUND
            std::cout << "Running on Metal\n";
            BackendPass metalPass(METAL);
            metalPass.globalApply(&list);
            sgd.initDevice();
    #else
            std::cout << "Running on CPU\n";
    #endif

        std::vector<float> losses;
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0.0f;
            for (int i = 0; i < nSamples; i++) {
                sgd.zeroGrad();
                sgd.forward(data.images[i], data.labels[i]);
                list.tail->output->toHost();
                epochLoss += list.tail->output->storage[0];
                sgd.backward();
                sgd.descentStep();
            }
            float avg = epochLoss / nSamples;
            std::cout << "Epoch " << epoch << " | Avg Loss: " << avg << "\n";
            losses.push_back(avg);
        }
        return losses;
    }

    void print_graph() { printLinkedList(list); }
};

// Parses and runs fusion
CompiledModel compile(nb::object ir) {
    nb::object json_mod = nb::module_::import_("json");
    std::string json_str = nb::cast<std::string>(json_mod.attr("dumps")(ir));
    json inputIR = json::parse(json_str);

    if (inputIR.is_array()) {
        inputIR = {{"input", inputIR}};
    }

    LinkedList list = parseJSON(inputIR);

    FusionPass fusion;
    int fused = fusion.globalApply(&list);
    std::cout << "FusionPass: " << fused << " fusion(s)\n";

    return CompiledModel{std::move(list)};
}

// Import tensor frontend
NB_MODULE(tensor_frontend, m) {
    m.doc() = "tensor compiler frontend";
    
    // Registers Compiled model as a Python class with 2 methods
    // "train" and "print_graph" and a compile function
    nb::class_<CompiledModel>(m, "CompiledModel")
        .def("train", &CompiledModel::train,
             nb::arg("img_path"),
             nb::arg("lbl_path"),
             nb::arg("lr")        = 0.01f,
             nb::arg("epochs")    = 3,
             nb::arg("n_samples") = 1000)
        .def("print_graph", &CompiledModel::print_graph);
    
    m.def("compile", &compile, nb::arg("ir"));
}
