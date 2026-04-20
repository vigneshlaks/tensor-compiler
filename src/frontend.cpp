
#include "../include/frontend.h"
#include "../include/passes.h"
#include <map>
#include <iostream>
#include <fstream>
#include <functional>
#include <random>

LinkedList parseInputs(json instrs);

int parseBytecode() {
    return 0;
};

Metadata parseMetaData(json inputIR) {
    Metadata meta;

    if (!inputIR.contains("metadata") || !inputIR["metadata"].contains("passes")) {
        return meta;
    };

    std::map<std::string, Backend> backendMap = {
        {"cpu", Backend::CPU},
        {"gpu", Backend::GPU}
    };

    std::map<std::string, Precision> precisionMap = {
        {"fp16", Float16},
        {"fp32", Float32},
        {"int8", Int8}
    };

    for (auto& pass : inputIR["metadata"]["passes"]) {
        std::string type = pass["type"];
        json config = pass["config"];

        if (type == "backend") {
            meta.passes.push_back(new BackendPass(backendMap[config["backend"]]));
        } else if (type == "fusion") {
            if (config["enabled"].get<bool>()) {
                meta.passes.push_back(new FusionPass());
            }
        } else if (type == "quantization") {
            meta.passes.push_back(new QuantizationPass(precisionMap[config["precision"]]));
        }
    }
    return meta;
}

LinkedList parseJSON(json inputIR) {
    if (inputIR.contains("input")) {
        return parseInputs(inputIR["input"]);
    }
    return parseInputs(inputIR);
}

// gives the shape of the neural network
// the actual values are filled later
// does initializations (xavier, zero)
LinkedList parseInputs(json instrs) {
    Node* head = nullptr;

    Node* curr = new Node;
    // previous of boundary is null
    curr->prev = head;
 
    // set to first node
    head = curr;

    // for deleting nodes in O(1)
    std::unordered_map<std::string, Node*> nodeMap;

    std::unordered_map<std::string, OpType> opTypeMap = {
        {"const", OpType::Const},
        {"matmul", OpType::Matmul},
        {"relu", OpType::Relu},
        {"mse_loss", OpType::MSE},
        {"softmax", OpType::Softmax},
        {"cross_entropy", OpType::CrossEntropy}
    };
    
    int seed = 12345;
    std::mt19937 gen(seed);
    
    for (json instr : instrs) {
        std::string id = instr["id"].get<std::string>();

        curr->id = id;
        curr->opType = opTypeMap[instr["op"].get<std::string>()];

        if (instr["op"] == "const") {
            auto dim = instr["dim"].get<std::vector<size_t>>();
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(dim);

            if (instr.contains("init")) {
                std::string initType = instr["init"].get<std::string>();
                
                // zero initialization is the default
                if (initType == "xavier") {
                    float batchDim = dim[0];
                    float inputDim = dim[1];

                    float limit = std::sqrt(6.0f / (batchDim + inputDim));
                    std::uniform_real_distribution<float> dist(-limit, limit);

                    for (size_t i = 0; i < output->storage.size(); i++) {
                        output->storage[i] = dist(gen);
                    }
                }
                else if (initType == "import") {
                    // load weights from binary file
                    std::string weightsPath;
                    if (instr.contains("path")) {
                        weightsPath = instr["path"].get<std::string>();
                    } else {
                        std::cerr << "Error: 'path' field required for import init type" << std::endl;
                        continue;
                    }
                    std::ifstream weightsFile(weightsPath, std::ios::binary);
                    if (!weightsFile.is_open()) {
                        std::cerr << "Failed to open weights file: " << weightsPath << std::endl;
                    } else {
                        weightsFile.read(reinterpret_cast<char*>(output->storage.data()),
                                        output->storage.size() * sizeof(float));
                        weightsFile.close();
                        std::cout << "Loaded weights from: " << weightsPath << std::endl;
                    }
                }
            }

            curr->output = output;
            curr->operation = std::make_unique<ConstOp>(output);

            if (instr.contains("trainable")) {
                curr->trainable = instr["trainable"].get<bool>();
            }
        }
        else if (instr["op"] == "matmul") {
            // get the input tensors
            std::shared_ptr<Tensor> lhs = nodeMap[instr["args"][0].get<std::string>()]->output;
            std::shared_ptr<Tensor> rhs = nodeMap[instr["args"][1].get<std::string>()]->output;
            std::vector<size_t> output_dim;

            // get batch dimension if it exists
            for (size_t i = 0; i < lhs->dimension.size() - 2; i++) {
                output_dim.push_back(lhs->dimension[i]);
            }

            output_dim.push_back(lhs->dimension[lhs->dimension.size() - 2]);
            output_dim.push_back(rhs->dimension[rhs->dimension.size() - 1]);

            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(output_dim);

            curr->output = output;
            // mark as trainable for optimizers
            if (instr.contains("trainable")) {
                curr->trainable = instr["trainable"].get<bool>();
            }
            curr->operation = std::make_unique<MatMulOp>(lhs, rhs, output);
        }
        else if (instr["op"] == "relu") {
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            
            // copy input dimension
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(input->dimension);
            curr->output = output;
            curr->operation = std::make_unique<ReluOp>(input, output);
        }
        else if (instr["op"] == "softmax") {
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            
            // copy input dimension
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(input->dimension);
            curr->output = output;
            curr->operation = std::make_unique<SoftmaxOp>(input, output);
        }
        // the dim and storage associated with any loss node
        // refers to the ground truth
        else if (instr["op"] == "mse_loss") {
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // losses are scalars
            std::vector<size_t> output_dim = {1};
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(output_dim);
            std::vector<size_t> dim = instr["dim"].get<std::vector<size_t>>();
            std::shared_ptr<Tensor> ground_truth = std::make_shared<Tensor>(dim);
            curr->output = output;
            curr->operation = std::make_unique<MSEOp>(input, output, ground_truth);
        }
        else if (instr["op"] == "cross_entropy") {
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // losses are scalars
            std::vector<size_t> output_dim = {1};
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(output_dim);
            std::vector<size_t> dim = instr["dim"].get<std::vector<size_t>>();
            std::shared_ptr<Tensor> ground_truth = std::make_shared<Tensor>(dim);
            curr->output = output;
            curr->operation = std::make_unique<CrossEntropyOp>(input, output, ground_truth);
        }

        // store in map
        nodeMap[instr["id"]] = curr;

        // assign pointers
        Node* next = new Node();
        next->prev = curr;
        curr->next = next;

        // move to next node
        curr = next;
    }

    // curr is the end boundary node
    // curr->prev is the last instruction parsed
    Node* tail = curr->prev;

    return {
        head,
        tail,
        nodeMap};
}

void printLinkedList(LinkedList ll) {
    std::map<OpType, std::string> opNames = {
        {OpType::Const, "Const"},
        {OpType::Matmul, "MatMul"},
        {OpType::Relu, "ReLU"},
        {OpType::MatmulRelu, "MatMul+ReLU"},
        {OpType::MSE, "MSE"},
        {OpType::Softmax, "Softmax"},
        {OpType::CrossEntropy, "CrossEntropy"}
    };

    std::cout << "LinkedList (" << ll.nodeMap.size() << " nodes):" << std::endl;

    Node* curr = ll.head;
    while (curr != nullptr && !curr->id.empty()) {
        std::cout << "  " << curr->id << " [" << opNames[curr->opType] << "]";

        if (curr->output) {
            std::cout << " → [";
            for (size_t i = 0; i < curr->output->dimension.size(); i++) {
                std::cout << curr->output->dimension[i];
                if (i < curr->output->dimension.size() - 1) std::cout << "×";
            }
            std::cout << "]";
        }

        if (curr->operation) {
            std::cout << " | Op: " << curr->operation->print();
        } else {
            std::cout << " | No operation (const)";
        }

        std::cout << " | Prev: " << (curr->prev ? curr->prev->id : "null");
        std::cout << " | Next: " << (curr->next && !curr->next->id.empty() ? curr->next->id : "null");

        std::cout << std::endl;
        curr = curr->next;
    }
}
