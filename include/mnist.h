#ifndef MNIST_H
#define MNIST_H

#include <fstream>
#include <vector>
#include <string>

struct MNISTData {
    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;
};

uint32_t readInt(std::ifstream& file);
MNISTData loadMNIST(const std::string& imgPath, const std::string& lblPath);

#endif
