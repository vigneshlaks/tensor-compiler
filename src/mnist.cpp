#include "../include/mnist.h"
#include <stdexcept>
#include <iostream>

uint32_t readInt(std::ifstream& file) {
    uint8_t bytes[4];
    file.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

MNISTData loadMNIST(const std::string& imgPath, const std::string& lblPath) {
    MNISTData data;

    std::ifstream imgFile(imgPath, std::ios::binary);
    if (!imgFile.is_open()) throw std::runtime_error("Cannot open: " + imgPath);
    readInt(imgFile); // magic
    uint32_t n    = readInt(imgFile);
    uint32_t rows = readInt(imgFile);
    uint32_t cols = readInt(imgFile);
    data.images.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        data.images[i].resize(rows * cols);
        imgFile.read((char*)data.images[i].data(), rows * cols);
    }

    std::ifstream lblFile(lblPath, std::ios::binary);
    if (!lblFile.is_open()) throw std::runtime_error("Cannot open: " + lblPath);
    readInt(lblFile); // magic
    uint32_t m = readInt(lblFile);
    data.labels.resize(m);
    lblFile.read((char*)data.labels.data(), m);

    std::cout << "Loaded " << data.images.size() << " images and "
              << data.labels.size() << " labels\n";

    return data;
}
