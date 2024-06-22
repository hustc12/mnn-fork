//
//  ViTTrain.cpp
//  MNN
//
//  Created by MNN on 2020/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "ViT.hpp"
#include "ViTUtils.hpp"
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class ViTTrain : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            std::cout << "usage: ./runTrainDemo.out ViTTrain path/to/train/images/ path/to/train/images/txt" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string trainImagesFolder = argv[1];
        std::string trainImagesTxt = argv[2];

        std::shared_ptr<Module> model(new ViT);

        ViTUtils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt);

        return 0;
    }
};

DemoUnitSetRegister(ViTTrain, "ViTTrain");
