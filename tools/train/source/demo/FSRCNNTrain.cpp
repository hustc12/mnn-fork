//
//  FSRCNNTrain.cpp
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
#include "FSRCNN.hpp"
#include "FSRCNNUtils.hpp"
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class FSRCNNTransferModule : public Module {
public:
    FSRCNNTransferModule(const char* fileName) {
        auto varMap  = Variable::loadMap(fileName);
        auto input   = Variable::getInputAndOutput(varMap).first.begin()->second;
        auto lastVar = varMap["FSRCNN/Logits/AvgPool"];

        NN::ConvOption option;
        option.channel = {1280, 4};
        mLastConv      = std::shared_ptr<Module>(NN::Conv(option));

        mFix.reset(NN::extract({input}, {lastVar}, false));

        // Only train last parameter
        registerModel({mLastConv});
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        auto pool   = mFix->forward(inputs[0]);
        auto result = _Softmax(_Reshape(_Convert(mLastConv->forward(pool), NCHW), {0, -1}));
        return {result};
    }
    std::shared_ptr<Module> mFix;
    std::shared_ptr<Module> mLastConv;
};

class FSRCNNTransfer : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            std::cout << "usage: ./runTrainDemo.out FSRCNNTransfer /path/to/FSRCNNModel path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt"
                      << std::endl;
            return 0;
        }

        std::string trainImagesFolder = argv[2];
        std::string trainImagesTxt = argv[3];
        std::string testImagesFolder = argv[4];
        std::string testImagesTxt = argv[5];

        std::shared_ptr<Module> model(new FSRCNNTransferModule(argv[1]));

        FSRCNNUtils::train(model, 4, 0, trainImagesFolder, testImagesFolder);

        return 0;
    }
};

class FSRCNNTrain : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            std::cout << "usage: ./runTrainDemo.out FSRCNNTrain path/to/train/images/ path/to/test/images/" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string trainImagesFolder = argv[1];
//        std::string trainImagesTxt = argv[2];
        std::string testImagesFolder = argv[2];
//        std::string testImagesTxt = argv[4];

        std::shared_ptr<Module> model(new FSRCNN);

        FSRCNNUtils::train(model, 1001, 1, trainImagesFolder, testImagesFolder);

        return 0;
    }
};

DemoUnitSetRegister(FSRCNNTrain, "FSRCNNTrain");
