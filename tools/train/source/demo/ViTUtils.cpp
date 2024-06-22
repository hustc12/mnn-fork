//
//  ViTUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ViTUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "ImageDataset.hpp"
#include "ImageNoLabelDataset.hpp"
#include "module/PipelineModule.hpp"
//#include "cpp/ConvertToFullQuant.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void ViTUtils::train(std::shared_ptr<Module> model, const int numClasses, const int addToLabel,
                                std::string trainImagesFolder,
                                std::string trainImageTxt, const int quantBits) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(0.9f);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(0.00004f);

    auto converImagesToFormat  = CV::RGB;
    int resizeHeight           = 224;
    int resizeWidth            = 224;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
    std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
    bool centerOrRandomCrop = false; // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means,cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto trainDataset = ImageDataset::create(trainImagesFolder, trainImageTxt, datasetConfig.get());
//    auto testDataset = ImageNoLabelDataset::create(testImagesFolder, datasetConfig.get());

    const int trainBatchSize = 1;
    const int trainNumWorkers = 4;
    const int testBatchSize = 10;
    const int testNumWorkers = 0;

    auto trainDataLoader = trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();

    // const int usedSize = 1000;
    // const int testIterations = usedSize / testBatchSize;
    int channels = 1;

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        {
            AUTOTIME;
            trainDataLoader->reset();
            model->setIsTraining(true);
            for (int i = 0; i < trainIterations; i++) {
                AUTOTIME;
                auto trainData  = trainDataLoader->next();
                auto example    = trainData[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(_Convert(example.first[0], NC4HW4)); // NC4HW4
//                MNN_PRINT("DEBUGGING: input dim size = %d\n", input->getInfo()->dim.size());
//                MNN_PRINT("DEBUGGING: input dim = (%d, %d, %d, %d)\n", input->getInfo()->dim.at(0), input->getInfo()->dim.at(1), input->getInfo()->dim.at(2), input->getInfo()->dim.at(3));
//                MNN_PRINT("DEBUGGING: predict dim size = %d\n", predict->getInfo()->dim.size());
//                MNN_PRINT("DEBUGGING: predict dim = (%d, %d, %d, %d)\n", predict->getInfo()->dim.at(0), predict->getInfo()->dim.at(1), predict->getInfo()->dim.at(2), predict->getInfo()->dim.at(3));

                auto loss    = _CrossEntropy(predict, newTarget);
//                 float rate   = LrScheduler::inv(0.0001, solver->currentStep(), 0.0001, 0.75);
                float rate = 1e-5;
                solver->setLearningRate(rate);
                if (solver->currentStep() % 10 == 0) {
                    std::cout << "train iteration: " << solver->currentStep();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate << std::endl;
                }
                solver->step(loss);
            }
        }
    }
}
