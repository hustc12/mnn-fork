//
//  FSRCNNUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FSRCNNUtils.hpp"
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

void FSRCNNUtils::train(std::shared_ptr<Module> model, const int numClasses, const int addToLabel,
                                std::string trainImagesFolder,
                                std::string testImagesFolder, const int quantBits) {
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
    auto trainDataset = ImageNoLabelDataset::create(trainImagesFolder, datasetConfig.get());
//    auto testDataset = ImageNoLabelDataset::create(testImagesFolder, datasetConfig.get());

    const int trainBatchSize = 1;
    const int trainNumWorkers = 4;
    const int testBatchSize = 10;
    const int testNumWorkers = 0;

    auto trainDataLoader = trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);
//    auto testDataLoader = testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();
//    const int testIterations = testDataLoader->iterNumber();

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

//                // Compute One-Hot
//                auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(example.second[0] + _Scalar<int32_t>(addToLabel), {})),
//                                  _Scalar<int>(numClasses), _Scalar<float>(1.0f),
//                                         _Scalar<float>(0.0f));

////                VARP input = example.first[0];
////                VARP newTarget = example.first[0]; // TODO: To update the target
//                VARP input = _Convert(example.first[0], NC4HW4);
//                VARP input = _Convert (_Const(1.03, {1, 1, resizeHeight, resizeWidth}), NC4HW4);
//                VARP newTarget = _Convert( _Const(1.03, {1, 1, resizeHeight*3, resizeWidth*3}), NC4HW4);

                VARP input = _Const(1.03, {1, channels, resizeHeight, resizeWidth}, NCHW);
                VARP newTarget = _Const(1.03, {1, channels, resizeHeight*3, resizeWidth*3}, NCHW);

//                VARP input = _Input({1,1,resizeHeight, resizeWidth});
//                VARP newTarget = _Input({1,1,resizeHeight*3, resizeWidth*3});

                auto predict = model->forward(input); // NC4HW4
//                MNN_PRINT("DEBUGGING: input dim size = %d\n", input->getInfo()->dim.size());
//                MNN_PRINT("DEBUGGING: input dim = (%d, %d, %d, %d)\n", input->getInfo()->dim.at(0), input->getInfo()->dim.at(1), input->getInfo()->dim.at(2), input->getInfo()->dim.at(3));
//                MNN_PRINT("DEBUGGING: predict dim size = %d\n", predict->getInfo()->dim.size());
//                MNN_PRINT("DEBUGGING: predict dim = (%d, %d, %d, %d)\n", predict->getInfo()->dim.at(0), predict->getInfo()->dim.at(1), predict->getInfo()->dim.at(2), predict->getInfo()->dim.at(3));

                auto loss    = _MSE(predict, newTarget);
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

//        int correct = 0;
//        int sampleCount = 0;
//        testDataLoader->reset();
//        model->setIsTraining(false);
//        exe->gc(Executor::PART);

//        AUTOTIME;
//        for (int i = 0; i < testIterations; i++) {
//            auto data       = testDataLoader->next();
//            auto example    = data[0];
//            auto predict    = model->forward(_Convert(example.first[0], NC4HW4));
//            predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
//            auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
//            sampleCount += label->getInfo()->size;
//            auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));
//            correct += accu->readMap<int32_t>()[0];
//
//            if ((i + 1) % 10 == 0) {
//                std::cout << "test iteration: " << (i + 1) << " ";
//                std::cout << "acc: " << correct << "/" << sampleCount << " = " << float(correct) / sampleCount * 100 << "%";
//                std::cout << std::endl;
//            }
//        }
//        auto accu = (float)correct / testDataLoader->size();
//        // auto accu = (float)correct / usedSize;
//        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;

//        {
//            auto forwardInput = _Input({1, channels, resizeHeight, resizeWidth}, NCHW); //NC4HW4
//            forwardInput->setName("data");
//            auto predict = model->forward(forwardInput);
//            Transformer::turnModelToInfer()->onExecute({predict});
//            predict->setName("prob");
//            std::string fileName = "temp.FSRCNN.mnn";
//            Variable::save({predict}, fileName.c_str());
////            ConvertToFullQuant::convert(fileName);
//        }
    }
}
