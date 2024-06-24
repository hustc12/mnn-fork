//
//  MnistUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnistUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "MnistDataset.hpp"
#include "../models/MobilenetV2.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "OpGrad.hpp"
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void MnistUtils::train(std::shared_ptr<Module> model, std::string root) {
    {
        // Load snapshot
//        auto para = Variable::load("mnist.snapshot.mnn");
//        model->loadParameters(para);
//        std::shared_ptr<Module> model = new MNN::Train::Model::MobilenetV2(1001, 224, 4);
    }
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 1;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    // size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    // auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    // size_t testIterations = testDataLoader->iterNumber();
    size_t iterations = 1;
    size_t testIterations = 1;
    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        {
            AUTOTIME;
            // dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                // auto cast       = _Cast<float>(example.first[0]);

#define VIT
#ifdef VIT
                /////////////////////////////
                // ViT
                int resizeWidth = 224, resizeHeight = 224;
                int numClasses = 1001;
                auto forwardInput = _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4, halide_type_of<float>());
                auto newTarget = _Input({numClasses}, NCHW, halide_type_of<float>());
                forwardInput->setName("data");
                 newTarget->setName("label");
                 forwardInput->writeMap<float>();
                newTarget->writeMap<float>();

                // auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(numClasses), _Scalar<float>(1.0f), _Scalar<float>(0.0f));
                // auto newTarget = _OneHot(_Squeeze(label , {}), _Scalar<int>(numClasses), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                auto p0 = forwardInput->writeMap<float>();
                for (int i=0;i<3* resizeHeight* resizeWidth;i++) {
                    p0[i] = 0.01;
                }

                auto p = newTarget->writeMap<float>();
                for (int i=0;i<numClasses;i++) {
                    p[i] = 0.1;
                }

////                 Compute One-Hot
//                auto label = _Input({1, 1, 1, 1}, NCHW);
//                label->writeMap<float>();
//                label->setName("label");
//
//                int classes = 1001;
//                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(classes), _Scalar<float>(1.0f),
//                                         _Scalar<float>(0.0f));

#else
                /////////////////////////////
                // FSRCNN

                auto forwardInput = _Input({batchSize, 3, 224, 224}, NC4HW4);
                forwardInput->writeMap<float>();
                forwardInput->setName("data");
                auto cast       = _Cast<float>(forwardInput)  * _Const(1.0f / 255.0f);


//                auto newTarget = _Input({batchSize, 3, 224*3, 224*3}, NC4HW4);
//                newTarget->writeMap<float>();
//                newTarget->setName("data");
//                auto cast_target       = _Cast<float>(newTarget)  * _Const(1.0f / 255.0f);

                VARP newTarget = _Const(1.03, {1, 3, 224*3, 224*3}, NCHW);
#endif
                // AUTOTIME;
                // auto modelInfo = model->getInfo()->inputNames;
                // for (auto info_:modelInfo) {
                    // std::cout << "info name" << modelInfo.size() << std::endl;
                // }
                auto predict = model->forward(forwardInput);
#ifdef VIT
                auto loss    = _CrossEntropy(predict, newTarget);
//                auto loss    = _CrossEntropy(predict, label);

#else
                auto loss    = _MSE(predict, newTarget);
#endif

//#define DEBUG_GRAD
#ifdef DEBUG_GRAD
                {
                    static bool init = false;
                    if (!init) {
                        init = true;
                        std::set<VARP> para;
                        example.first[0].fix(VARP::INPUT);
                        newTarget.fix(VARP::CONSTANT);
                        auto total = model->parameters();
                        for (auto p :total) {
                            para.insert(p);
                        }
                        auto grad = OpGrad::grad(loss, para);
                        total.clear();
                        for (auto iter : grad) {
                            total.emplace_back(iter.second);
                        }
                        Variable::save(total, ".temp.grad");
                    }
                }
#endif
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                std::cout << " begin step" << std::endl;
                sgd->step(loss);
                // if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    // std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
                // }

            }
        }
        // Variable::save(model->parameters(), "mnist.snapshot.mnn");
        // {
        //     model->setIsTraining(false);
        //     auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
        //     forwardInput->setName("data");
        //     auto predict = model->forward(forwardInput);
        //     predict->setName("prob");
        //     Transformer::turnModelToInfer()->onExecute({predict});
        //     Variable::save({predict}, "temp.mnist.mnn");
        // }

        // int correct = 0;
        // testDataLoader->reset();
        // model->setIsTraining(false);
        // int moveBatchSize = 0;
        // for (int i = 0; i < testIterations; i++) {
        //     auto data       = testDataLoader->next();
        //     auto example    = data[0];
        //     moveBatchSize += example.first[0]->getInfo()->dim[0];
        //     if ((i + 1) % 100 == 0) {
        //         std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
        //     }
        //     auto cast       = _Cast<float>(example.first[0]);
        //     example.first[0] = cast * _Const(1.0f / 255.0f);
        //     auto predict    = model->forward(example.first[0]);
        //     predict         = _ArgMax(predict, 1);
        //     auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
        //     correct += accu->readMap<int32_t>()[0];
        // }
        // auto accu = (float)correct / (float)testDataLoader->size();
        // std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
    }
}
