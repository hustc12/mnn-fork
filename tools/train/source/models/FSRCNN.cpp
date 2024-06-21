#include <algorithm>
#include <iostream>
#include "FSRCNN.hpp"

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

// Draft the FSRCNN structure

//// Feature Extraction
class _Feature_Extraction : public Module {
public:
    _Feature_Extraction(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d_fe;
};

_Feature_Extraction::_Feature_Extraction(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding, bool depthwise) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inputChannels, outputChannels};
//    convOption.padMode = Express::SAME;
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};
    convOption.depthwise = depthwise;
    conv2d_fe.reset(NN::Conv(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv2d_fe});
}
std::vector<Express::VARP> _Feature_Extraction::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;

    VARP x = inputs[0];
    x = conv2d_fe->forward(x);
    return {x};
}

std::shared_ptr<Module> FeatureExtraction(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Feature_Extraction(inputOutputChannels, kernelSize, stride, padding, depthwise));
}

//// Shrinking
class _Shrinking: public Module {
public:
    _Shrinking(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d_shrink;
};

_Shrinking::_Shrinking(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
                       bool depthwise) {
    int inChannels = inputOutputChannels[0], outChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inChannels, outChannels};
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};
    convOption.depthwise = depthwise;

    conv2d_shrink.reset(NN::Conv(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
    registerModel({conv2d_shrink});
}

std::vector<Express::VARP> _Shrinking::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv2d_shrink->forward(x);
    return {x};
}

std::shared_ptr<Module> Shrinking(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Shrinking(inputOutputChannels, kernelSize, stride, padding, depthwise));
}

///// Mapping
class _MappingBlock: public Module {
public:
    _MappingBlock(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
             bool depthwise);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d_mapping;
};

_MappingBlock::_MappingBlock(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
       bool depthwise) {
    int inChannels = inputOutputChannels[0], outChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inChannels, outChannels};
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};
    convOption.depthwise = depthwise;

    conv2d_mapping.reset(NN::Conv(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
    registerModel({conv2d_mapping});
}

std::vector<Express::VARP> _MappingBlock::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv2d_mapping->forward(x);
    // TODO: PReLU
    return {x};
}

std::shared_ptr<Module> MappingBlock(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
                                     bool depthwise) {
    return std::shared_ptr<Module>(new _MappingBlock(inputOutputChannels, kernelSize, stride, padding, depthwise));
}

//// Expanding
class _Expanding: public Module {
public:
    _Expanding(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d_expand;
};

_Expanding::_Expanding(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
                       bool depthwise) {
    int inChannels = inputOutputChannels[0], outChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inChannels, outChannels};
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};
    convOption.depthwise = depthwise;

    conv2d_expand.reset(NN::Conv(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
    registerModel({conv2d_expand});
}

std::vector<Express::VARP> _Expanding::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv2d_expand->forward(x);
    return {x};
}

std::shared_ptr<Module> Expanding(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Expanding(inputOutputChannels, kernelSize, stride, padding, depthwise));
}

//// Deconvolution
class _Deconvolution: public Module {
public:
    _Deconvolution(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, int padding = 1, int out_padding = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> deconv;
};

_Deconvolution::_Deconvolution(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
                               int out_padding, bool depthwise) {
    int inChannels = inputOutputChannels[0], outChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inChannels, outChannels};
    convOption.stride = {stride, stride};
    convOption.pads = {padding, padding};

    // TODO: out_padding?
    convOption.depthwise = depthwise;
    deconv.reset(NN::ConvTranspose(convOption, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({deconv});
}
std::vector<Express::VARP> _Deconvolution::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = deconv->forward(x);
    return {x};
}

std::shared_ptr<Module> Deconvolution(std::vector<int> inputOutputChannels, int kernelSize, int stride, int padding,
                                      int out_padding, bool depthwise) {
    return std::shared_ptr<Module>(new _Deconvolution(inputOutputChannels, kernelSize, stride, padding, out_padding, depthwise));
}

/////////////////////////////////////
//// FSRCNN Model
FSRCNN::FSRCNN(int num_channels, int d, int s, int m, int upscale_factor) {

    // 1. Feature Extraction
    feature_extraction = FeatureExtraction({1, 56}, 5, 1, 2);

    // 2. Shrinking
    shrinking = Shrinking({56, 12}, 1, 1, 0);

    // 3. Mapping
    for (int i=0; i<m; i++) {
        mapping.emplace_back(MappingBlock({12,12}, 3,1,1, false));
    }

    // 4. Expanding
    expanding = Expanding({12, 56}, 1,1,0);

    // 5. Deconvolution
    deconvolution = Deconvolution({56, 1}, 9, 3, 4, 2, false);

    registerModel({feature_extraction, shrinking, expanding, deconvolution});
    registerModel(mapping);
}

std::vector<Express::VARP> FSRCNN::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    // TODO: PReLU
//    MNN_PRINT("DEBUGGING: Start FSRCNN forwarding!\n");
    x = feature_extraction->forward(x);
    x = shrinking->forward(x);
    for (int i = 0; i < mapping.size(); i++) {
        x = mapping[i]->forward(x);
    }
    x = expanding->forward(x);
    x = deconvolution->forward(x);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
