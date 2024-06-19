#include <algorithm>
#include <iostream>
#include "ViT.hpp"

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

// Draft the ViT structure

//// Conv2d
class _Conv2d : public Module {
public:
    _Conv2d(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv2d;
};

_Conv2d::_Conv2d(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel = {inputChannels, outputChannels};
    convOption.padMode = Express::SAME;
    convOption.stride = {stride, stride};
    convOption.depthwise = depthwise;
    conv2d.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv2d});
}
std::vector<Express::VARP> _Conv2d::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;

    VARP x = inputs[0];
    x = conv2d->forward(x);
    return {x};
}

std::shared_ptr<Module> Conv2d(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false) {
    return std::shared_ptr<Module>(new _Conv2d(inputOutputChannels, kernelSize, stride, depthwise));
}

////// Encoder = dropout + sequential(EncoderBlocks * 12)
//class _Encoder : public Module{
//public:
//    _Encoder();
//    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
//    std::shared_ptr<Module> dropout;
//};
//
//std::shared_ptr<Module> Encoder() {
//    return std::shared_ptr<Module>(new _Encoder());
//}

//// EncoderBlock = LayerNorm(Skip) + MultiheadAttention + Dropout + LayerNorm(Skip) + MLPBlock
class _EncoderBlock:public Module{
public:
    _EncoderBlock();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
};

std::shared_ptr<Module> EncoderBlock(){
    // LayerNorm
    // MultiheadAttention
    // Dropout
    // LayerNorm
    // MLPBlock
    return std::shared_ptr<Module>(new _EncoderBlock());
}

//// Attention
class _Attention : public Module {
public:
    _Attention(int in_dim, int Attention_dim);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    std::shared_ptr<Module> matmul0;
};

std::shared_ptr<Module> Attention(int hidden_dim, int num_heads) {
    return std::shared_ptr<Module>(new _Attention(hidden_dim, num_heads));
}

_Attention::_Attention(int hidden_dim, int num_heads) {
    matmul0.reset(NN::Linear(hidden_dim, hidden_dim*3));
    registerModel({matmul0});
}

std::vector<Express::VARP> _Attention::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = matmul0->forward(x);
    // yy = _Slice(yy, _Const(sliceStartData, {4}, NCHW), _Const(sliceEndData, {4}, NCHW));
    auto q = _Slice(x,                       0, _Const(768, {4}, NCHW));
    auto k = _Slice(x, _Const(768, {4}, NCHW) , _Const(768, {4}, NCHW));
    auto v = _Slice(x, _Const(1536, {4}, NCHW), _Const(768, {4}, NCHW));
    q = _Transpose(_Reshape(q, {197, 12, 64}), {1,0,2});
    k = _Transpose(_Reshape(q, {197, 12, 64}), {1,2,0});
    v = _Transpose(_Reshape(q, {197, 12, 64}), {1,0,2});
    q = _Divide(q, _Const(8, {4}, NCHW));
    qk = _Softmax()
    return {x};
}

//// MLPBlock = Linear + GELU + Dropout + Linear + Dropout
class _MLP : public Module {
public:
    _MLP(int in_dim, int mlp_dim);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> linear0;
    std::shared_ptr<Module> dropout0;
    std::shared_ptr<Module> gelu;
    std::shared_ptr<Module> linear1;
    std::shared_ptr<Module> dropout1;
};

_MLP::_MLP(int in_dim, int mlp_dim) {
    linear0.reset(NN::Linear(in_dim, mlp_dim));
    dropout0.reset(NN::Dropout(0.1));
    linear1.reset(NN::Linear(mlp_dim, in_dim));
    dropout1.reset(NN::Dropout(0.1));

    registerModel({linear0, dropout0, linear1, dropout1});
}

std::vector<Express::VARP> _MLP::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = linear0->forward(x);
    x = _Gelu(x);
    x = dropout0->forward(x);
    x = linear1->forward(x);
    x = dropout1->forward(x);
    return {x};
}

std::shared_ptr<Module> MLP(int in_dim, int mlp_dim) {
    return std::shared_ptr<Module>(new _MLP(in_dim, mlp_dim));
}
//
////// Final Linear Block
//class _Linear : public Module {
//public:
//    _Linear(int in_feature, int out_feature, bool bias);
//    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
//
//    std::shared_ptr<Module> linear_block;
//};
//
//_Linear::_Linear(int in_feature, int out_feature, bool bias) {
//    linear_block.reset(NN::Linear(in_feature, out_feature, bias));
//
//    registerModel({linear_block});
//}
//
//std::vector<Express::VARP> _Linear::onForward(const std::vector<Express::VARP> &inputs) {
//    using namespace Express;
//
//    VARP x = inputs[0];
//    x = linear_block->forward(x);
//    return {x};
//}
//
//std::shared_ptr<Module> Linear(int in_feature, int out_feature, bool bias) {
//    return std::shared_ptr<Module>(new _Linear(in_feature, out_feature, bias));
//}


ViT::ViT(int numClasses, int patch_size, int num_layers, int num_heads, int hidden_dim, int mlp_dim) {

    // 1. conv_proj
    // TODO: Double check the encoder_layer
    conv_proj = Conv2d({}, 3, 1);

    // 2. dropout
    drop_out.reset(NN::Dropout(0.1));

    // 2. encoder_layers
    // TODO: Double check the encoder_layer
    for (int i=0; i<12; i++) {
        encoder_layers.emplace_back(EncoderBlock());
    }

//    // 3. last_layer_norm. NOTE: Currently we skip LayerNorm temporarily

    // 4. Final Linear Block
    // TODO: Double check the parameters of Linear block
//    linear = Linear(768, 1000, false);
    linear.reset(NN::Linear(768, 1000, false));

    registerModel({conv_proj, linear});
    registerModel(encoder_layers);
}

std::vector<Express::VARP> ViT::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x = conv_proj->forward(x);
    x = drop_out->forward(x);
    
    // TODO: To chain blocks
    for (int i=0; i<12; i++) {
        x = encoder_layers[i]->forward(x);
    }

//    3. last_layer_norm. NOTE: Currently we skip LayerNorm temporarily
//    x = last_layer_norm->forward(x);
    x = linear->forward(x);
    return {x};
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ResNet::ResNet(int numClasses, ResNetType resNetType) {
//    std::vector<int> numbers;
//    {
//        auto numbersMap = std::map<ResNetType, std::vector<int>>({
//            {ResNet18, {2, 2, 2, 2}},
//            {ResNet34, {3, 4, 6, 3}},
//            {ResNet50, {3, 4, 6, 3}},
//            {ResNet101, {3, 4, 23, 3}},
//            {ResNet152, {3, 8, 36, 3}}
//        });
//        numbers = numbersMap[resNetType];
//    }
//    std::vector<int> channels({64, 64, 128, 256, 512});
//    {
//        if (resNetType != ResNet18 && resNetType != ResNet34) {
//            channels[0] = 16;
//        }
//    }
//
//    std::vector<int> strides({1, 2, 2, 2});
//
//    firstConv = ConvBnRelu2({3, 64}, 7, 2);
//    // firstConv = ConvBnRelu2({3, 1}, 7, 2);
//    for (int i = 0; i < 4; ++i) {
//        if (resNetType == ResNet18 || resNetType == ResNet34) {
//            //layers.emplace_back(ConvBnRelu2({inputChannels, expandChannels}, 1));
//            std::cout << "channels: " << channels[i] << " " << channels[i+1] << std::endl;
//            residualBlocks.emplace_back(Residule({channels[i], channels[i+1]}, strides[i]));
//            for (int i = 1; i < numbers[i]; ++i) {
//                residualBlocks.emplace_back(Residule({channels[i+1], channels[i+1]}, strides[i]));
//            }
//        }
//        // else {
//        //     x = bottleNeckBlock(x, {channels[i] * 4, channels[i+1], channels[i+1] * 4}, strides[i], numbers[i]);
//        // }
//    }
//    // lastConv = ConvBnRelu2({3, 512}, 1, 1); // reshape FC with Conv1x1
//    int last_c = 128;
//    // fc.reset(NN::Linear(last_c, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
//    fc = ConvBnRelu2({512, numClasses}, 1, 1); // reshape FC with Conv1x1
//    // x = _Conv(0.0f, 0.0f, x, {x->getInfo()->dim[1], numClass}, {1, 1}, VALID, {1, 1}, {1, 1}, 1); // reshape FC with Conv1x1
//    // registerModel({firstConv, fc});
//    registerModel({firstConv, fc});
//    registerModel(residualBlocks);
//}
//
//std::vector<Express::VARP> ResNet::onForward(const std::vector<Express::VARP> &inputs) {
//    using namespace Express;
//    VARP x = inputs[0];
//    std::cout << "forward0 " << x->getInfo()->dim[0] << " " << x->getInfo()->dim[1] << std::endl;
//    x = firstConv->forward(x);
//    std::cout << "forward1 " << x->getInfo()->dim[0] << " " << x->getInfo()->dim[1] << " " << x->getInfo()->dim[2] << " " << x->getInfo()->dim[3] << std::endl;
//    x = _MaxPool(x, {3, 3}, {2, 2}, SAME);
//    std::cout << "forward2 " << x->getInfo()->dim[0] << " " << x->getInfo()->dim[1] << " " << x->getInfo()->dim[2] << " " << x->getInfo()->dim[3] << std::endl;
//    for (int i = 0; i < 4; i++) {
//        x = residualBlocks[i]->forward(x);
//    }
//    std::cout << "forwardX " << x->getInfo()->dim[0] << " " << x->getInfo()->dim[1] << " " << x->getInfo()->dim[2] << " " << x->getInfo()->dim[3] << std::endl;
//
//    // global avg pooling
//    x = _AvePool(x, {7, 7}, {1, 1}, VALID);
//
//    // x = _Convert(x, NCHW);
//    // x = _Reshape(x, {0, -1});
//    std::cout << "forward3 " << x->getInfo()->dim[0] << " " << x->getInfo()->dim[1] << " " << x->getInfo()->dim[2] << " " << x->getInfo()->dim[3] << std::endl;
//
//    x = fc->forward(x);
//    std::cout << "forward4 " << x->getInfo()->dim[0] << " " << x->getInfo()->dim[1] << " " << x->getInfo()->dim[2] << " " << x->getInfo()->dim[3] << std::endl;
//    x = _Softmax(x, -1);
//    return {x};
//}

} // namespace Model
} // namespace Train
} // namespace MNN
