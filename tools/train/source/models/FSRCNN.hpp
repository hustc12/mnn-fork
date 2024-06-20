#ifndef FSRCNN_hpp
#define FSRCNN_hpp

#include "Initializer.hpp"
#include <vector>
// #include "MobilenetUtils.hpp"
#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

       

class MNN_PUBLIC FSRCNN : public Express::Module {
public:
    // use tensorflow numClasses = 1001, which label 0 means outlier of the original 1000 classes
    // so you maybe need to add 1 to your true labels, if you are testing with ImageNet dataset
    FSRCNN(int num_channels = 1, int d=56, int s=12, int m=4, int upscale_factor=3);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    
    std::shared_ptr<Express::Module> feature_extraction;
    std::shared_ptr<Express::Module> shrinking;
    std::vector<std::shared_ptr<Express::Module>> mapping;
    std::shared_ptr<Express::Module> expanding;
    std::shared_ptr<Express::Module> deconvolution;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // FSRCNN_hpp
