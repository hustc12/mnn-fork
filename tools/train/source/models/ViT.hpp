#ifndef ViT_hpp
#define ViT_hpp

#include "Initializer.hpp"
#include <vector>
// #include "MobilenetUtils.hpp"
#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

       

class MNN_PUBLIC ViT : public Express::Module {
public:
    // use tensorflow numClasses = 1001, which label 0 means outlier of the original 1000 classes
    // so you maybe need to add 1 to your true labels, if you are testing with ImageNet dataset
    ViT(int numClasses = 1000, int patch_size=16, int num_layers=12, int num_heads=12, int hidden_dim=768, int mlp_dim=3072);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    
    std::shared_ptr<Express::Module> conv_proj;
    std::vector<std::shared_ptr<Express::Module> > encoder_layers;
//    std::shared_ptr<Express::Module> last_layer_norm;
    std::shared_ptr<Express::Module> linear;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // ViT_hpp
