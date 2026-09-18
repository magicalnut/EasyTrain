#include "model.h"
#include "layer_factory.h"

#include <torch/torch.h>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

struct Impl : torch::nn::Module {
    explicit Impl(const std::vector<LayerConfig>& filled) {
        for (const auto& L : filled) {
            auto m = create_layer(L);               // in_channels 已回填，.value() 安全
            register_module(L.name, m.ptr());       // 注册：to/parameters/train 才能递归
            mods_.emplace(L.name, std::move(m));    // 存 AnyModule 给 forward 调用
            topo_.push_back(L.name);
        }
    }

    torch::Tensor forward(torch::Tensor x) {        // 不 override（Module 无虚 forward）
        for (const auto& n : topo_)
            x = mods_.at(n).forward(x);
        return x;
    }

private:
    std::vector<std::string> topo_;
    std::unordered_map<std::string, torch::nn::AnyModule> mods_;
};

// ============ 优化器工厂 ============
namespace {
std::unique_ptr<torch::optim::Optimizer> makeOptimizer(
    const OptimizerConfig& cfg, std::vector<torch::Tensor> params) {
    if (cfg.type == "adam")
        return std::make_unique<torch::optim::Adam>(
            std::move(params), torch::optim::AdamOptions(cfg.lr));
    // case "sgd":   return std::make_unique<torch::optim::SGD>(std::move(params), torch::optim::SGDOptions(cfg.lr));
    // case "adamw": return std::make_unique<torch::optim::AdamW>(...);
    throw std::runtime_error("未知优化器类型: " + cfg.type);
}
}

Model::Model() = default;
Model::Model(const Shape& input) : input_(input) {}
Model::~Model() = default;

// —— 结构 ——
void Model::setInputShape(const Shape& input) {
    input_ = input;
    dirty_ = true;
}

void Model::addLayer(const LayerConfig& layer) {
    layers_.push_back(layer);
    dirty_ = true;
}

void Model::clear() {
    layers_.clear();
    dirty_ = true;
}

void Model::ensureBuilt() { if (dirty_) build(); }

void Model::build() {
    if (!dirty_) return;

    auto filled = layers_;               // 副本：推断回填在副本上，真相源保持 nullopt
    infer_shapes(input_, filled);        // 承重墙：形状不合法直接 throw，UI 捕获显示

    impl_ = std::make_shared<Impl>(filled);
    optimizer_.reset();                  // 参数换了，旧优化器作废，下次 fit 重建
    impl_->to(device_);                  // 新建模块默认 CPU，补回设备
    dirty_ = false;
}

// 读回配置（UI「刷新」用）
const Shape& Model::inputShape() const {
    return input_;
}

const std::vector<LayerConfig>& Model::layers() const {
    return layers_;
}

const OptimizerConfig& Model::optimizerConfig() const {
    return opt_cfg_;
}

// —— 前向 ——
torch::Tensor Model::forward(torch::Tensor x) {
    ensureBuilt();
    return impl_->forward(x);
}

// —— 训练 ——
void Model::ensureOptimizer() {
    ensureBuilt();
    if (!optimizer_) {
        lr_.store(opt_cfg_.lr);          // 首次：把配置初始 lr 灌进原子量
        optimizer_ = makeOptimizer(opt_cfg_, impl_->parameters());
    }
}

void Model::applyLearningRate() {
    double lr = lr_.load();
    for (auto& g : optimizer_->param_groups())
        g.options().set_lr(lr);   // OptimizerOptions 有虚 set_lr，各优化器已 override，无需 downcast
}

// float Model::fit(const torch::Tensor& x, const torch::Tensor& y, int batch,const std::atomic<bool>* stop) {
//     ensureBuilt();
//     ensureOptimizer();
//     applyLearningRate();                 // 训练中改 lr → 下个 epoch 生效
//     impl_->train();

//     int64_t n = x.size(0);
//     if (n <= 0) return 0.0f;
//     if (y.size(0) != n) throw std::runtime_error("fit: x/y 样本数不一致");
//     if (batch <= 0) batch = 32;

//     torch::nn::CrossEntropyLoss loss_fn;

//     // 洗牌（索引放到 x 同设备，避免 device mismatch）
//     auto perm = torch::randperm(n, torch::TensorOptions().dtype(torch::kLong).device(x.device()));
//     auto xs = x.index_select(0, perm);
//     auto ys = y.index_select(0, perm);

//     double total = 0.0;
//     int64_t seen = 0;
//     for (int64_t i = 0; i < n; i += batch) {
//         if (stop && stop->load()) break;   // 急停
//         int64_t e = std::min<int64_t>(i + batch, n);
//         auto xb = xs.slice(0, i, e);
//         auto yb = ys.slice(0, i, e);

//         optimizer_->zero_grad();
//         auto loss = loss_fn(impl_->forward(xb), yb);
//         loss.backward();
//         optimizer_->step();

//         total += loss.item().toFloat() * (e - i);   // item<float>() 已废弃，用 item().toFloat()
//         seen += (e - i);
//     }
//     return static_cast<float>(total / seen);
// }

void Model::setOptimizer(const OptimizerConfig& cfg) {
    opt_cfg_ = cfg;
    lr_.store(cfg.lr);
    optimizer_.reset();                  // 下次 fit 按新配置重建（不重建模块）
}

void Model::setLearningRate(double lr) {
    lr_.store(lr);
    opt_cfg_.lr = lr;                    // 同步回配置，便于保存/查看
}

double Model::learningRate() const { return lr_.load(); }

// —— 模式 / 设备 ——
void Model::to(torch::Device device) {
    device_ = device;
    if (impl_) impl_->to(device);
}
void Model::train() { if (impl_) impl_->train(); }
void Model::eval()  { if (impl_) impl_->eval(); }

// —— 持久化 ——
void Model::saveWeights(const std::string& path) const {
    if (!impl_) throw std::runtime_error("saveWeights: 模型尚未 build");
    torch::save(impl_, path);
}
void Model::loadWeights(const std::string& path) {
    if (!impl_) throw std::runtime_error("loadWeights: 模型尚未 build");
    torch::load(impl_, path);
    optimizer_.reset();
}

void Model::prepareTraining() {
    ensureBuilt();
    ensureOptimizer();
    applyLearningRate();
    impl_->train();
}

float Model::trainBatch(const torch::Tensor& xb, const torch::Tensor& yb,torch::nn::CrossEntropyLoss& loss_fn) {
    optimizer_->zero_grad();
    auto loss = loss_fn(impl_->forward(xb), yb);
    loss.backward();
    optimizer_->step();
    return loss.item().toFloat();
}
