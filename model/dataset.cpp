#include "dataset.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

// 读图走二进制流 + imdecode：Windows 中文路径不踩 cv::imread 的窄字符坑
cv::Mat read_image(const fs::path& p) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("读图失败: " + p.string());
    }
    std::vector<uchar> buf((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("解码失败: " + p.string());
    }
    return img;
}

bool is_image(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// 一张图 → resize 到 (W,H) → 转 C 通道 → [C,H,W] float 张量
torch::Tensor to_tensor(const cv::Mat& img, int C, int H, int W) {
    cv::Mat m;
    if (C == 1)       {
        cv::cvtColor(img, m, cv::COLOR_BGR2GRAY);
    }
    else if (C == 3)  {
        m = img;        // 已是 BGR 三通道
    }
    else{
        throw std::runtime_error("不支持的通道数 C（v1 只支持 1 或 3）");
    }

    cv::resize(m, m, cv::Size(W, H));
    m.convertTo(m, CV_32F, 1.0 / 255.0);

    torch::Tensor t;
    if (C == 1) {
        t = torch::from_blob(m.data, {H, W}, torch::kFloat32);
        t = t.unsqueeze(0).clone();                  // [1,H,W]；clone：from_blob 不拥有数据
    }
    else {
        t = torch::from_blob(m.data, {H, W, 3}, torch::kFloat32);
        t = t.permute({2, 0, 1}).clone();            // [3,H,W]
    }
    return t;
}

}

Dataset load_dataset(const std::wstring& src, int C, int H, int W, bool is_csv) {
    fs::path root(src);
    std::vector<torch::Tensor> xs;
    std::vector<int64_t> ys;
    int64_t nclasses = 0;

    if (!is_csv) {
        // 文件夹：每个子目录一个类，目录名按字典序编号 0..C-1
        std::vector<fs::path> class_dirs;
        for (const auto& e : fs::directory_iterator(root)){
            if (e.is_directory()) {
                class_dirs.push_back(e.path());
            }
        }
        std::sort(class_dirs.begin(), class_dirs.end());
        nclasses = (int64_t)class_dirs.size();
        if (nclasses == 0) {
            throw std::runtime_error("没有找到任何类别子目录");
        }

        for (int64_t c = 0; c < nclasses; ++c) {
            std::vector<fs::path> files;
            for (const auto& e : fs::directory_iterator(class_dirs[c])){
                if (e.is_regular_file() && is_image(e.path())) {
                    files.push_back(e.path());
                }
            }
            std::sort(files.begin(), files.end());
            for (const auto& f : files) {
                xs.push_back(to_tensor(read_image(f), C, H, W));
                ys.push_back(c);
            }
        }
    }
    else {
        // CSV：每行 path,label；整数当类索引，字符串按出现顺序映射
        std::ifstream in(root);
        if (!in) {
            throw std::runtime_error("打开 CSV 失败: " + root.string());
        }
        fs::path csv_dir = root.parent_path();
        std::map<std::string, int64_t> str2idx;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) {
                continue;
            }
            size_t comma = line.find(',');
            if (comma == std::string::npos) {
                continue;
            }
            std::string path  = trim(line.substr(0, comma));
            std::string label = trim(line.substr(comma + 1));
            if (path.empty() || label.empty()) {
                continue;
            }

            fs::path img_path(path);
            if (img_path.is_relative()) {
                img_path = csv_dir / img_path;   // 相对路径相对 CSV 所在目录
            }

            int64_t idx;
            bool numeric = std::all_of(label.begin(), label.end(),[](unsigned char c){
                return std::isdigit(c) || c == '-';
            });
            if (numeric) {
                idx = std::stoll(label);
            }
            else {
                auto it = str2idx.find(label);
                if (it == str2idx.end()) {
                    idx = (int64_t)str2idx.size();
                    str2idx[label] = idx;
                }
                else {
                    idx = it->second;
                }
            }
            xs.push_back(to_tensor(read_image(img_path), C, H, W));
            ys.push_back(idx);
        }
        nclasses = 0;
        for (auto y : ys) {
            nclasses = std::max(nclasses, y + 1);
        }
    }

    if (xs.empty()){
        throw std::runtime_error("数据集为空");
    }

    auto x = torch::stack(xs, 0);   // [N,C,H,W]
    auto y = torch::tensor(ys, torch::TensorOptions().dtype(torch::kLong));
    return Dataset{x, y, nclasses};
}

Dataset random_dataset(int N, int C, int H, int W, int classes) {
    auto x = torch::randn({N, C, H, W});
    auto y = torch::randint(0, classes, {N}, torch::TensorOptions().dtype(torch::kLong));
    return Dataset{x, y, classes};
}
