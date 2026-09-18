#include "mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QFile>
#include <exception>
#include <c10/cuda/CUDACachingAllocator.h>

MainWindow::MainWindow(QWidget *parent)
    : QWidget{parent}
{
    this->resize(1200,700);
    layer_id=0;

    monitor=new Monitor();
    button1=new QPushButton(this);
    button1->setText("资源监视器");
    connect(button1,&QPushButton::clicked,this,[this](){
        this->monitor->show();
    });

    mainlayout=new QVBoxLayout(this);
    mainlayout->addWidget(button1);

    hsplitter1=new QSplitter(Qt::Horizontal);
    mainlayout->addWidget(hsplitter1);
    widget1=new QWidget(this);
    widget2=new QWidget(this);
    widget3=new QWidget(this);
    hsplitter1->addWidget(widget1);
    hsplitter1->addWidget(widget2);
    hsplitter1->addWidget(widget3);
    hsplitter1->setStretchFactor(0, 1);
    hsplitter1->setStretchFactor(1, 2);
    hsplitter1->setStretchFactor(2, 2);

    //========widget1========
    vlayout1_1 = new QVBoxLayout(widget1);
    layerList = new QListWidget(widget1);
    vlayout1_1->addWidget(layerList);

    addLayerBtn = new QPushButton("加层");
    removeLayerBtn = new QPushButton("删层");
    hlayout1_1 = new QHBoxLayout();
    hlayout1_1->addWidget(addLayerBtn);
    hlayout1_1->addWidget(removeLayerBtn);
    vlayout1_1->addLayout(hlayout1_1);

    applyBtn = new QPushButton("应用");
    refreshBtn = new QPushButton("刷新");
    hlayout1_2 = new QHBoxLayout();
    hlayout1_2->addWidget(applyBtn);
    hlayout1_2->addWidget(refreshBtn);
    vlayout1_1->addLayout(hlayout1_2);

    importBtn = new QPushButton("导入");
    exportBtn = new QPushButton("导出");
    hideBtn = new QPushButton("隐藏详情");
    hlayout1_3 = new QHBoxLayout();
    hlayout1_3->addWidget(importBtn);
    hlayout1_3->addWidget(exportBtn);
    hlayout1_3->addWidget(hideBtn);
    vlayout1_1->addLayout(hlayout1_3);

    connect(addLayerBtn, &QPushButton::clicked, this, &MainWindow::onAddLayer);
    connect(removeLayerBtn, &QPushButton::clicked, this, &MainWindow::onRemoveLayer);
    connect(applyBtn, &QPushButton::clicked, this, &MainWindow::onApply);
    connect(refreshBtn, &QPushButton::clicked, this, &MainWindow::onRefresh);
    connect(importBtn, &QPushButton::clicked, this, &MainWindow::onImportJson);
    connect(exportBtn, &QPushButton::clicked, this, &MainWindow::onExportJson);

    //========widget2========
    vlayout2_1 = new QVBoxLayout(widget2);
    detailTop = new QWidget(widget2);
    hlayout2_top = new QHBoxLayout(detailTop);
    hlayout2_top->setContentsMargins(0, 0, 0, 0);
    nameEdit = new QLineEdit(detailTop);
    typeCombo = new QComboBox(detailTop);
    typeCombo->addItem("卷积 Conv2d", "conv2d");
    typeCombo->addItem("最大池化 MaxPool2d", "maxpool2d");
    typeCombo->addItem("全连接 Linear", "linear");
    typeCombo->addItem("ReLU", "relu");
    typeCombo->addItem("展平 Flatten", "flatten");
    typeCombo->addItem("Dropout", "dropout");
    hlayout2_top->addWidget(nameEdit, 1);
    hlayout2_top->addWidget(typeCombo);
    vlayout2_1->addWidget(detailTop);

    detailStack = new QStackedWidget(widget2);
    vlayout2_1->addWidget(detailStack);

    inputPage = new QWidget();
    convPage = new QWidget();
    poolPage = new QWidget();
    linearPage = new QWidget();
    dropoutPage = new QWidget();

    {   // 输入页
        auto* form = new QFormLayout(inputPage);

        inputC = new QSpinBox();
        inputC->setRange(1, 4096);
        inputC->setValue(3);

        inputH = new QSpinBox();
        inputH->setRange(1, 4096);
        inputH->setValue(32);

        inputW = new QSpinBox();
        inputW->setRange(1, 4096);
        inputW->setValue(32);

        form->addRow("通道 C", inputC);
        form->addRow("高 H", inputH);
        form->addRow("宽 W", inputW);
    }

    {   // conv2d 页
        auto* form = new QFormLayout(convPage);

        convOut = new QSpinBox();
        convOut->setRange(1, 4096);

        convKernel = new QSpinBox();
        convKernel->setRange(1, 31);

        convStride = new QSpinBox();
        convStride->setRange(1, 8);

        convPadding = new QSpinBox();
        convPadding->setRange(0, 16);

        convDilation = new QSpinBox();
        convDilation->setRange(1, 8);

        form->addRow("输出通道 out_channels", convOut);
        form->addRow("卷积核 kernel", convKernel);
        form->addRow("步长 stride", convStride);
        form->addRow("填充 padding", convPadding);
        form->addRow("膨胀 dilation", convDilation);
    }

    {   // maxpool2d 页
        auto* form = new QFormLayout(poolPage);

        poolKernel = new QSpinBox();
        poolKernel->setRange(1, 31);

        poolStride = new QSpinBox();
        poolStride->setRange(1, 8);

        poolPadding = new QSpinBox();
        poolPadding->setRange(0, 16);

        form->addRow("核 kernel", poolKernel);
        form->addRow("步长 stride", poolStride);
        form->addRow("填充 padding", poolPadding);
    }

    {   // linear 页
        auto* form = new QFormLayout(linearPage);

        linearOut = new QSpinBox();
        linearOut->setRange(1, 65536);

        form->addRow("输出 out_features", linearOut);
    }

    {   // dropout 页
        auto* form = new QFormLayout(dropoutPage);

        dropoutP = new QDoubleSpinBox();
        dropoutP->setRange(0.0, 1.0);
        dropoutP->setSingleStep(0.05);
        dropoutP->setValue(0.5);

        form->addRow("概率 p", dropoutP);
    }

    {   // relu 页 / flatten 页：无参数，放一行说明
        reluPage = new QWidget();
        auto* l1 = new QVBoxLayout(reluPage);
        l1->addWidget(new QLabel("ReLU：无参数"));
        flattenPage = new QWidget();
        auto* l2 = new QVBoxLayout(flattenPage);
        l2->addWidget(new QLabel("Flatten：从 dim=1 展平，无参数"));
    }

    detailStack->addWidget(inputPage);
    detailStack->addWidget(convPage);
    detailStack->addWidget(poolPage);
    detailStack->addWidget(linearPage);
    detailStack->addWidget(dropoutPage);
    detailStack->addWidget(reluPage);
    detailStack->addWidget(flattenPage);

    // 主从联动 + 写回
    connect(layerList, &QListWidget::currentRowChanged, this, &MainWindow::onLayerSelected);
    connect(nameEdit, &QLineEdit::textChanged, this, &MainWindow::onNameChanged);
    connect(typeCombo, &QComboBox::currentIndexChanged, this, &MainWindow::onTypeChanged);
    connect(hideBtn, &QPushButton::clicked, this, &MainWindow::onToggleWidget2);

    connect(inputC, &QSpinBox::valueChanged, this, [this](int v){
        setInputShapeField(0, v);
    });
    connect(inputH, &QSpinBox::valueChanged, this, [this](int v){
        setInputShapeField(1, v);
    });
    connect(inputW, &QSpinBox::valueChanged, this, [this](int v){
        setInputShapeField(2, v);
    });

    connect(convOut, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("out_channels", v);
    });
    connect(convKernel, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("kernel", v);
    });
    connect(convStride, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("stride", v);
    });
    connect(convPadding, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("padding", v);
    });
    connect(convDilation, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("dilation", v);
    });

    connect(poolKernel, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("kernel", v);
    });
    connect(poolStride, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("stride", v);
    });
    connect(poolPadding, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("padding", v);
    });

    connect(linearOut, &QSpinBox::valueChanged, this, [this](int v){
        setLayerField("out_features", v);
    });
    connect(dropoutP, &QDoubleSpinBox::valueChanged, this, [this](double v){
        setLayerField("p", v);
    });

    //========widget3========
    vlayout3_1 = new QVBoxLayout(widget3);
    vsplitter3 = new QSplitter(Qt::Vertical);
    vlayout3_1->addWidget(vsplitter3);

    trainPanel = new QWidget();
    trainLayout = new QVBoxLayout(trainPanel);

    // 数据集
    formatCombo = new QComboBox();
    formatCombo->addItem("文件夹（每类一子目录）", false);
    formatCombo->addItem("CSV（path,label）", true);
    loadDatasetBtn = new QPushButton("加载数据集");
    datasetLabel = new QLabel("未加载");
    randomDataBtn = new QPushButton("生成随机数据");
    {
        auto* h = new QHBoxLayout();
        h->addWidget(formatCombo);
        h->addWidget(loadDatasetBtn);
        trainLayout->addLayout(h);
        auto* h2 = new QHBoxLayout();
        h2->addWidget(datasetLabel, 1);
        h2->addWidget(randomDataBtn);
        trainLayout->addLayout(h2);
    }

    // 训练参数
    deviceCombo = new QComboBox();
    deviceCombo->addItem("cpu");
    if (torch::cuda::is_available()){
        for (int i = 0; i < torch::cuda::device_count(); ++i){
            deviceCombo->addItem(QString("cuda:%1").arg(i));
        }
    }

    batchSpin = new QSpinBox();
    batchSpin->setRange(1, 4096);
    batchSpin->setValue(32);

    lrSpin = new QDoubleSpinBox();
    lrSpin->setRange(1e-6, 1.0);
    lrSpin->setDecimals(6);
    lrSpin->setSingleStep(0.0001);
    lrSpin->setValue(0.001);

    epochSpin = new QSpinBox();
    epochSpin->setRange(-1, 100000);
    epochSpin->setValue(10);

    {
        auto* form = new QFormLayout();
        form->addRow("设备", deviceCombo);
        form->addRow("batch", batchSpin);
        form->addRow("lr", lrSpin);
        form->addRow("epoch", epochSpin);
        trainLayout->addLayout(form);
    }

    // 权重
    saveWeightsBtn = new QPushButton("保存权重");
    loadWeightsBtn = new QPushButton("导入权重");
    {
        auto* h = new QHBoxLayout();
        h->addWidget(saveWeightsBtn);
        h->addWidget(loadWeightsBtn);
        trainLayout->addLayout(h);
    }

    vsplitter3->addWidget(trainPanel);

    startBtn = new QPushButton("开始");
    pauseBtn = new QPushButton("暂停");
    abortBtn = new QPushButton("急停");
    {
        auto* h = new QHBoxLayout();
        h->addWidget(startBtn);
        h->addWidget(pauseBtn);
        h->addWidget(abortBtn);
        trainLayout->addLayout(h);
    }
    pauseBtn->setEnabled(false);
    abortBtn->setEnabled(false);

    connect(startBtn, &QPushButton::clicked, this, &MainWindow::onStart);
    connect(pauseBtn, &QPushButton::clicked, this, &MainWindow::onPause);
    connect(abortBtn, &QPushButton::clicked, this, &MainWindow::onAbort);

    // 下半：loss 曲线（y 自动缩放：地板 1，天花板 0）
    lossPlot = new Plot_Widget(200, nullptr, 1, 0);
    vsplitter3->addWidget(lossPlot);

    connect(loadDatasetBtn, &QPushButton::clicked, this, &MainWindow::onLoadDataset);
    connect(randomDataBtn, &QPushButton::clicked, this, &MainWindow::onRandomData);
    connect(deviceCombo, &QComboBox::currentIndexChanged, this, &MainWindow::onDeviceChanged);
    connect(lrSpin, &QDoubleSpinBox::valueChanged, this, &MainWindow::onLrChanged);
    connect(saveWeightsBtn, &QPushButton::clicked, this, &MainWindow::onSaveWeights);
    connect(loadWeightsBtn, &QPushButton::clicked, this, &MainWindow::onLoadWeights);

    initDefaultJson();
    rebuildLayerList(0);
};

MainWindow::~MainWindow(){
    if (!randomDir_.isEmpty()){
        QDir(randomDir_).removeRecursively();
    }
    monitor->deleteLater();
}

void MainWindow::closeEvent(QCloseEvent *event){
    if (worker) {
        worker->abort();
    }
    if (thread) {
        thread->quit();
        thread->wait();
    }
    monitor->close();
    event->accept();
}

void MainWindow::initDefaultJson() {
    QJsonObject optimizer;
    optimizer["type"] = "adam";
    optimizer["lr"] = 0.001;
    QJsonObject root;
    root["input_shape"] = QJsonArray{-1, 3, 32, 32};
    root["optimizer"] = optimizer;
    root["layers"] = QJsonArray{};
    json_ = root;
}

void MainWindow::rebuildLayerList(int selectRow) {
    layerList->blockSignals(true);
    layerList->clear();
    layerList->addItem(QString("输入 (%1)").arg(currentInputShapeText()));
    for (const auto& v : json_["layers"].toArray()) {
        auto o = v.toObject();
        layerList->addItem(QString("%1(%2)").arg(o["name"].toString(), o["type"].toString()));
    }
    layerList->blockSignals(false);
    int n = layerList->count();
    if (selectRow < 0) {
        selectRow = 0;
    }
    if (selectRow >= n) {
        selectRow = n - 1;
    }
    layerList->setCurrentRow(selectRow);
}

QString MainWindow::currentInputShapeText() const {
    auto s = json_["input_shape"].toArray();
    return QString("%1×%2×%3").arg(s[1].toInt()).arg(s[2].toInt()).arg(s[3].toInt());
}

int MainWindow::currentLayerIndex() const {
    int row = layerList->currentRow();
    return (row <= 0) ? -1 : row - 1;
}

QJsonObject MainWindow::currentLayerJson() const {
    int li = currentLayerIndex();
    if (li < 0) {
        return QJsonObject{};
    }
    return json_["layers"].toArray()[li].toObject();
}

QJsonObject MainWindow::defaultLayerForType(const QString& type) {
    QJsonObject o;
    o["type"] = type;
    if (type == "conv2d") {
        o["out_channels"] = 16;
        o["kernel"] = 3;
        o["stride"] = 1;
        o["padding"] = 1;
        o["dilation"] = 1;
    }
    else if (type == "maxpool2d") {
        o["kernel"] = 2;
        o["stride"] = 2;
        o["padding"] = 0;
    }
    else if (type == "linear") {
        o["out_features"] = 10;
    }
    else if (type == "dropout") {
        o["p"] = 0.5;
    }
    // relu / flatten 无参数
    return o;
}

void MainWindow::applyJsonToModel() {
    try {
        apply_json_to_model(json_, model_);
        modelApplied_ = true;
        trainedEpochs_ = 0;
        QMessageBox::information(this, "提示", "模型已应用");
    } catch (const std::exception& e) {
        modelApplied_ = false;
        QMessageBox::critical(this, "应用失败", e.what());
    }
}

// ============ widget1 按钮 ============

void MainWindow::onAddLayer() {
    auto arr = json_["layers"].toArray();
    QJsonObject o = defaultLayerForType("linear");
    o["name"] = QString("linear_%1").arg(layer_id++);
    arr.append(o);
    json_["layers"] = arr;
    rebuildLayerList(arr.size());   // 新层在列表第 arr.size() 行（第 0 行是输入）
}

void MainWindow::onRemoveLayer() {
    int li = currentLayerIndex();
    if (li < 0) {
        QMessageBox::information(this, "提示", "输入层不能删除");
        return;
    }
    auto arr = json_["layers"].toArray();
    arr.removeAt(li);
    json_["layers"] = arr;
    rebuildLayerList(li + 1);   // 选中原下一层
}

void MainWindow::onApply() { applyJsonToModel(); }

void MainWindow::onRefresh() {
    if (!modelApplied_) {
        QMessageBox::information(this, "提示", "模型尚未应用，无可刷新的内容");
        return;
    }
    json_ = model_to_json(model_);
    rebuildLayerList(layerList->currentRow());
}

void MainWindow::onImportJson() {
    QString path = QFileDialog::getOpenFileName(this, "导入模型结构", QString(), "JSON (*.json)");
    if (path.isEmpty()) {
        return;
    }
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "错误", "无法打开文件");
        return;
    }
    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        QMessageBox::critical(this, "错误", "JSON 解析失败: " + err.errorString());
        return;
    }
    QJsonObject root = doc.object();
    if (!root.contains("input_shape") || !root.contains("layers")) {
        QMessageBox::critical(this, "错误", "JSON 缺少 input_shape 或 layers");
        return;
    }
    json_ = root;
    modelApplied_ = false;
    rebuildLayerList(0);
}

void MainWindow::onExportJson() {
    QString path = QFileDialog::getSaveFileName(this, "导出模型结构", QString(), "JSON (*.json)");
    if (path.isEmpty()) {
        return;
    }
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "错误", "无法写入文件");
        return;
    }
    f.write(QJsonDocument(json_).toJson());
}

void MainWindow::onLayerSelected(int row) {
    if (row < 0) {
        return;
    }
    loadDetailPage();
}

void MainWindow::loadDetailPage() {
    int row = layerList->currentRow();
    if (row < 0) {
        return;
    }
    updatingDetail_ = true;

    bool isInput = (row == 0);
    nameEdit->setEnabled(!isInput);
    typeCombo->setEnabled(!isInput);

    if (isInput) {
        showLayerPage("input");
        auto s = json_["input_shape"].toArray();
        inputC->setValue(s.size() > 1 ? s[1].toInt() : 3);
        inputH->setValue(s.size() > 2 ? s[2].toInt() : 32);
        inputW->setValue(s.size() > 3 ? s[3].toInt() : 32);
        updatingDetail_ = false;
        return;
    }

    auto o = json_["layers"].toArray()[row - 1].toObject();
    QString type = o["type"].toString();
    nameEdit->setText(o["name"].toString());
    int ti = typeCombo->findData(type);
    typeCombo->setCurrentIndex(ti < 0 ? 0 : ti);
    showLayerPage(type);

    if (type == "conv2d") {
        convOut->setValue(o["out_channels"].toInt());
        convKernel->setValue(o["kernel"].toInt(3));
        convStride->setValue(o["stride"].toInt(1));
        convPadding->setValue(o["padding"].toInt(0));
        convDilation->setValue(o["dilation"].toInt(1));
    }
    else if (type == "maxpool2d") {
        poolKernel->setValue(o["kernel"].toInt(2));
        poolStride->setValue(o["stride"].toInt(2));
        poolPadding->setValue(o["padding"].toInt(0));
    }
    else if (type == "linear") {
        linearOut->setValue(o["out_features"].toInt());
    }
    else if (type == "dropout") {
        dropoutP->setValue(o["p"].toDouble(0.5));
    }

    updatingDetail_ = false;
}

void MainWindow::showLayerPage(const QString& type) {
    if (type == "conv2d"){
        detailStack->setCurrentWidget(convPage);
    }
    else if (type == "maxpool2d"){
        detailStack->setCurrentWidget(poolPage);
    }
    else if (type == "linear"){
        detailStack->setCurrentWidget(linearPage);
    }
    else if (type == "dropout"){
        detailStack->setCurrentWidget(dropoutPage);
    }
    else if (type == "relu"){
        detailStack->setCurrentWidget(reluPage);
    }
    else if (type == "flatten"){
        detailStack->setCurrentWidget(flattenPage);
    }
    else{
        detailStack->setCurrentWidget(inputPage);
    }
}

void MainWindow::setLayerField(const QString& key, QJsonValue v) {
    if (updatingDetail_) {
        return;
    }
    int li = currentLayerIndex();
    if (li < 0) {
        return;
    }
    auto arr = json_["layers"].toArray();
    auto o = arr[li].toObject();
    o[key] = v;
    arr[li] = o;
    json_["layers"] = arr;
}

void MainWindow::setInputShapeField(int idx, int v) {
    if (updatingDetail_) {
        return;
    }
    auto s = json_["input_shape"].toArray();
    s[idx + 1] = v;
    json_["input_shape"] = s;
    layerList->item(0)->setText(QString("输入 (%1)").arg(currentInputShapeText()));
}

void MainWindow::onNameChanged(const QString& text) {
    if (updatingDetail_) {
        return;
    }
    int li = currentLayerIndex();
    if (li < 0) {
        return;
    }
    auto arr = json_["layers"].toArray();
    auto o = arr[li].toObject();
    o["name"] = text;
    arr[li] = o;
    json_["layers"] = arr;
    layerList->item(li + 1)->setText(QString("%1(%2)").arg(text, o["type"].toString()));
}

void MainWindow::onTypeChanged(int idx) {
    if (updatingDetail_) {
        return;
    }
    int li = currentLayerIndex();
    if (li < 0) {
        return;
    }
    QString type = typeCombo->itemData(idx).toString();
    QJsonObject o = defaultLayerForType(type);
    o["name"] = nameEdit->text();
    auto arr = json_["layers"].toArray();
    arr[li] = o;
    json_["layers"] = arr;
    layerList->item(li + 1)->setText(QString("%1(%2)").arg(o["name"].toString(), type));
    loadDetailPage();   // 切页 + 回填新默认值
}

void MainWindow::onToggleWidget2() {
    widget2->setVisible(!widget2->isVisible());
    hideBtn->setText(widget2->isVisible() ? "隐藏详情" : "显示详情");
}

void MainWindow::onLoadDataset() {
    bool is_csv = formatCombo->currentData().toBool();
    QString src;
    if (is_csv){
        src = QFileDialog::getOpenFileName(this, "选择 CSV", QString(), "CSV (*.csv)");
    }
    else{
        src = QFileDialog::getExistingDirectory(this, "选择数据集根目录");
    }
    if (src.isEmpty()){
        return;
    }

    auto s = json_["input_shape"].toArray();   // C/H/W 取「输入」页当前值
    int C = s[1].toInt(), H = s[2].toInt(), W = s[3].toInt();
    try {
        dataset_ = load_dataset(src.toStdWString(), is_csv);
        datasetLabel->setText(QString("已加载 %1 样本 / %2 类").arg(dataset_.size()).arg(dataset_.classes));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", e.what());
    }
}

void MainWindow::onRandomData() {
    auto s = json_["input_shape"].toArray();
    int C = s[1].toInt(), H = s[2].toInt(), W = s[3].toInt();

    if (!randomDir_.isEmpty()){
        QDir(randomDir_).removeRecursively();
    }
    QString dir = QDir::tempPath() + "/EasyTrain_rand_" + QUuid::createUuid().toString(QUuid::WithoutBraces);
    QDir().mkpath(dir);

    try {
        dataset_ = random_dataset(dir.toStdWString(), 256, C, H, W, 10);
        randomDir_ = dir;
        datasetLabel->setText(QString("随机数据 %1 样本 / 10 类").arg(dataset_.size()));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", e.what());
    }
}

void MainWindow::onDeviceChanged(int) {
    model_.to(currentDevice());
}

void MainWindow::onLrChanged(double v) {
    model_.setLearningRate(v);
}

torch::Device MainWindow::currentDevice() const {
    QString s = deviceCombo->currentText();
    if (s == "cpu") {
        return torch::kCPU;
    }
    return torch::Device(s.toStdString());
}

void MainWindow::onSaveWeights() {
    if (!ensureModelReady()) {
        return;
    }
    QString path = QFileDialog::getSaveFileName(this, "保存权重", QString(), "PyTorch 权重 (*.pt)");
    if (path.isEmpty()) {
        return;
    }
    try {
        model_.saveWeights(path.toStdString());
        QMessageBox::information(this, "提示", "权重已保存");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", e.what());
    }
}

void MainWindow::onLoadWeights() {
    if (!ensureModelReady()) {
        return;
    }
    QString path = QFileDialog::getOpenFileName(this, "导入权重", QString(), "PyTorch 权重 (*.pt)");
    if (path.isEmpty()) {
        return;
    }
    try {
        model_.loadWeights(path.toStdString());   // 结构与保存时不一致会 throw（key 不匹配）
        trainedEpochs_ = 0;
        QMessageBox::information(this, "提示", "权重已导入");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", e.what());
    }
}

bool MainWindow::ensureDatasetReady() {
    if (dataset_.paths_.empty()) {
        QMessageBox::warning(this, "提示", "请先加载数据集（或生成随机数据）");
        return false;
    }
    return true;
}

bool MainWindow::ensureModelReady() {
    if (!modelApplied_) {
        QMessageBox::warning(this, "提示", "请先点「应用」构建模型");
        return false;
    }
    return true;
}

void MainWindow::onStart() {
    if (!ensureDatasetReady() || !ensureModelReady()) {
        return;
    }

    auto s = json_["input_shape"].toArray();
    int C = s[1].toInt(), H = s[2].toInt(), W = s[3].toInt();

    worker = new TrainingWorker(&model_, &dataset_, C, H, W, aug_, currentDevice(),batchSpin->value(), trainedEpochs_, epochSpin->value());
    thread = new QThread(this);
    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, &TrainingWorker::run);
    connect(worker, &TrainingWorker::lossReady, this, &MainWindow::onLoss);
    connect(worker, &TrainingWorker::epochDone, this, &MainWindow::onEpochDone);
    connect(worker, &TrainingWorker::finished, this, &MainWindow::onTrainingFinished);
    connect(worker, &TrainingWorker::finished, thread, &QThread::quit);
    connect(worker, &TrainingWorker::finished, worker, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    thread->start();
    setTrainingUI(true);
    datasetLabel->setText("训练中…");
}

void MainWindow::onPause() {
    if (worker) {
        worker->pause();
    }
}   // 跑完当前 epoch 停
void MainWindow::onAbort() {
    if (worker) {
        worker->abort();
    }
}   // 跑完当前 batch 停

void MainWindow::onLoss(float loss) {
    lossPlot->add_point(loss);
}

void MainWindow::onEpochDone(int epoch) {
    trainedEpochs_ = epoch + 1;
    int target = epochSpin->value();
    QString t = (target < 0) ? QString::fromUtf8("∞") : QString::number(target);
    datasetLabel->setText(QString("训练中：%1/%2").arg(trainedEpochs_).arg(t));
}

void MainWindow::onTrainingFinished() {
    setTrainingUI(false);
    worker = nullptr;
    thread = nullptr;
    datasetLabel->setText("训练结束");

    if (currentDevice().is_cuda()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

void MainWindow::setTrainingUI(bool training) {
    addLayerBtn->setEnabled(!training);
    removeLayerBtn->setEnabled(!training);
    applyBtn->setEnabled(!training);
    refreshBtn->setEnabled(!training);
    importBtn->setEnabled(!training);
    exportBtn->setEnabled(!training);
    loadDatasetBtn->setEnabled(!training);
    randomDataBtn->setEnabled(!training);
    deviceCombo->setEnabled(!training);
    saveWeightsBtn->setEnabled(!training);
    loadWeightsBtn->setEnabled(!training);
    startBtn->setEnabled(!training);
    pauseBtn->setEnabled(training);
    abortBtn->setEnabled(training);
}
