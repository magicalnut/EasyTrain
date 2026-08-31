#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QWidget{parent}
{
    this->resize(600,600);
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

    vlayout1_1=new QVBoxLayout(widget1);
    widget1->setLayout(vlayout1_1);
    hlayout1_1=new QHBoxLayout();
    vlayout1_1->addLayout(hlayout1_1);

    vlayout3_1=new QVBoxLayout(widget3);
    widget3->setLayout(vlayout3_1);
};

MainWindow::~MainWindow(){
    monitor->deleteLater();
}

void MainWindow::closeEvent(QCloseEvent *event){
    monitor->close();
    event->accept();
}