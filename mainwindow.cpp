#include "mainwindow.h"

MainWindow::MainWindow(Monitor *monitor,QWidget *parent)
    : QWidget{parent},monitor(monitor)
{
    this->resize(600,600);

    button1=new QPushButton(this);
    button1->setText("资源监视器");
    connect(button1,&QPushButton::clicked,this,[this](){
        this->monitor->show();
    });
    button2=new QPushButton(this);
    button2->setText("测试");
    connect(button2,&QPushButton::clicked,this,[this](){
        test=new Test();
        delete test;
    });

    mainlayout=new QVBoxLayout(this);
    mainlayout->addWidget(button1);
    mainlayout->addWidget(button2);
};

MainWindow::~MainWindow(){
}