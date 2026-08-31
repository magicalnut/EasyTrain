#include "plot_widget.h"

#include <QPainter>

Plot_Widget::Plot_Widget(int times, QWidget *parent, double ymax, double ymin)
    : QWidget{parent}, maxx(times), maxy(ymax), miny(ymin),
    maxy_floor(ymax), miny_floor(ymin)
{
    // label=new QLabel(this);
    // label->setScaledContents(true);
    // label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    // layout=new QVBoxLayout(this);
    // layout->addWidget(label);
    // setLayout(layout);

    // image=cv::Mat::zeros(this->height(),this->width(),CV_8UC3);
    if(maxx<2){
        maxx=2;
    }
    y=new std::list<double>(2,0);
}

Plot_Widget::~Plot_Widget(){
    delete y;
}

void Plot_Widget::paintEvent(QPaintEvent *event){
    double width=this->width();
    double height=this->height();
    itemx=width/double(y->size()-1);
    itemy=height/(maxy-miny);
    double zero_y=height-(0-miny)*itemy;

    QPainter painter(this);
    painter.setPen(QPen(Qt::blue,2));

    painter.drawLine(0,0,width,0);
    painter.drawLine(0,0,0,height);
    painter.drawLine(width,0,width,height);
    painter.drawLine(0,height,width,height);

    // image=cv::Mat::zeros(height,width,CV_8UC3);
    // image.setTo(cv::Scalar(255,255,255));

    // cv::line(image,
    //          cv::Point(0,0),cv::Point(width,0),
    //          cv::Scalar(255,0,0),2);
    // cv::line(image,
    //          cv::Point(0,0),cv::Point(0,height),
    //          cv::Scalar(255,0,0),2);
    // cv::line(image,
    //          cv::Point(width,0),cv::Point(width,height),
    //          cv::Scalar(255,0,0),2);
    // cv::line(image,
    //          cv::Point(0,height),cv::Point(width,height),
    //          cv::Scalar(255,0,0),2);

    painter.setPen(QPen(Qt::blue,1));
    for(int i=1;i*10.0<=maxy;i++){
        double tempy=zero_y-i*10.0*itemy;
        // cv::line(image,
        //          cv::Point(0,tempy),cv::Point(width,tempy),
        //          cv::Scalar(255,0,0),1);
        painter.drawLine(0,tempy,width,tempy);
    }
    for(int i=-1;i*10.0>=miny;i--){
        double tempy=zero_y-i*10.0*itemy;
        // cv::line(image,
        //          cv::Point(0,tempy),cv::Point(width,tempy),
        //          cv::Scalar(255,0,0),1);
        painter.drawLine(0,tempy,width,tempy);
    }
    // cv::line(image,
    //          cv::Point(0,zero_y),cv::Point(width,zero_y),
    //          cv::Scalar(0,0,0),2);
    painter.setPen(QPen(Qt::black,1));
    painter.drawLine(0,zero_y,width,zero_y);

    auto iy=y->begin();
    auto ny=iy;
    ny++;
    painter.setPen(QPen(Qt::black,2));
    for(int i=0;i<y->size()-1;i++){
        // cv::line(image,
        //          cv::Point(i*itemx,zero_y-*iy++*itemy),cv::Point((i+1)*itemx,zero_y-*ny++*itemy),
        //          cv::Scalar(0,0,0),1);
        painter.drawLine(i*itemx,zero_y-*iy++*itemy,(i+1)*itemx,zero_y-*ny++*itemy);
    }

    // QImage qimg(image.data, image.cols, image.rows, image.step, QImage::Format_BGR888);
    //label->setPixmap(QPixmap::fromImage(qimg));

    painter.end();
}

void Plot_Widget::add_point(double p){
    y->push_back(p);
    while(y->size()>maxx){
        y->pop_front();
    }
    auto it=y->begin();
    maxy=*it;
    miny=*it;
    it++;
    while(it!=y->end()){
        maxy=maxy>*it?maxy:*it;
        miny=miny<*it?miny:*it;
        it++;
    }
    maxy = maxy_floor > maxy ? maxy_floor : maxy;
    miny = miny_floor < miny ? miny_floor : miny;
    if (maxy <= miny) maxy = miny + 1.0;

    update();
}