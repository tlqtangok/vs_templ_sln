// #define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#undef NDEBUG

#include <assert.h>

// #include <omp.h> // ok , can work

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <Windows.h>

#include <fmt/fmt_world.h>

#include <map>
#include <unordered_map>
#include <bitset>
#include <functional>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath> // 确保包含此头文件
#include <regex>
#include <rex.h>  // jd define regex

// #define PCRE2_CODE_UNIT_WIDTH 8
// #include <pcre2.h>   // .\vcpkg install pcre2:x64-windows && .\vcpkg integrate install 




#define byte uchar

using namespace std;
using namespace cv;

#include "com.h"
class com;

static com s_com;

using v3b = cv::Vec3b;
using v2b = cv::Vec2b;
using uchar3 = cv::Vec3b;
using uchar2 = cv::Vec2b;
using uchar4 = cv::Vec4b;




#define WIDTH_ALIGN(_w, _align) \
    ((_w % _align != 0) ? (_w / _align + 1) * _align : _w)
#define WIDTH_4(_w) WIDTH_ALIGN(_w, 4)

#define OF_W (ios::out | ios::trunc)

#if 0
template<typename T>
string to_string_(T n)
{
    ostringstream ss;
    ss << n;
    return ss.str();
}
#endif

#if 1
// cimg_ hpp start

class cimg
{
public:
    cv::Mat img;
    cv::Mat img_copy;

    std::vector<std::vector<cv::Point>> v_contours;
    cv::Mat img_contour_mask;
    cv::Mat img_contour_line;

    void normalized();

    void read_img(string fn);
    void read_img_gif(const std::string& filePath, int frameIndex = 0);
    // void cvtcolor();
    void cvtcolor(string colormode = "GRAY");
    void s_i(cv::Mat &id_img);
    void s_i(); // show img on GUI
    string read_bin_to_string(string fn);
    void read_bin_to_mat(string fn, int rows, int cols, int channels);
    void write_mat_to_bin(string fn);
    void write_mat_to_txt(string filename, int flag);
    void fcout(string fn, string id_s, string flag_w = "ios::out");   // str2txt str2bin can use
    void fcoutln(string fn, string id_s, string flag_w = "ios::out"); // str2txt str2bin can use
    void write_mat_to_csv(string fn);
    void str_to_bin_file(string fn, string &str_to_serial);
    std::vector<std::string> filter_out_vstr(std::vector<std::string>& vstr, const std::string& pattern);
    std::vector<std::string> filter_vstr(std::vector<std::string>& vstr, const std::string& pattern);

    template <typename T>
    std::vector<T> combine_vec(const std::vector<T> &v0, const std::vector<T> &v1);

    template <typename T>
    T sum_vec(const vector<T> &v);
    template <typename T>
    float mean_vec(const vector<T> &v);
    template <typename T>
    float stddev_vec(const vector<T> &v);

    template <typename T>
    vector<T> diff_v(const vector<T> &v);

    void serial_to_mat(const string &fn);
    void deserial_from_mat(const string &fn);
    string info();

    vector<string> split_str_2_vec(string &str, const char delimiter);
    void read_txt_to_img(string fn, int flag_has_header = 1);
    vector<string> read_txt_to_vec_str(string fn);
    void read_cmyimage_11chn_add_dna(string fn);
    cv::Mat get_rectangle_mat(cv::Mat &id_m32, int start_rows, int end_rows, int start_cols, int end_cols);
    void read_cmyimage(string fn);
    string get_timestamp();
    string run_cmd(string cmd_);
    string get_env(string env_name);
    void td_sleep(double seconds);
    double area_contour(vector<cv::Point2i> &vp);
    cv::Mat hconcat(const vector<cv::Mat> &vimg, int intv = 0);
    vector<cv::Mat> unify_vimg(const vector<cv::Mat> & vimg);
    cv::Mat vconcat(const vector<cv::Mat> &vimg, int intv = 0);
    double perimeter_contour(vector<cv::Point2i> &vp);
    void hist_img(string fn);
    void threshold(int thres, int dst_val = 255);
    cv::Mat find_contours(int dst_val = 255, int dst_sz = 300);


    template <typename T>
    cv::Mat P(const vector<T> &data, int flag_show = 1);
    void resize(float scale_f);
    void puttext(const string &text, cv::Scalar color = cv::Scalar(0, 0, 0));
    void puttext(cv::Mat &img, const string &text, cv::Scalar color = cv::Scalar(0, 0, 0));
    cv::Mat create_img_rc_chn(int rows, int cols, int chn);
    std::vector<double> linspace(double start, double stop, int num);
    std::string replace_str(const std::string &str, const string &from_str, const string &to_str);

    std::vector<std::string> get_file_list(const std::string& dir_or_pattern, bool recursive = false);
    // get int range(starti, endi), use iterator

    vector<int> R(int starti, int endi, int step = 1);
    vector<int> R(int endi);

    

    template <typename T>
    string serial_struct_2_str(T &pt)
    {
        string id_s = "";
        char *c_pt = (char *)&pt;
        id_s += string(c_pt, c_pt + sizeof(T));
        return id_s;
    }

    template <typename T>
    string serial_p_2_str(T *cstr, size_t sz);
};

template <typename T>
string cimg::serial_p_2_str(T *cstr, size_t sz)
{
    char *cstr_ = (char *)cstr;
    return string(cstr_, cstr_ + sz);
}

class SQE6E7Common
{
public:
    SQE6E7Common();

    ~SQE6E7Common();

    cv::Mat img;
    cv::Mat gray;
    vector<cv::Mat> v_hsv;

    void normalized();

    void toChnHSV();
    void toBinaryImg(int thres);
    cv::Mat filterContour();
};

// cimg_ hpp end

// cimg_ cpp start

SQE6E7Common::SQE6E7Common()
{
}

SQE6E7Common::~SQE6E7Common()
{
}

void SQE6E7Common::normalized()
{
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
}
void SQE6E7Common::toBinaryImg(int thres)
{
    // if ci.img value > 10, set to 255, else set to 0
    for (int r = 0; r < img.rows; r++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            if (img.ptr<uchar>(r)[c] > thres)
            {
                img.ptr<uchar>(r)[c] = 255;
            }
            else
            {
                img.ptr<uchar>(r)[c] = 0;
            }
        }
    }
}
cv::Mat SQE6E7Common::filterContour()
{
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    auto ci_clear = img.clone();
    ci_clear.setTo(0);

    vector<int> v_contour_size = {};

    // draw contours

    int cnt_all = 0;

    int cnt_positive = 0;
    for (size_t i = 0; i < contours.size(); i++)
    {

        // filter via contours size
        // v_contour_size.push_back(contours[i].size());

        if (contours[i].size() < 32)
        {
            continue;
        }

        cv::Rect out_rectbox = cv::boundingRect(contours[i]);

        if (out_rectbox.width > 100 && out_rectbox.height > 100)
        {
            cv::drawContours(ci_clear, contours, i, cv::Scalar(128), 3);
            cnt_positive++;
            cnt_all++;
        }
        else
        {
            cv::Moments moments = cv::moments(contours[i], true);
            auto center_xy = cv::Point2i(moments.m10 / moments.m00, moments.m01 / moments.m00);
            auto &c = center_xy;

            int sum = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    sum += gray.ptr(c.y + i)[c.x + j];
                }
            }

            auto mean = sum / 9.0f;

            if (mean < 211)
            {
                cv::drawContours(ci_clear, contours, i, cv::Scalar(255), 2);
                cnt_all++;
            }
        }
    }

    std::cout << cnt_positive << " / " << cnt_all << endl;
    cout << cnt_positive * 1.0f / cnt_all << endl;

    return ci_clear;
}

void SQE6E7Common::toChnHSV()
{
    assert(3 == img.channels());
    cv::Mat HSV;
    cv::cvtColor(img, HSV, cv::COLOR_BGR2HSV);
    // 分离 HSV 通道
    cv::split(HSV, v_hsv);

    gray = v_hsv[2].clone(); // as background image
    img = gray;
    normalized();

    img = v_hsv[1]; // to binary then
    normalized();
}

/// //////

double cimg::area_contour(vector<cv::Point2i> &vp)
{

    // vector<cv::Point2i> vp{ {0,0},{10,0},{0,5}, {0,2} ,{-2,0} };
    int cnt = 0;
    float sum = 0;

    for (int i = 0; i < vp.size() - 1; i++)
    {
        auto &x = vp[i].x;
        auto &y = vp[i].y;
        auto &xn = vp[i + 1].x;
        auto &yn = vp[i + 1].y;
        auto e_val = (x * yn - xn * y);
        sum += e_val;
    }

    // cout << sum * 1 / 2 << endl;
    return sum * 1.0 / 2;
}

cv::Mat cimg::vconcat(const vector<cv::Mat> &vimg_r0, int intv)
{
    auto vimg = unify_vimg(vimg_r0);

    auto img_copy_ = img_copy.clone();
    if (intv == 0)
    {

        // img_copy = cv::Mat(vimg[0].rows * vimg.size(), vimg[0].cols, vimg[0].type());
        cv::vconcat(vimg, img_copy_);
    }
    else
    {
        cv::Mat img_ = vimg[0].clone();

        cv::Scalar px_avg = cv::mean(img_);

        uchar intv_val = 0;
        if (px_avg[0] < 122)
        {
            intv_val = 255;
        }

        // img_copy = img.clone();

        cv::Mat img_intv = create_img_rc_chn(intv, img_.cols, img_.channels());
        img_intv.setTo(intv_val);

        vector<cv::Mat> vimg_new;
        for (auto eimg : vimg)
        {
            vimg_new.push_back(eimg);
            vimg_new.push_back(img_intv);
        }

        vimg_new.pop_back();
        // img = img_copy.clone();

        cv::vconcat(vimg_new, img_copy_);
    }

    return img_copy_;
}

vector<cv::Mat> cimg::unify_vimg(const vector<cv::Mat> & vimg)
{

    assert(vimg.size() >= 2);

    vector<cv::Mat> vimg_ret = {};

    int flag_samesz = 1; 

    auto erows = vimg[0].rows;
    auto ecols = vimg[1].cols;

    for(auto e: vimg)
    {

        if (e.rows != erows || e.cols != ecols)
        {
            flag_samesz = 0;
            break;
        }
    }


    if (! flag_samesz)
    {
        
        vector<int> vrows;
        vector<int> vcols;
        for (auto& eimg : vimg)
        {
            vrows.push_back(eimg.rows);
            vcols.push_back(eimg.cols);
        }


        std::sort(vrows.begin(), vrows.end());
        int maxrows = vrows[vrows.size() - 1];

        std::sort(vcols.begin(), vcols.end());
        int maxcols = vcols[vcols.size() - 1];



        for (auto& e : vimg)
        {
            const uchar BLANK_PIXEL_VAL = 0;
            cv::Mat elarge = cv::Mat(maxrows, maxcols, e.type()).setTo(BLANK_PIXEL_VAL); 

            // copy e to elarge's (0,0) 
            e.copyTo(elarge(cv::Rect(0, 0, e.cols, e.rows)));
            vimg_ret.push_back(elarge);
        }

    }
    else
    {
        vimg_ret =  vimg;
    }

    return vimg_ret;
}

cv::Mat cimg::hconcat(const vector<cv::Mat> &vimg_r0, int intv)
{

    assert(vimg_r0.size() >= 1);

    if (vimg_r0.size() == 1)
    {
		return vimg_r0[0];
    }

    auto vimg = unify_vimg(vimg_r0);

    cv::Mat img_copy_ = img_copy.clone(); 
    if (intv == 0)
    {
        cv::hconcat(vimg, img_copy_);
    }
    else
    {
        // img_copy = vimg[0].clone();
        cv::Mat img_ = vimg[0];
        // get pixel average

        cv::Scalar px_avg = cv::mean(img_);

        uchar intv_val = 0;
        if (px_avg[0] < 122)
        {
            intv_val = 255;
        }

        cv::Mat img_intv = create_img_rc_chn(img_.rows, intv, img_.channels());
        img_intv.setTo(intv_val);

        vector<cv::Mat> vimg_new;
        for (auto eimg : vimg)
        {
            vimg_new.push_back(eimg);
            vimg_new.push_back(img_intv);
        }

        vimg_new.pop_back();
        // img = img_copy.clone();

        cv::hconcat(vimg_new, img_copy_);
    }

    return img_copy_;
}


double cimg::perimeter_contour(vector<cv::Point2i> &vp)
{
    auto &vpi = vp;
    // vpi = { {0,0}, {2,0}, {0,4},{-1,0} };
    assert(vpi.size() > 0);

    if (vpi[0] != vpi[vpi.size() - 1])
    {
        vpi.push_back(vpi[0]);
    }

    auto start_p = vpi[0];
    auto p = start_p;

    unordered_map<string, int> d_cls{
        {"1,0", 0},
        {"1,1", 1},
        {"0,1", 2},
        {"-1,1", 3},
        {"-1,0", 4},
        {"-1,-1", 5},
        {"0,-1", 6},
        {"1,-1", 7},
        {"0,0", 8},
    };

    vector<int> vi{};
    for (int i = 1; i < vpi.size(); i++)
    {

        auto n = vpi[i];
        auto xd = n.x - p.x;
        auto yd = n.y - p.y;

        auto cnt_x = abs(xd) / 1;
        auto cnt_y = abs(yd) / 1;

        auto cnt_first = min(cnt_x, cnt_y);

        for (int j = 0; j < cnt_first; j++)
        {
            auto s = s_("{:d},{:d}", xd / cnt_x, yd / cnt_y);
            vi.push_back(d_cls[s]);
        }

        auto cnt_second = max(cnt_x, cnt_y);
        auto xd_off = 0;
        auto yd_off = 0;
        if (cnt_second == cnt_x)
        {
            yd_off = 0;
            xd_off = xd / cnt_x;
        }
        else
        {
            xd_off = 0;
            yd_off = yd / cnt_y;
        }

        for (int j = cnt_first; j < cnt_second; j++)
        {
            auto s = s_("{:d},{:d}", xd_off, yd_off);
            vi.push_back(d_cls[s]);
        }

        p = n;
    }

    // cout << s_("{}", vi) << endl;

    return vi.size() * 1.0f;
}

std::vector<double> cimg::linspace(double start, double stop, int num)
{
    if (num < 1)
    {
        throw std::invalid_argument("num must be at least 1");
    }

    std::vector<double> result(num);
    double step = (stop - start) / (num - 1); // Calculate step size

    for (int i = 0; i < num; ++i)
    {
        result[i] = start + step * i; // Generate evenly spaced values
    }

    return result;
}

std::string cimg::replace_str(const std::string &str, const string &from_str, const string &to_str)
{

    std::string S = str;

    size_t pos = S.find(from_str);

    if (pos != std::string::npos)
    {

        S.replace(pos, from_str.size(), to_str);
    }

    return S;
}
vector<int> cimg::R(int starti, int endi, int step)
{

    int cnt_all = (endi - starti) / step + 1;
    if ((endi - starti) % step == 0)
    {
        cnt_all--;
    }

    vector<int> vi(cnt_all, 0);

    // vector<int> vi((endi - starti) / step + 1, 0);

    // std::iota(vi.begin(), vi.end(), starti)

    int cnt = 0;
    for (int i = starti; i < endi; i += step)
    {
        vi[cnt] = i;
        cnt++;
    }
    assert(cnt == vi.size());
    vi.resize(cnt);

    return vi;
}
vector<int> cimg::R(int endi)
{
    // cout << "R" << "";
    return R(0, endi, 1);
}


cv::Mat cimg::create_img_rc_chn(int rows, int cols, int chn)
{
    auto img_ = cv::Mat(rows, cols, CV_8UC(chn));
    return img_;
}


void cimg::puttext(const string &text, cv::Scalar color)
{

    putText(img, "img: " + text,
            Point(img.cols / 2 - text.size(), img.rows - text.size() / 600 - 11), FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
}

void cimg::puttext(cv::Mat &img_, const string &text, cv::Scalar color)
{
    putText(img_, "img: " + text,
            Point(img_.cols / 2 - text.size(), img_.rows - text.size() / 600 - 11), FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
}
void cimg::resize(float scale_f)
{
    //    assert(img.channels() < 4);
    cv::resize(img, img, cv::Size((int)(img.cols * scale_f), int(img.rows * scale_f)));
}
#if 0
template <typename T>
cv::Mat cimg::P(const vector<T>& data, int flag_show) 
{
    if (data.empty()) {
        cout << "Data vector is empty!" << endl;
        return cv::Mat();
    }

    // 计算数据的最大值和最小值  
    auto maxValue = *max_element(data.begin(), data.end());
    auto minValue = *min_element(data.begin(), data.end());

    int width = 900;
    int height = 600;
    Mat image(height, width, CV_8UC3, Scalar(211, 233, 222)); // 白色背景  

    // 绘制坐标轴  
    line(image, Point(50, height - 50), Point(50, 50), Scalar(0, 0, 0), 2); // y 轴  
    line(image, Point(50, height - 50), Point(width - 50, height - 50), Scalar(0, 0, 0), 2); // x 轴  

    // 绘制 y 轴刻度  
    for (int i = 0; i <= 10; i++) {
        int y = height - 50 - (i * (height - 100) / 10);
        line(image, Point(45, y), Point(55, y), Scalar(0, 0, 0), 1);
        putText(image, to_string(static_cast<int>(i * (maxValue / 10))), Point(10, y + 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // 绘制 x 轴刻度  
    int dataSize = static_cast<int>(data.size());
    for (int i = 0; i < dataSize; i++) {
        int x = 50 + (i * (width - 100) / (dataSize - 1));
        line(image, Point(x, height - 45), Point(x, height - 55), Scalar(0, 0, 0), 1);
        putText(image, to_string(i), Point(x - 10, height - 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // 声明 prevX 和 prevY  
    double prevX = 50 + (0 * (width - 100) / (dataSize - 1)); // 初始化第一个点位置  
    double prevY = height - 50 - (static_cast<double>(data[0]) / maxValue * (height - 100)); // 初始化第一个点位置  

    // 根据最大值进行归一化并绘制数据  
    for (int i = 0; i < dataSize; i++) {
        double normalizedValue = static_cast<double>(data[i]) / maxValue; // 归一化到0到1之间  
        double x = 50 + (i * (width - 100) / (dataSize - 1));
        double y = height - 50 - (normalizedValue * (height - 100));

        // 绘制点  
        circle(image, Point(x, y), 5, Scalar(0, 0, 255), -1); // 用红色圆圈标记数据点  

        // 如果是第一个点，记录它的x和y  
        if (i > 0) {
            // 绘制线段  
            line(image, Point(prevX, prevY), Point(x, y), Scalar(255, 0, 0), 2);
        }

        // 更新上一个点的信息  
        prevX = x;
        prevY = y;
    }


    auto tostring_02f=[](float f)
    {
            char buf[1024] = { 0 };

            sprintf_s(buf, "%0.2f", std::round(f * 100) / 100.0f);
            string sb(buf);

            cimg ci; 
            sb = ci.replace_str(sb, ".00", "");
            
            return sb;
    };



    // 用箭头标注最大值及其位置  
    int maxIndex = distance(data.begin(), max_element(data.begin(), data.end()));
    int maxX = 50 + (maxIndex * (width - 100) / (dataSize - 1));
    int maxY = height - 50 - (maxValue / maxValue * (height - 100)); // maxY 永远是顶部  

    putText(image, "v:" + tostring_02f(maxValue) + ",i:" + to_string(maxIndex),
        Point(maxX + 10, maxY - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    // 用箭头标注最小值及其位置  
    int minIndex = distance(data.begin(), min_element(data.begin(), data.end()));
    int minX = 50 + (minIndex * (width - 100) / (dataSize - 1));
    int minY = height - 50 - (minValue / maxValue * (height - 100)); // 最小值的 y 坐标  

    
    putText(image, "v:" + tostring_02f(minValue) + ",i:" + to_string(minIndex),
        Point(minX + 10, minY + 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    // 显示图像  
    
    if (flag_show)
    {
        s_i(image);
    }

    return image;
    //imshow(windowName, image);
    //waitKey(0);
}
#endif

#if 1

template <typename T>
cv::Mat cimg::P(const vector<T> &data, int flag_show)
{
    if (data.empty())
    {
        cout << "Data vector is empty!" << endl;
        return cv::Mat();
    }

    // Calculate data max and min values
    auto maxValue = *max_element(data.begin(), data.end());
    auto minValue = *min_element(data.begin(), data.end());

    // Determine scaled height based on range
    double range = maxValue - minValue;
    if (range == 0)
        range = 1; // Avoid division by zero if all values are the same

    int width = 900;
    int height = 600;
    Mat image(height, width, CV_8UC3, Scalar(211, 233, 222)); // Light background

    // Draw axes
    line(image, Point(50, height - 50), Point(50, 50), Scalar(0, 0, 0), 2);                  // y axis
    line(image, Point(50, height - 50), Point(width - 50, height - 50), Scalar(0, 0, 0), 2); // x axis

    // Draw y-axis ticks
    for (int i = 0; i <= 10; i++)
    {
        int y = height - 50 - (i * (height - 100) / 10);
        line(image, Point(45, y), Point(55, y), Scalar(0, 0, 0), 1);
        putText(image, to_string(static_cast<int>(minValue + i * (range / 10))), Point(10, y + 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Draw x-axis ticks
    int dataSize = static_cast<int>(data.size());
    for (int i = 0; i < dataSize; i += dataSize / 10)
    {
        int x = 50 + (i * (width - 100) / (dataSize - 1));
        line(image, Point(x, height - 45), Point(x, height - 55), Scalar(0, 0, 0), 1);
        putText(image, to_string(i), Point(x - 10, height - 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Draw data
    double prevX = 50 + (0 * (width - 100) / (dataSize - 1));
    double prevY = height - 50 - ((static_cast<double>(data[0]) - minValue) / range * (height - 100)); // Normalize

    for (int i = 0; i < dataSize; i++)
    {
        double normalizedValue = (static_cast<double>(data[i]) - minValue) / range; // Normalize between 0 and 1
        double x = 50 + (i * (width - 100) / (dataSize - 1));
        double y = height - 50 - (normalizedValue * (height - 100));

        // Draw point
        circle(image, Point(x, y), 5, Scalar(0, 0, 255), -1); // Red circle for data point

        // Draw line segment
        if (i > 0)
        {
            line(image, Point(prevX, prevY), Point(x, y), Scalar(255, 0, 0), 2);
        }

        // Update previous point
        prevX = x;
        prevY = y;
    }

    auto tostring_02f = [](float f)
    {
        char buf[1024] = {0};
        sprintf_s(buf, "%0.2f", std::round(f * 100) / 100.0f);
        string sb(buf);
        cimg ci;
        sb = ci.replace_str(sb, ".00", "");
        return sb;
    };

    // Label max value
    int maxIndex = distance(data.begin(), max_element(data.begin(), data.end()));
    int maxX = 50 + (maxIndex * (width - 100) / (dataSize - 1));
    int maxY = height - 50 - ((maxValue - minValue) / range * (height - 100)); // Adjusted for minValue

    putText(image, "v:" + tostring_02f(maxValue) + ",i:" + to_string(maxIndex),
            Point(maxX + 10, maxY - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    // Label min value
    int minIndex = distance(data.begin(), min_element(data.begin(), data.end()));
    int minX = 50 + (minIndex * (width - 100) / (dataSize - 1));
    int minY = height - 50 - ((minValue - minValue) / range * (height - 100)); // minY is adjusted

    putText(image, "v:" + tostring_02f(minValue) + ",i:" + to_string(minIndex),
            Point(minX + 10, minY - 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    // Show image
    if (flag_show)
    {
        s_i(image);
    }

    return image;
}
#endif




cv::Mat cimg::find_contours(int dst_val, int dst_sz)
{
    // std::vector<std::vector<cv::Point>> v_contours;
    cv::findContours(img, v_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    img_contour_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::drawContours(img_contour_mask, v_contours, -1, cv::Scalar(dst_val), cv::FILLED);

    img_contour_line = cv::Mat::zeros(img.size(), CV_8UC1);;
    int line_size = 1; 
    cv::drawContours(img_contour_line, v_contours, -1, cv::Scalar(dst_val), line_size);

    int w = img.cols;
    int h = img.rows;
    double f = 1;
    int dstsz = dst_sz;
    if (w > dstsz)
    {
        f = w * 1.0 / dstsz;
        w = w * 1.0 / f;
        h = h * 1.0 / f;
    }

    cv::Rect bb(0, 0, w, h);
    return hconcat({ img(bb),img_contour_mask(bb), img_contour_line(bb) }, 4);
}

void cimg::threshold(int thres, int dst_val)
{
    assert(img.channels() == 1);
    cv::threshold(img, img, thres, dst_val, cv::THRESH_BINARY);
}



void cimg::hist_img(string fn)
{
    auto &ci = *this;
    // string fn = "d:/jd/t/img_rgb_cmp/_007/_007_ok/d_rgb_w.jpg";
    ci.read_img(fn);
    std::vector<cv::Mat> channels;
    cv::split(ci.img, channels);

    // ????每??通?赖?直??图
    std::vector<cv::Mat> hist(3);
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    for (int i = 0; i < 3; i++)
    {
        cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist[i], 1, &histSize, &histRange);
    }

    // ????直??图
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::normalize(hist[0], hist[0], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist[1], hist[1], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist[2], hist[2], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[0].at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(hist[0].at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[1].at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(hist[1].at<float>(i))),
                 cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[2].at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(hist[2].at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // ???雍???????
    cv::line(histImage, cv::Point(0, hist_h), cv::Point(hist_w, hist_h), cv::Scalar(0, 0, 0), 1, 8, 0);
    cv::line(histImage, cv::Point(0, hist_h), cv::Point(0, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
    for (int i = 0; i < histSize; i += 32)
    {
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, hist_h - 5), cv::Scalar(0, 0, 0), 1, 8, 0);
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
        std::stringstream ss;
        ss << i;
        cv::putText(histImage, ss.str(), cv::Point(bin_w * i, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // ????直??图图??
    cv::imwrite("histogram.png", histImage);

    cout << "ok" << endl;
}
void cimg::td_sleep(double seconds)
{
    auto sleep_time_s = (uint64_t)(seconds * 1e3);

    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_s));
}

string cimg::get_env(string env_name)
{
    const int TO_READ_SZ = 1024;
    char *buf = nullptr;
    size_t bufcnt = -1;
    auto err = _dupenv_s(&buf, &bufcnt, env_name.c_str());
    string str_buf = "NULL";

    if (err == 0 && buf != nullptr)
    {
        str_buf = string(buf);
    }
    // assert(err == 0);
    // assert(buf != nullptr);

    // string str_buf(buf);
    delete[] buf;
    return str_buf;
}

string cimg::run_cmd(string cmd_)
{
    // windows pipe _popen()
    const int TO_READ_SZ = 10240;
    char *res = new char[TO_READ_SZ];
    memset(res, '\0', TO_READ_SZ);
    auto cmd = cmd_.c_str();

    FILE *pf = _popen(cmd, "r");

    if (0 != fread(res, TO_READ_SZ, 1, pf))
    {
        printf("- cannot run system cmd %s\n", cmd);
        return "NULL";
    };
    _pclose(pf);

    string str_res(res);
    delete[] res;
    return str_res;
}

void cimg::read_cmyimage_11chn_add_dna(string fn)
{

#if 1
    // string dirname = "D:\\jd\\t\\platform_test_data\\";

    // string fn = dirname + "t0_133\\w_2464_h_2056_chn_11_si_8_fx_3_fy_4_tm_20230808_124322.dat";
    int w = 2464;
    int h = 2056;
    int chn = 1;
    int &cols = w;
    int &rows = h;

    string bin_content = read_bin_to_string(fn);
    assert(bin_content.size() == 12 * rows * cols);

    uchar *imgdata_r0 = (uchar *)bin_content.data();
    uchar *imgdata = imgdata_r0;

    auto img_ = cv::Mat(rows, cols, CV_8UC(3));

    std::copy_n(imgdata, rows * cols * 3, img_.data);
    cv::imwrite("d:/jd/t/t0/c_0_1_2_rgb.jpg", img_);

    vector<cv::Mat> v_mat;
    v_mat.resize(12);

    v_mat.push_back(img_);
    v_mat.push_back(img_);
    v_mat.push_back(img_);

    for (int chn_ = 0; chn_ < 12; chn_++)
    {
        if (chn_ < 3)
            continue;

        auto *img_addr = imgdata_r0 + chn_ * rows * cols;
        auto img_c = cv::Mat(rows, cols, CV_8UC(1));
        std::copy_n(img_addr, rows * cols * 1, img_c.data);
        auto fn_c = s_("d:/jd/t/t0/c_{}.jpg", chn_);
        cv::imwrite(fn_c.c_str(), img_c);
        v_mat.push_back(img_c);
    }

    // auto diff_mat = v_mat[3] - v_mat[11];

#endif
}
cv::Mat cimg::get_rectangle_mat(cv::Mat &id_m32, int start_rows, int end_rows, int start_cols, int end_cols)
{
    // end_rows  include!!!
    int rows = id_m32.rows;
    int cols = id_m32.cols;
    int sz_rows = end_rows - start_rows + 1; // coordinate of end_rows
    int sz_cols = end_cols - start_cols + 1;

    assert(rows >= sz_rows);
    assert(cols >= sz_cols);

    cv::Mat id_mm(sz_rows, sz_cols, CV_8UC(id_m32.channels()));

    for (int i = start_rows; i < end_rows; i++)
    {
        for (int j = start_cols; j < end_cols; j++)
        {

            auto t_from = id_m32.row(i).col(j);
            auto t_to = id_mm.row(i - start_rows).col(j - start_cols);

            for (int c = 0; c < id_m32.channels(); c++)
            {

                t_to.data[c] = t_from.data[c];
            }
        }

    } // end for i
    return id_mm;
}

void cimg::read_cmyimage(string fn)
{
    // string dirname = "D:\\jd\\t\\platform_test_data\\";

    // string fn = dirname + "t0_133\\w_2464_h_2056_chn_11_si_8_fx_3_fy_4_tm_20230808_124322.dat";
    int w = 2464;
    int h = 2056;
    int chn = 1;
    int &cols = w;
    int &rows = h;

    string bin_content = read_bin_to_string(fn);
    assert(bin_content.size() == 8 * rows * cols);

    uchar *imgdata = (uchar *)bin_content.data();
    auto img_ = cv::Mat(rows, cols, CV_8UC(3));

    std::copy_n(imgdata, rows * cols * 3, img_.data);
    vector<cv::Mat> vm_;
    cv::imwrite("bgr_big.jpg", img_);
    cv::split(img_, vm_);

    auto start = 888;
    auto end = start + 1024;

    vector<cv::Mat> vm;
    vm.resize(5 + 3 - 3);

    int start_chn = 3;
    for (auto &em : vm)
    {
        em = cv::Mat(rows, cols, CV_8UC(1));
        std::copy_n(imgdata + rows * cols * start_chn, rows * cols * 1, em.data);
        start_chn++;
        // ci.s_i(em);
    }
    int idx = 0;
    auto &i3_ = vm[idx++]; // i3_ == dna_ image
    auto &i4_ = vm[idx++]; // DAB image

    auto &dna_ = vm[idx++];

    auto &overlay_ = vm[idx++];
    auto &contour_ = vm[idx++];

    for (auto em : vm)
    {
        cv::Mat em_cut = em({100, 100 + 256}, {300, 300 + 256});
        // s_i(em_cut);
        // cv::imwrite("d:/jd/t/"+ get_timestamp() + ".jpg", em_cut);
        // td_sleep(0.5);
    }

    cv::imwrite("b.jpg", vm_[0]({start, end}, {start, end}));
    cv::imwrite("g.jpg", vm_[1]({start, end}, {start, end}));
    cv::imwrite("r.jpg", vm_[2]({start, end}, {start, end}));

    cv::imwrite("rgb.jpg", img_({start, end}, {start, end}));
    cv::imwrite("dna_gray.jpg", dna_({start, end}, {start, end}));
    cv::imwrite("overlay.bmp", overlay_({start, end}, {start, end}));
    // cv::imwrite("overlay.png", overlay_({ start,end }, { start,end }), { IMWRITE_PNG_COMPRESSION, 0 });
}

vector<string> cimg::split_str_2_vec(string &str, const char delimiter)
{

    string eline = "1,2,4,77,99";
    eline = str;
    // cout << eline << endl;

    int next = 0;
    char sp = delimiter;
    int prev = 0;

    vector<string> vec_ret{};

    vector<pair<int, int>> sub_loc;

    for (char e : eline)
    {

        if (e == sp)
        {
            // cout << string(eline.begin() + prev , eline.begin() + next) << endl;;

            // v_s.push_back(eline.substr(prev, next - prev));
            sub_loc.push_back({prev, (int)(next - prev)});
            prev = next + 1;
        }
        next++;
    }

    // cout << string(eline.begin() + prev, eline.end()) << endl;;
    // v_s.push_back(eline.substr(prev, eline.end() - eline.begin() - prev));

    sub_loc.push_back({prev, (int)(eline.end() - eline.begin() - prev)});

    for (auto ep : sub_loc)
    {
        auto sub_str = eline.substr(ep.first, ep.second);
        vec_ret.push_back(sub_str);
    }
    return vec_ret;

#if 0
    vector<string> vec_ret{};
    std::stringstream data(str);

    string line;
    while (std::getline(data, line, delimiter))
    {
        vec_ret.push_back(line);
    }

#endif

    return vec_ret;
}

vector<string> cimg::read_txt_to_vec_str(string fn)
{
    int has_header = 1;
    vector<string> vstr{};

    // vector<vector<double>> v_row_col;

    ifstream if_(fn, ios::in);

    assert(if_.is_open());

    string e_line;

    while (getline(if_, e_line))
    {
        vstr.push_back(e_line);
    }

    if_.close();

    return vstr;
}

void cimg::read_txt_to_img(string fn, int flag_has_header)
{
    int has_header = 1;

    // vector<vector<double>> v_row_col;

    ifstream if_(fn, ios::in);

    assert(if_.is_open());

    string e_line;
    int rows = 0;
    int cols = 0;
    int channels = 0;

    while (getline(if_, e_line))
    {
        auto v_s = split_str_2_vec(e_line, ';');
        for (auto e : v_s)
        {
            cout << e << endl;
        }

        int idx = 0;
        auto loc = v_s[idx].find_last_of(" ");
        assert(loc > 0);
        rows = atoi(v_s[idx].substr(loc).c_str());
        idx++;

        loc = v_s[idx].find_last_of(" ");
        assert(loc > 0);
        cols = atoi(v_s[idx].substr(loc).c_str());
        idx++;

        loc = v_s[idx].find_last_of(" ");
        assert(loc > 0);
        channels = atoi(v_s[idx].substr(loc).c_str());
        idx++;

        cout << rows << cols << channels << endl;
        break;
    }

    img = cv::Mat(rows, cols, CV_8UC(channels));

    int r = 0;
    while (getline(if_, e_line))
    {
        auto v_s = split_str_2_vec(e_line, ',');
        uchar *tr = img.row(r).data;
        int cnt = 0;
        for (auto &es : v_s)
        {
            auto &etr = tr[cnt];
            etr = (uchar)atoi(es.c_str());

            cnt++;
        }
        r++;
    }

    assert(r == rows);

    if_.close();
}

void cimg::write_mat_to_txt(string filename, int flag = 0)
{

    int i = 0;
    int j = 0;
    string sb = "";
    char buf[128] = {0};

    int rows, cols, channels;
    rows = img.rows;
    cols = img.cols;
    channels = img.channels();
    ofstream of_t(filename, ios::binary);
    assert(of_t.is_open());
    of_t.close(); // just clear file content
    ofstream of_(filename, ios::app);
    assert(of_.is_open());
    if (flag == 0)
    {
        sprintf_s(buf, "cols = %d;rows = %d;channels = %d\n", rows, cols, channels);
        sb += buf;
    }

    of_ << sb;

    const int LEN_BUF = 3072 * 4 * 3;
    char *buf_big = new char[LEN_BUF];
    char *buf_big_r0 = buf_big;

    assert(buf_big_r0 != nullptr);

    int buf_big_offset = 0;
    int sum_offset = 0;

    for (i = 0; i < rows; i++)
    {
        uchar *tr = img.row(i).data;

        for (int j = 0; j < cols * channels - 1; j++)
        {
            buf_big_offset = sprintf_s(buf_big, LEN_BUF - sum_offset, "%u,", (*tr));

            tr++;

            buf_big += buf_big_offset;
            sum_offset = (int)(buf_big - buf_big_r0);
        }
        buf_big_offset = sprintf_s(buf_big, LEN_BUF - sum_offset, "%u\n", (*tr));
        buf_big += buf_big_offset;
        sum_offset = (int)(buf_big - buf_big_r0);

        of_ << buf_big_r0;

        sum_offset = 0;
        buf_big = buf_big_r0;
    }

    delete[] buf_big_r0;
    of_.close();
}

void cimg::fcoutln(string fn, string id_s, string flag_w)
{
    auto F_W = ios::out | ios::trunc;
    if (flag_w == "ios::app")
    {
        F_W = ios::app;
    }

    // const auto F_W = (ios::out | ios::trunc);
    ofstream if_(fn.c_str(), F_W);
    if (!if_.is_open())
    {
        cout << "- make sure the file path is accessible!" << endl;
    }
    assert(if_.is_open());

    if_ << id_s << std::endl;
    if_.close();
}

void cimg::fcout(string fn, string id_s, string flag_w)
{
    auto F_W = ios::out | ios::trunc;
    if (flag_w == "ios::app")
    {
        F_W = ios::app;
    }

    // const auto F_W = (ios::out | ios::trunc);
    ofstream if_(fn.c_str(), F_W);
    if (!if_.is_open())
    {
        cout << "- make sure the file path is accessible!" << endl;
    }
    assert(if_.is_open());

    if_ << id_s;
    if_.close();
}

void cimg::write_mat_to_csv(string fn)
{
    ofstream of_(fn);
    assert(of_.is_open());

    for (int r = 0; r < img.rows; r++)
    {
        uchar *tr = img.row(r).data;

        auto cols = img.cols;
        auto chn = img.channels();
        // of_.write((char*)tr, cols * chn);
    }
}

void cimg::str_to_bin_file(string fn, string &str_to_serial)
{
    ofstream of_(fn.c_str(), ios::binary);
    of_ << str_to_serial; // never have "<< endl";
    of_.close();
}


std::vector<std::string> cimg::filter_out_vstr(std::vector<std::string>& vstr, const std::string& pattern) 
{
    // vstr = ci.filter_out_vstr(vstr, R"(^#.*)");
    std::regex regex_pattern(pattern);
    vstr.erase(
        std::remove_if(vstr.begin(), vstr.end(),
            [&regex_pattern](const std::string& s) {
        return std::regex_match(s, regex_pattern);
    }),
        vstr.end());

    return vstr;
}

std::vector<std::string> cimg::get_file_list(const std::string& dir_or_pattern, bool recursive)
{
    std::vector<cv::String> cv_paths;
    std::vector<std::string> result;

    // Use glob to get all matching files
    cv::glob(dir_or_pattern, cv_paths, recursive);

    // Convert cv::String to std::string
    for (const auto& path : cv_paths)
    {
        result.push_back(std::string(path));
    }

    return result;
}

std::vector<std::string> cimg::filter_vstr(std::vector<std::string>& vstr, const std::string& pattern) 
{
    std::regex regex_pattern(pattern);
    vstr.erase(
        std::remove_if(vstr.begin(), vstr.end(),
            [&regex_pattern](const std::string& s) {
        return ! std::regex_match(s, regex_pattern);
    }),
        vstr.end());

    return vstr;
}

template <typename T>
std::vector<T> cimg::combine_vec(const std::vector<T> &v0, const std::vector<T> &v1)
{
    // Preallocate the combined vector with the total size of both input vectors
    std::vector<T> combined;
    combined.reserve(v0.size() + v1.size()); // Reserve space to avoid multiple allocations

    // Copy elements from the first vector
    combined.insert(combined.end(), v0.begin(), v0.end());
    // Copy elements from the second vector
    combined.insert(combined.end(), v1.begin(), v1.end());

    return combined; // Return the combined vector
}

template <typename T>
T cimg::sum_vec(const vector<T> &v)
{

    T sum = 0;
    for (auto &e : v)
    {
        sum += e;
    }

    return sum;
}

template <typename T>
vector<T> cimg::diff_v(const vector<T> &v)
{
    assert(v.size() >= 2);
    std::vector<T> diff_p(v.size());
    std::adjacent_difference(v.begin(), v.end(), diff_p.begin());

    return vector<T>(diff_p.begin() + 1, diff_p.end());
}

template <typename T>
float cimg::stddev_vec(const vector<T> &v)
{
    float sumOfSquares = 0.0f;
    auto mean = mean_vec(v);

    for (float num : v)
    {
        sumOfSquares += (num - mean) * (num - mean);
    }

    return std::sqrt(sumOfSquares / v.size());
}

template <typename T>
float cimg::mean_vec(const vector<T> &v)
{

    float sum = 0.0f;

    sum += sum_vec(v);

    return sum * 1.0f / v.size();
}

void cimg::serial_to_mat(const string &fn)
{
    cv::FileStorage fs(fn, cv::FileStorage::WRITE);
    fs << "matrix" << img; // "matrix" is the key, and mat is the value
    fs.release();
}
void cimg::deserial_from_mat(const string &fn)
{
    cv::FileStorage fs(fn, cv::FileStorage::READ);
    // cv::Mat mat;
    fs["matrix"] >> img; // Read the matrix with the key "matrix"
    fs.release();
}

string cimg::info()
{

    string sb = "";
    char buf[128] = {0};

    int rows, cols, channels;
    rows = img.rows;
    cols = img.cols;
    channels = img.channels();

    sprintf_s(buf, "cols = %d;rows = %d;channels = %d\n", rows, cols, channels);
    sb += buf;

    return string(buf);
}

void cimg::write_mat_to_bin(string fn)
{
    ofstream of_(fn, ios::binary);
    assert(of_.is_open());

    for (int r = 0; r < img.rows; r++)
    {
        uchar *tr = img.row(r).data;

        auto cols = img.cols;
        auto chn = img.channels();
        of_.write((char *)tr, cols * chn);
    }

    of_.close();
}
string cimg::read_bin_to_string(string fn)
{

    // first get fn size
    std::ifstream stream(fn, ios::binary);
    assert(stream.is_open());
    stream.seekg(0, stream.end);
    size_t size_ = stream.tellg();
    // stream.seekg(0, stream.beg);
    stream.close();
    // cout << size_ << endl;

    // read content
    std::ifstream if_(fn, ios::binary);
    string bin(size_, '\0');
    if_.read((char *)bin.data(), size_);
    auto readok_bytes = if_.gcount();
    assert(readok_bytes == static_cast<std::streamsize>(size_));

    if_.close();

    return bin;
}

#if 1
void cimg::read_bin_to_mat(string fn, int rows, int cols, int channels)
{

    if (fn.empty())
        throw std::runtime_error("No binary file specified");

    // load bit stream
    std::ifstream stream(fn, ios::binary);
    assert(stream.is_open());

    stream.seekg(0, stream.end);
    size_t size_ = stream.tellg();
    stream.seekg(0, stream.beg);

    std::vector<char> bin(size_);
    stream.read(bin.data(), size_);
    auto readok_bytes = stream.gcount();
    assert(readok_bytes == static_cast<std::streamsize>(size_));

    // auto cols = size / rows;
    assert(cols * rows * channels == size_);

    img = cv::Mat(rows, cols, CV_8UC(channels));

    auto *src = bin.data();

    for (auto r = 0; r < rows; r++)
    {

        uchar *tr = img.row(r).data;

        memcpy(tr, src, cols * channels);
        src += cols * channels;
    }
}
#endif
void cimg::normalized()
{
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
}

void cimg::cvtcolor(string colormode)
{
    img.copyTo(img_copy);

    if (img.channels() == 3 && colormode == "RGB")
    {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    else if (img.channels() == 4 && colormode == "RGB")
    {
        cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
    }
	else if (img.channels() == 1 && colormode == "RGB")
	{
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	}
	
    if (colormode == "GRAY" || colormode == "gray")
    {
		if (img.channels() == 3)
		{
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
		else if (img.channels() == 4)
		{
			cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
		}
        // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        assert(img.channels() == 1);
    }
}

void cimg::read_img_gif(const std::string& filename, int frameIndex)
{
    cv::VideoCapture cap(filename);

    if (!cap.isOpened())
    {
        return;
    }

    cv::Mat frame;
    int currentIndex = 0;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        if (currentIndex == frameIndex)
        {
            cap.release();
			img = frame.clone(); // Store the frame in the img member variable
            return ;
        }

        currentIndex++;
    }

    cap.release();
    return;
}

void cimg::read_img(string fn)
{
    // if fn is not exists , then assertion error
	if (!std::filesystem::exists(fn))
	{
		assertx("error, not exist fn", 0 == 1);
	}

    std::string extension;
    if (auto pos = fn.find_last_of('.'); pos != std::string::npos)
    {
        extension = fn.substr(pos);
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char c) { return std::tolower(c); });
    }

    if (extension == ".gif" || extension == ".GIF")
	{
		read_img_gif(fn);
        return;
	}

    img = cv::imread(fn, cv::IMREAD_UNCHANGED);

	// if img is empty, then assertion error
	if (img.empty())
	{
		assertx("error, img is empty", 0 == 1);
        throw std::runtime_error("Error: Image is empty. Failed to load image from: " + fn);
	}

    

}

void cimg::s_i(cv::Mat &id_img)
{
    cv::namedWindow("id_img", cv::WINDOW_AUTOSIZE);
    imshow("id_img", id_img);
    cv::waitKey(0);
}

void cimg::s_i()
{
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    imshow("img", img);
    cv::waitKey(0);
}

string cimg::get_timestamp()
{
    const int MAX = 1024;
    char buf[MAX] = {0};
    time_t now = time(0);

    tm localtime;
    tm *local_time = &localtime;

    if (localtime_s(&localtime, &now) == 0)
    {

        int year = local_time->tm_year + 1900;
        int month = local_time->tm_mon + 1;
        int date = local_time->tm_mday;

        int hour = local_time->tm_hour;
        int minute = local_time->tm_min;
        int second = local_time->tm_sec;

        // cout << "- year:" << year << endl;

        auto len_ = snprintf(buf, MAX, "%04d%02d%02d_%02d%02d%02d", year, month, date, hour, minute, second);
        // cout << len_ << endl;
    }
    else
    {
        assert(0 == 1); // not get time!
    }
    return string(buf);
}

// cimg_ cpp end

#endif

void bgr_copy_r_to_dst(uchar *bgr_all, int rows, int cols, uchar *dst)
{
    assert(rows % 4 == 0);
    auto width_step = rows;
    int cnt = 0;

    auto sz_img = width_step * cols * strlen("BGR");

    for (int k = 0; k < sz_img; k++)
    {

        if (k % 3 == 2) // R
        {
            int k_ = k / 3;
            dst[k_] = bgr_all[k];
        }
    }

#if 0
    for (auto row_ = 0; row_ < rows; row_++)
    {
        for (auto col_ = 0; col_ < cols; col_++)
        {
            // row_ col_
            uchar& epixel = bgr_all[row_ * width_step + col_];


        }
    }
#endif
}

void bin_img(cv::Mat &e_chn_img, int thr = 140)
{
    auto &img = e_chn_img;

    for (auto r = 0; r < img.rows; r++)
    {
        uchar *tr = img.row(r).data;

        for (auto c = 0; c < img.cols; c++)
        {
            if (*tr < thr)
            {
                *tr = 0;
            }
            else
            {
                *tr = 255;
            }
            tr++;
        }
    }
}

namespace ns0
{

    cv::Mat id_mat;
    void fun0()
    {
        cout << "- fun0" << endl;
    }

    void fun1()
    {
        cout << "- fun0" << endl;
    }

};

#include <direct.h>

int id_fun1(string id_str)
{

    cout << id_str << id_str << id_str << endl;
    return 99;
};

void merge_bgr_to_color_batch(vector<string> &arr_fn, string &fn_save)
{
    cimg ci;
    // -------- //
#if 1

    assert(arr_fn.size() == 3);

    // ci.s_i(ci.img);

    int cnt = 0;
    ci.read_img(arr_fn[cnt]);
    auto color_img = cv::Mat(ci.img.rows, ci.img.cols, CV_8UC(3));

    vector<cv::Mat> v_gray_img;

    cv::split(color_img, v_gray_img);

    cnt = 0;
    for (auto &e_img : v_gray_img)
    {
        ci.read_img(arr_fn[cnt]);
        e_img = ci.img.clone();
        cnt++;
    }

    cv::merge(v_gray_img, color_img);

    // cv::cvtColor(color_img, color_img, COLOR_BGR2RGB);
    ci.s_i(color_img);

    // fn_save;

    cv::imwrite(fn_save.c_str(), color_img);

#endif
}

#define WIDTH_ALIGN(_w, _align) \
    ((_w % _align != 0) ? (_w / _align + 1) * _align : _w)
#define WIDTH_4(_w) WIDTH_ALIGN(_w, 4)

void write_byte_2_bin(string fn_prefix, BYTE *buf, int rows, int cols, int chn)
{

    int byte_len = rows * WIDTH_4(cols) * chn;

    auto id_s = string(buf, buf + byte_len);

    string fn = fn_prefix + "" + s_("row_{}_col_{}_chn_{}.bin", rows, cols, chn);
    ofstream of_(fn.c_str(), ios::binary);
    assert(of_.is_open());
    of_ << id_s;
    of_.flush();

    of_.close();
}

void byte2cvmat(byte *src, int rows, int cols, int chn, cv::Mat &id_mat)
{
    assert(id_mat.rows = rows);
    assert(id_mat.cols = cols);

    for (auto r = 0; r < rows; r++)
    {
        uchar *tr = id_mat.row(r).data;
        memcpy(tr, src, cols * chn);
        src += cols * chn;
    }
}

float &ret0(float *arr, int index)
{

    return arr[index];
}
cv::Mat get_overlap_img(const cv::Mat &img, const cv::Rect &rec)
{
    // ????一???碌? Mat ???螅???小????????同????始值为??色
    cv::Mat result(rec.size(), img.type(), cv::Scalar(255, 255, 255));

    // ??????????图???械慕???????
    cv::Rect intersection = rec & cv::Rect(0, 0, img.cols, img.rows);

    // ?????????眨?直?臃??匕?色??图??
    if (intersection.empty())
    {
        return result;
    }

    // ???????????图???械?位??
    cv::Rect result_intersection(intersection.x - rec.x, intersection.y - rec.y, intersection.width, intersection.height);

    // ?呀??????执?原图?锌?????????图????
    cv::Mat subImg = img(intersection);
    subImg.copyTo(result(result_intersection));

    return result;
}

void tosvg(std::string svg_filename, std::vector<float> &x, std::vector<float> &y)
{
    std::ofstream svg_file(svg_filename);

    if (!svg_file.is_open())
    {
        std::cerr << "Error: Unable to open file " << svg_filename << std::endl;
        return;
    }

    // Define the padding for the axes and data points
    int padding = 32;

    // Find the maximum and minimum values of x and y
    float min_x = *min_element(x.begin(), x.end());
    float max_x = *max_element(x.begin(), x.end());
    float min_y = *min_element(y.begin(), y.end());
    float max_y = *max_element(y.begin(), y.end());

    // Calculate the x and y scales based on the maximum and minimum values
    float x_range = max_x - min_x;
    float y_range = max_y - min_y;
    float x_scale = (500 - 2 * padding) / x_range;
    float y_scale = (500 - 2 * padding) / y_range;

    // Write the SVG header
    svg_file << "<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='500' height='500' viewBox='0 0 500 500'>\n";

    // Draw x and y axes
    svg_file << "<line x1='" << padding << "' y1='" << 500 - padding << "' x2='" << 500 - padding << "' y2='" << 500 - padding << "' stroke='black' />\n";
    svg_file << "<line x1='" << padding << "' y1='" << padding << "' x2='" << padding << "' y2='" << 500 - padding << "' stroke='black' />\n";

    // Write x-axis label
    svg_file << "<text x='" << (500 - padding) << "' y='" << (495 - padding) << "' font-family='Arial' font-size='12' fill='black'>DI</text>\n";

    // Write y-axis label
    svg_file << "<text x='" << (padding + 5) << "' y='" << (padding) << "' font-family='Arial' font-size='12' fill='black'>Area</text>\n";

    // Draw x-axis ticks and labels
    for (float i = min_x; i <= max_x; i += x_range / 4)
    {
        float x_tick = (i - min_x) * x_scale + padding;
        svg_file << "<line x1='" << x_tick << "' y1='" << 500 - padding - 5 << "' x2='" << x_tick << "' y2='" << 500 - padding + 5 << "' stroke='black' />\n";
        svg_file << "<text x='" << x_tick << "' y='" << (500 - padding + 15) << "' font-family='Arial' font-size='10' fill='black'>" << i << "</text>\n";
    }

    // Draw y-axis ticks and labels
    for (float i = min_y; i <= max_y; i += y_range / 4)
    {
        float y_tick = 500 - (i - min_y) * y_scale - padding;
        svg_file << "<line x1='" << padding - 5 << "' y1='" << y_tick << "' x2='" << padding + 5 << "' y2='" << y_tick << "' stroke='black' />\n";
        svg_file << "<text x='" << (padding - 30) << "' y='" << y_tick << "' font-family='Arial' font-size='10' fill='black'>" << i << "</text>\n";
    }

    // Write each data point as a circle in the SVG file
    for (size_t i = 0; i < x.size(); ++i)
    {
        float x_pos = (x[i] - min_x) * x_scale + padding;
        float y_pos = 500 - (y[i] - min_y) * y_scale - padding; // Invert y values for SVG coordinates

        // Determine the color based on x value
        std::string color = (x[i] > 2.5) ? "red" : "green";

        svg_file << "<circle cx='" << x_pos << "' cy='" << y_pos << "' r='3' fill='" << color << "' />\n";
    }

    // Write the SVG footer
    svg_file << "</svg>\n";

    svg_file.close();
}

#if 0
void generateSvg(const string & fn, const std::vector<float>& x_values, const std::vector<float>& y_values) {
    std::ofstream svgFile(fn.c_str());

    if (svgFile.is_open()) {
        svgFile << "<?xml version=\"1.0\" standalone=\"no\"?>" << std::endl;
        svgFile << "<svg width=\"500\" height=\"500\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;

        // ??????????
        svgFile << "<line x1=\"50\" y1=\"450\" x2=\"450\" y2=\"450\" stroke=\"black\" stroke-width=\"2\" />" << std::endl;
        svgFile << "<line x1=\"50\" y1=\"450\" x2=\"50\" y2=\"50\" stroke=\"black\" stroke-width=\"2\" />" << std::endl;

        // ???????莸????映???
        svgFile << "<polyline points=\"";
        for (size_t i = 0; i < x_values.size(); ++i) {
            float x = 50 + x_values[i] * 400;
            float y = 450 - y_values[i] * 400;
            svgFile << x << "," << y << " ";
        }
        svgFile << "\" stroke=\"red\" stroke-width=\"2\" fill=\"none\" />" << std::endl;

        svgFile << "</svg>" << std::endl;

        svgFile.close();
    }
}
#endif

struct obj_data
{
    double dnaindex;
    int celltype;
};

int hist_di(vector<obj_data> &vec_obj_data)
{
    float DNAUpLimit = 5;
    int m_wParameterDiscreteCount = 100;
    int index = 0;
    const int HIST_LEN = 206;
    const int HIST_LAST_IDX = HIST_LEN - 1;
    int HistCount[8][206];
    for (auto &e : HistCount)
    {
        for (auto &ee : e)
        {
            ee = 0;
        }
    }

    double step = DNAUpLimit / m_wParameterDiscreteCount * 1.0;
    for (auto it = vec_obj_data.begin(); it != vec_obj_data.end(); ++it)
    {
        auto celltype = it->celltype;
        auto dnaindex = it->dnaindex;
        if (dnaindex > DNAUpLimit)
        {
            dnaindex = DNAUpLimit;
        }

        index = int(dnaindex / step);

        HistCount[celltype][index]++;
        HistCount[celltype][HIST_LAST_IDX]++;
    }

    int cnt_row = 0;
    for (auto &e : HistCount)
    {
        int cnt_col = 0;
        for (auto &ee : e)
        {
            if (cnt_col == HIST_LAST_IDX)
                break;
            if (ee > 0)
            {
                cout << "celltype:" << cnt_row << " dnaindex:" << cnt_col << " cnt:" << ee << endl;
            }
            cnt_col++;
        }

        cnt_row++;
        cout << endl;
    }

    return HIST_LEN;
}

void mat2hsv_v(const cv::Mat &inputImage, cv::Mat &value)
{
    // ????????图???欠??? 3 通?赖?
    if (inputImage.channels() != 3)
    {
        throw std::invalid_argument("Input image must be a 3-channel (BGR) image.");
    }

    // ?? BGR 图??转??为 HSV
    cv::Mat HSV;
    cv::cvtColor(inputImage, HSV, cv::COLOR_BGR2HSV);

    // ???? HSV 通??
    std::vector<cv::Mat> hsvChannels;
    cv::split(HSV, hsvChannels);
    value = hsvChannels[2]; // ??取 value 通??

    value.convertTo(value, CV_8U);
    // ?? value 通??转??为 CV_8U
}

void imadjust(const cv::Mat &input, cv::Mat &output,
              double low_in = 0.0, double high_in = 255.0,
              double low_out = 0.0, double high_out = 255.0)
{
    // ????????图?癫⒊?始??
    output = cv::Mat(input.size(), input.type());

    // ???????????谋???
    double scale = (high_out - low_out * 1.0) / (high_in - low_in);

    // ????每?????兀????械???
    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {

            // ??取??前???氐?值
            double pixelValue = input.ptr<uchar>(y)[x];

            // ????????值
            if (pixelValue < low_in)
            {
                output.ptr<uchar>(y)[x] = low_out;
            }
            else if (pixelValue > high_in)
            {
                output.ptr<uchar>(y)[x] = high_out;
            }
            else
            {
                // ???????员???
                output.ptr<uchar>(y)[x] = cv::saturate_cast<uchar>(
                    low_out + (pixelValue - low_in) * scale);
            }
        }
    }
}

void rgb2y_v0(cv::Mat &inputImage, cv::Mat &value)
{

    mat2hsv_v(inputImage, value);

    // value.convertTo(value, CV_8U);
    //  ??去 30 ??确??值?????? 0
    value = value - 30;
    cv::threshold(value, value, 0, 0, cv::THRESH_TOZERO); // 确??值 >= 0
}

void rgb2y_v1(cv::Mat &inputImage, cv::Mat &value)
{
    mat2hsv_v(inputImage, value);
    // value.convertTo(value, CV_8U);
    //  ??去 30 ??确??值?????? 0

    // 执??????值????值????
    value.forEach<uchar>([](uchar &pixel, const int *position)
                         {
        if (pixel < 50) {
            pixel = 0; // ????小?? 50 ??????值为 0
        } });
}

void rgb2y_v2(cv::Mat &inputImage, cv::Mat &value)
{
    mat2hsv_v(inputImage, value);

    // ??去 30 ??确??值?????? 0
    cv::Mat value_out;
    imadjust(value, value_out, 0, 255, 0, 255);

    value = value_out;
}

std::string replace_str(const std::string &filename, const string &from_str, const string &to_str)
{

    std::string newFilename = filename; // ????原始?募???

    size_t pos = newFilename.find(from_str); // ???? "rgb_"

    if (pos != std::string::npos)
    { // ?????业???

        newFilename.replace(pos, from_str.size(), to_str); // ????"rgb_" 为 "y_"
    }

    return newFilename; // ?????薷暮????募???
}

#if 1

void doCurrentBackground(const cv::Mat& _orgGrey, int* _currentBackground)
{
    // todo 为什么??么??????值???


    cv::Mat _orgRGB; 

    cv::cvtColor(_orgGrey, _orgRGB, cv::COLOR_GRAY2BGR);

    int cx_ = _orgRGB.cols;
    int cy_ = _orgRGB.rows;

    int rows = cy_;
    int cols = cx_;

    int64_t sumthrehold = (cx_ * cy_) / 50;
    int hist[256] = { 0 };

    int i, j, k;
    int sum = 0;
    for (k = 0; k < 3; k++)
    {
        memset(hist, 0, 256 * sizeof(int));
        uchar* pData = _orgRGB.data + k;
        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                hist[*pData]++;
                pData += 3;
            }
        }
        sum = 0;

        for (i = 255; i > 10; i--)
        {
            sum += hist[i];
            if (sum > sumthrehold)
            {
                break;
            }
        }
        _currentBackground[k] = i;
    }

    int channel = 1;
    for (k = 0; k < channel; k++)
    {
        memset(hist, 0, 256 * sizeof(int));
        uchar* pData = _orgGrey.data + cols * rows * k;
        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                hist[*pData]++;
                pData++;
            }
        }
        sum = 0;
        for (i = 255; i > 10; i--)
        {
            sum += hist[i];
            if (sum > sumthrehold)
            {
                break;
            }
        }
        _currentBackground[3 + k] = i;
    }
}

void doCurrentBackground(const cv::Mat &_orgRGB, const cv::Mat &_orgGrey, int *_currentBackground)
{
    // todo 为什么??么??????值???
    int cx_ = _orgRGB.cols;
    int cy_ = _orgRGB.rows;

    int rows = cy_;
    int cols = cx_;

    int64_t sumthrehold = (cx_ * cy_) / 50;
    int hist[256] = {0};

    int i, j, k;
    int sum = 0;
    for (k = 0; k < 3; k++)
    {
        memset(hist, 0, 256 * sizeof(int));
        uchar *pData = _orgRGB.data + k;
        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                hist[*pData]++;
                pData += 3;
            }
        }
        sum = 0;

        for (i = 255; i > 10; i--)
        {
            sum += hist[i];
            if (sum > sumthrehold)
            {
                break;
            }
        }
        _currentBackground[k] = i;
    }

    int channel = 1;
    for (k = 0; k < channel; k++)
    {
        memset(hist, 0, 256 * sizeof(int));
        uchar *pData = _orgGrey.data + cols * rows * k;
        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                hist[*pData]++;
                pData++;
            }
        }
        sum = 0;
        for (i = 255; i > 10; i--)
        {
            sum += hist[i];
            if (sum > sumthrehold)
            {
                break;
            }
        }
        _currentBackground[3 + k] = i;
    }
}

void doinit_logtab(float *_LogTab, uint64_t *_Pow4)
{
    // float Analysis::_LogTab[256] = { 0 };
    // uint64_t Analysis::_Pow4[128] = { 0 };
    for (int i = 4; i < 256; i++)
    {
        _LogTab[i] = log10f((float)i);
        _LogTab[0] = _LogTab[1] = _LogTab[2] = _LogTab[3] = _LogTab[4];
    }

    for (int i = 0; i < 128; i++)
    {
        _Pow4[i] = (unsigned long)i * i * i * i;
    }
}
void traverse(const cv::Mat &mat, const std::function<void(int y, int x)> &func)
{
    for (int y = 0; y < mat.rows; ++y)
    {
        for (int x = 0; x < mat.cols; ++x)
        {
            func(y, x);
        }
    }
}

uint8_t &blue(cv::Mat &_orgRGB, int y, int x)
{
    assert(!_orgRGB.empty());

    return ((uchar *)(_orgRGB.data + y * _orgRGB.step))[x * 3];
    //    return _orgRGB.at<cv::Vec3b>(y, x)[0];
}

uint8_t &green(cv::Mat &_orgRGB, int y, int x)
{
    assert(!_orgRGB.empty());
    return ((uchar *)(_orgRGB.data + y * _orgRGB.step))[x * 3 + 1];

    //    return _orgRGB.at<cv::Vec3b>(y, x)[1];
}
uint8_t &red(cv::Mat &_orgRGB, int y, int x)
{
    assert(!_orgRGB.empty());
    return ((uchar *)(_orgRGB.data + y * _orgRGB.step))[x * 3 + 2];
    //    return _orgRGB.at<cv::Vec3b>(y, x)[2];
}
uint8_t &dna_grey(cv::Mat &_orgGrey, int y, int x)
{
    assert(!_orgGrey.empty());
    return _orgGrey.data[y * _orgGrey.step + x];
    //    return _orgGrey.at<uint8_t>(y, x);
}

void processCurrentBackground(cv::Mat &_orgRGB, cv::Mat &_orgGrey, float f0, float f1, float f2, float f3)
{
    float _LogTab[256] = {0};
    uint64_t _Pow4[128] = {0};
    int _currentBackground[6];

    static float UnMixIndex[11][4] = {{-0.02f, -0.06f, -0.348f, 1.33f},
                                      {-0.01f, -0.11f, -0.47f, 1.39f},
                                      {1.1f, -0.31f, -0.12f, 0.06f},
                                      {-0.43f, 0.05f, -0.28f, 1.20f},
                                      {1.1f, -0.28f, -0.02f, -0.3f},
                                      {-0.53, 0.04f, -0.34, 1.5f},
                                      {1.1f, 0.0f, -0.62f, 0.0f},
                                      {-0.27f, 0.00f, 1.20f, 0.0f},
                                      {1.18f, -0.30f, -0.280f, 0.0f},
                                      {-0.28f, 0.06f, 1.120f, 0.0f},
                                      {0.0f, -0.09f, 0.0f, 1.0f}};

    // auto& f0 = UnMixIndex[1][0];
    // auto& f1 = UnMixIndex[1][1];
    // auto& f2 = UnMixIndex[1][2];
    // auto& f3 = UnMixIndex[1][3];

    doinit_logtab(_LogTab, _Pow4);

    doCurrentBackground(_orgRGB, _orgGrey, _currentBackground);

    double BackOD[4];
    for (int i = 0; i < 4; i++)
    {
        BackOD[i] = _LogTab[_currentBackground[i]];
    }

    {
        //?然?,????,????,??木??
        traverse(_orgRGB, [&](int y, int x)
                 {
            double ODb = BackOD[0] - _LogTab[blue(_orgRGB,y, x)];
            double ODg = BackOD[1] - _LogTab[green(_orgRGB, y, x)];
            double ODr = BackOD[2] - _LogTab[red(_orgRGB, y, x)];
            double ODy = BackOD[3] - _LogTab[dna_grey(_orgGrey, y, x)];

            ODy = f0 * ODb + f1 * ODg + f2 * ODr + f3 * ODy;
            int Grey = (int)(powf(10, BackOD[3] - ODy));
            if (Grey < 0) Grey = 0;
            if (Grey > _currentBackground[3])
            {
                Grey = _currentBackground[3];
            }

            dna_grey(_orgGrey, y, x) = Grey; });
    }
}
#endif

#if 1

std::string filename2y(const std::string &filename)
{
    std::string newFilename = filename;    // ????原始?募???
    size_t pos = newFilename.find("rgb_"); // ???? "rgb_"
    if (pos != std::string::npos)
    {                                      // ?????业???
        newFilename.replace(pos, 4, "y_"); // ????"rgb_" 为 "y_"
    }
    return newFilename; // ?????薷暮????募???
}

void Mat2Hsv(const cv::Mat &inputImage, cv::Mat &value)
{
    // ????????图???欠??? 3 通?赖?
    if (inputImage.channels() != 3)
    {
        throw std::invalid_argument("Input image must be a 3-channel (BGR) image.");
    }

    // ?? BGR 图??转??为 HSV
    cv::Mat HSV;
    cv::cvtColor(inputImage, HSV, cv::COLOR_BGR2HSV);

    // ???? HSV 通??
    std::vector<cv::Mat> hsvChannels;
    cv::split(HSV, hsvChannels);
    value = hsvChannels[2]; // ??取 value 通??

    value.convertTo(value, CV_8U);
    // ?? value 通??转??为 CV_8U
}

void MatlabImadjust(const cv::Mat &input, cv::Mat &output,
                    double low_in = 0.0, double high_in = 255.0,
                    double low_out = 0.0, double high_out = 255.0)
{
    // ????????图?癫⒊?始??
    output = cv::Mat(input.size(), input.type());

    // ???????????谋???
    double scale = (high_out - low_out) / (high_in - low_in);

    // ????每?????兀????械???
    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {

            // ??取??前???氐?值
            double pixelValue = input.ptr<uchar>(y)[x];

            // ????????值
            if (pixelValue < low_in)
            {
                output.ptr<uchar>(y)[x] = low_out;
            }
            else if (pixelValue > high_in)
            {
                output.ptr<uchar>(y)[x] = high_out;
            }
            else
            {
                // ???????员???
                output.ptr<uchar>(y)[x] = cv::saturate_cast<uchar>(
                    low_out + (pixelValue - low_in) * scale);
            }
        }
    }
}

void Rgb2yV0(cv::Mat &inputImage, cv::Mat &value)
{

    Mat2Hsv(inputImage, value);

    // value.convertTo(value, CV_8U);
    //  ??去 30 ??确??值?????? 0
    value = value - 30;
    cv::threshold(value, value, 0, 0, cv::THRESH_TOZERO); // 确??值 >= 0
}

void Rgb2yV1(cv::Mat &inputImage, cv::Mat &value)
{
    Mat2Hsv(inputImage, value);
    // value.convertTo(value, CV_8U);
    //  ??去 30 ??确??值?????? 0

    // 执??????值????值????
    value.forEach<uchar>([](uchar &pixel, const int *position)
                         {
        if (pixel < 50) {
            pixel = 0; // ????小?? 50 ??????值为 0
        } });
}

void Rgb2yV2(cv::Mat &inputImage, cv::Mat &value)
{
    Mat2Hsv(inputImage, value);

    // ??去 30 ??确??值?????? 0
    cv::Mat value_out;
    MatlabImadjust(value, value_out, 0, 255, 0, 255);

    value = value_out;
}

#endif

#if 1
// ?氐??????????诨????谋???
void on_trackbar(int, void *)
{
    // ???????????????眨???为???腔?????循???写???图??????
}
void on_trackbar_f0(int, void *)
{
    // ???????????????眨???为???腔?????循???写???图??????
}

void on_trackbar_f1(int, void *)
{
    // ???????????????眨???为???腔?????循???写???图??????
}

void on_trackbar_f2(int, void *)
{
    // ???????????????眨???为???腔?????循???写???图??????
}

void on_trackbar_f3(int, void *)
{
    // ???????????????眨???为???腔?????循???写???图??????
}

float factor_i_to_f(int i0, int i_l = 0, int i_h = 100, float f_l = -2, float f_h = 2)
{
    return ((float)i0 - i_l) / (i_h - i_l) * (f_h - f_l) + f_l;
}

int factor_f_to_i(float f0, float f_l = -2, float f_h = 2, int i_l = 0, int i_h = 100)
{
    return (int)((f0 - f_l) / (f_h - f_l) * (i_h - i_l) + i_l);
}

bool ifMaskIsFilled(const cv::Mat &overlay_rect)
{
    bool retcode = true;
    int cen_r = (overlay_rect.rows) / 2;
    int cen_c = (overlay_rect.cols) / 2;
    int r = 0;
    int c = 0;
    int r0 = 0;
    int r1 = 0;
    int c0 = 0;
    int c1 = 0;

    if (overlay_rect.ptr(cen_r)[cen_c] == 0)
    {
        retcode = false;
    }
    else
    {
        // if a cell kernel is filled with non-zero pixel?
        const int step = 6;
        int cnt = 0;
        for (int k = 0; k < 2; k++)
        {
            for (c = 0; c < overlay_rect.cols; c++)
            {
                uchar pixel = overlay_rect.ptr(cen_r + k * step)[c];
                if (pixel > 0 && c0 == 0)
                {
                    c0 = c;
                    continue;
                }

                if (c0 != 0 && pixel == 0)
                {
                    c1 = c;
                    break;
                }
            }

            if (c1 - c0 != 1)
            {
                c0 = 0;
                c1 = 0;
            }
            else
            {
                break;
            }
        }

        for (int k = 0; k < 2; k++)
        {

            for (r = 0; r < overlay_rect.rows; r++)
            {
                uchar pixel = overlay_rect.ptr(r)[cen_c + k * step];
                if (pixel > 0 && r0 == 0)
                {
                    r0 = r;
                    continue;
                }

                if (r0 != 0 && pixel == 0)
                {
                    r1 = r;
                    break;
                }
            }

            if (r1 - r0 != 1)
            {
                r0 = 0;
                r1 = 0;
            }
            else
            {
                break;
            }
        }
    }

    if (c1 - c0 == 1 || r1 - r0 == 1)
    {
        retcode = false;
    }

    return retcode;
}

void pk_algo(const cv::Mat &_rgb)
{

    cv::Mat hsv;
    cvtColor(_rgb, hsv, COLOR_BGR2HSV);
    cv::Mat mask1;
    cv::Mat mask2;

    inRange(hsv, Scalar(0, 150, 70), Scalar(3, 255, 255), mask1);
    inRange(hsv, Scalar(150, 150, 70), Scalar(180, 255, 255), mask2);

    cv::Mat mask;
    mask = mask1 | mask2;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (contourArea(contours[i]) < 700)
        {
            continue;
        }
        double pixelRedThreshold = area / 50;
        Rect rect = boundingRect(contours[i]);
        Mat roi = _rgb(rect);
        Mat hsv_roi;
        cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
        Mat mask_roi;
#if 0			
        inRange(hsv_roi, Scalar(0, 70, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv_roi, Scalar(180, 70, 70), Scalar(180, 255, 255), mask2);
#endif

#if 1
        inRange(hsv_roi, Scalar(0, 150, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv_roi, Scalar(180, 150, 70), Scalar(180, 255, 255), mask2);
#endif

        mask_roi = mask1 | mask2;

        int cropSize = 256;

        int count = countNonZero(mask_roi);
        if (count > pixelRedThreshold)
        {
            Moments m = moments(contours[i]);
            Point centroid(m.m10 / m.m00, m.m01 / m.m00);
            int x = centroid.x - cropSize / 2;
            int y = centroid.y - cropSize / 2;
            x = max(x, 0);
            y = max(y, 0);
            x = min(x, _rgb.cols - cropSize);
            y = min(y, _rgb.rows - cropSize);
            Mat crop = _rgb(Rect(x, y, cropSize, cropSize));
            /*      int globalX = globalPos.split("_").first().toUInt();
                  int globalY = globalPos.split("_").last().toUInt();*/
            Mat brightnessMat;
            extractChannel(hsv_roi, brightnessMat, 2);
            Scalar averageBrightness = mean(brightnessMat);
            double value = 255 - averageBrightness.val[0];
            if (value > 100)
            {
                value = 100;
            }
            // resultMatMap.insert(QString("%1_%2_%3").arg(centroid.x + globalX).arg(centroid.y + globalY).arg(value), crop);
        }
    }
}

#endif

#if 1

// scanapp_ hpp start
class scanapp
{
public:
    scanapp();
};
// scanapp_ hpp end

// scandlg_ hpp start
class scandlg
{
public:
    scandlg();
};
// scandlg_ hpp end

// stage_ hpp start
class stage
{
public:
    stage();

    stage *get_stage();
    stage *id_stage;
    void init();
};
// stage_ hpp end

// measure_ hpp start
class measure
{

public:
    measure();
};
// measure_ hpp end

// dbmgr_ hpp start
class dbmgr
{
public:
    dbmgr();
};
// dbmgr_ hpp end

#endif

#if 1

struct Peak
{
    int index;
    double value;

    // 用于比较两个波峰，以便可以按值排序
    bool operator<(const Peak &other) const
    {
        return value < other.value; // 从小到大排序，用于找出最大波峰
    }
};

std::vector<Peak> find_top3peak(const std::vector<double> &data, int min_distance, int margin)
{
    std::vector<Peak> peaks; // 存储波峰的结构体
    int n = data.size();

    for (int i = margin; i < n - margin; ++i)
    {
        // 检查当前点是否是波峰
        bool is_peak = true;

        // 检查左右 10 个元素
        for (int j = 1; j <= margin; ++j)
        {
            if (data[i] <= data[i - j] || data[i] <= data[i + j])
            {
                is_peak = false;
                break;
            }
        }

        // 如果是波峰，检测与已有波峰的距离
        if (is_peak)
        {
            if (peaks.empty() || i - peaks.back().index >= min_distance)
            {
                peaks.push_back({i, data[i]}); // 存储波峰的索引和值
            }
        }
    }

    // 按波峰值排序，找到最大的三个波峰
    std::sort(peaks.begin(), peaks.end()); // 从小到大排序
    std::vector<Peak> top_three_peaks;

    // 取出最大的三个波峰
    int count = 0;
    for (int i = peaks.size() - 1; i >= 0 && count < 3; --i)
    {
        top_three_peaks.push_back(peaks[i]);
        ++count;
    }

    // 反转以保持索引顺序
    // std::reverse(top_three_peaks.begin(), top_three_peaks.end());
    return top_three_peaks;
}

pair<int, int> diag_diffsum(const cv::Mat &img, int step)
{

    int cnt = 0;
    // int step = 4;
    int rows = img.rows;
    int cols = img.cols;

    int sr = 0;
    // assert(rows/2 > center_edge);
    int center_edge = img.rows - step - 1;
    // int sr = rows/2;
    assert(sr + center_edge < img.rows);

    int sc = rows / 2;

    const uchar *gray_r0 = (img.data);

    int widthstep = cols * 1;
    uchar *gray = (uchar *)gray_r0;

    vector<int> vdiag;
    vdiag.resize(center_edge / step);
    vector<int> vdiag_a;
    vdiag_a.resize(center_edge / step);

    int sum = 0;
    int sum_a = 0;
    cnt = 0;

    int possum = cnt + sr + cnt + sr + center_edge;

    while (cnt < center_edge / step)
    {
        auto *gray = gray_r0 + (cnt + sr) * widthstep + (cnt + sr);
        auto *gray_a = gray_r0 + (cnt + sr) * widthstep + (possum - cnt - sr);
        vdiag[cnt] = *gray;
        vdiag_a[cnt] = *gray_a;

        sum += (*gray);
        sum_a += (*gray_a);
        cnt++;
    }

    float mean = sum * 1.0f / vdiag.size();
    float mean_a = sum_a * 1.0f / vdiag_a.size();

    int stdv = 0;
    for (auto e : vdiag)
    {
        stdv += (e - mean) * (e - mean); // Summing squared differences
    }

    int stdv_a = 0;
    for (auto e : vdiag_a)
    {
        stdv_a += (e - mean) * (e - mean); // Summing squared differences
    }
    stdv = sqrt(stdv / vdiag.size());
    stdv_a = sqrt(stdv_a / vdiag_a.size());

    // get diff

    int sumdiff = 0;
    for (int i = 0; i < vdiag.size() - 1; i++)
    {
        auto &prev = vdiag[i];
        auto &next = vdiag[i + 1];

        auto d = abs(next - prev);
        sumdiff += d;
    }
    int sumdiff_a = 0;
    for (int i = 0; i < vdiag_a.size() - 1; i++)
    {
        auto &prev = vdiag_a[i];
        auto &next = vdiag_a[i + 1];

        auto d = abs(next - prev);
        sumdiff_a += d;
    }

    // ci.s_i();

    // cout << stdv << "," << stdv_a << "   ,  " << sumdiff << "," << sumdiff_a << endl;

    // cout << "sum:"<<stdv + stdv_a << ", " << sumdiff + sumdiff_a << endl;

    // cout <<"min:"<< min(stdv, stdv_a) << ", " << min(sumdiff, sumdiff_a) << endl;

    return {stdv + stdv_a, sumdiff + sumdiff_a};
}

void fmts(string &s)
{
    string sa;
    for (auto ec : s)
    {
        if ((ec >= '0' && ec <= '9') || ec == ',')
        {
            sa.push_back(ec);
        }
    }

    s = sa;
}

void deal_eline(cimg &ci, string &es, vector<string> &vp, vector<string> &vn)
{

    auto fall = ci.split_str_2_vec(es, '[');

    auto p = fall[1];
    auto n = fall[2];

    fmts(p);
    fmts(n);

    vp = ci.split_str_2_vec(p, ',');
    vn = ci.split_str_2_vec(n, ',');

    //  cout << vp.size() << SP << vn.size() << endl;

    auto sz = min(vp.size(), vn.size());

    vp = vector<string>(vp.begin(), vp.begin() + sz);
    vn = vector<string>(vn.begin(), vn.begin() + sz);
    assert(vp.size() == vn.size());
}

#if 1
void from_dat_to_v_p(string title, int max_idx, int step)
{

    string fn_d = "d:/jd/t/t0";
    cimg ci;
    int rows = 2056;
    int cols = 2464;

    vector<int> v_sum_diff;

    for (int i = 0; i < max_idx; i++)
    {
        string fn_dat = s_("zidx_{}_{}.dat", title, i);
        string fn_dat_full = fn_d + "/" + fn_dat;

        ci.read_bin_to_mat(fn_dat_full, rows, cols, 1);

        // ci.s_i();

        int sz = min(rows, cols);

        vector<int> vp;
        vector<int> vn;

        for (int i = 0; i < sz; i++)
        {
            auto &e_pv = ci.img.ptr<uchar>(i)[i];
            auto &e_nv = ci.img.ptr<uchar>(i)[sz - i - 1];

            if (i % step == 0)
            {
                vp.push_back(e_pv);
                vn.push_back(e_nv);
            }
        }

        std::vector<int> diff_p(vp.size());
        std::adjacent_difference(vp.begin(), vp.end(), diff_p.begin());

        std::vector<int> diff_n(vn.size());
        std::adjacent_difference(vn.begin(), vn.end(), diff_n.begin());

        vector<int> diff_all = ci.combine_vec(diff_p, diff_n);

        std::transform(diff_all.begin(), diff_all.end(), diff_all.begin(),
                       [](int ev)
                       { return std::abs(ev); });

        auto esum = std::accumulate(diff_all.begin(), diff_all.end(), 0);

        v_sum_diff.push_back(esum);
    }

    cout_("{}\n", v_sum_diff);

    cout_("{},{}\n", v_sum_diff[206], v_sum_diff[238]);

    ci.img = ci.P(v_sum_diff, 0);
    ci.puttext(title);

    ci.s_i();
}

#endif

class A
{

public:
    A();
    A(int x_);

    ~A();

protected:
    int x;
};

A::A()
{
    x = -9;
}
A::A(int x_)
{
    // this->A();

    cout << x << endl;
    x = x_;
    cout << x << endl;
}
A::~A()
{
}

char *ret_test()
{
    vector<uchar> vu{'a', 'b', 'c', 'd'};
    return (char *)(vu.data());
    // char* s = new char[4];
    // memcpy(s, vu.data(), 4);

    // for (uint i = 0; i < 4; i++)
    //{
    //     cout << s[i] << endl;
    // }

    // return (s);
}

#if 1
void from_dat_to_v_p_png(string fn_d, string title, int max_idx, int step)
{
    // .\x64\Debug\t0.exe   d:/jd/doc/sw_monograph a 399 16
    // fn_d = "d:/jd/t/t0";
    cimg ci;
    int rows = 2056;
    int cols = 2464;

    vector<int> v_sum_diff;

    auto t0 = GetTickCount64();

    int sum_time_load = 0;
    for (int i = 0; i < max_idx; i++)
    {

        string fn_dat_full = s_("{}/zidx_{}_{}.dat.png", fn_d, title, i);
        // string fn_dat_full = fn_d + "/" + fn_dat;

        // ci.read_bin_to_mat(fn_dat_full, rows, cols, 1);
        auto t00 = GetTickCount64();

        ci.read_img(fn_dat_full);

        auto d00 = GetTickCount64() - t00;
        sum_time_load += d00;

        if (i % 20 == 0 && i != 0)
        {
            auto fn_t = ci.replace_str(fn_dat_full, fn_d + "/", "");
            cout << " | process:" << fn_t << endl;
            // cout << "process:" << fn_dat_full << endl;
        }
        else
        {
            if (i == 0)
            {
            }
            else
            {
                printf("%3d,", i);
            }
        }
        // ci.s_i();

        int sz = min(rows, cols);

        vector<int> vp;
        vector<int> vn;

        for (int i = 0; i < sz; i++)
        {
            auto &e_pv = ci.img.ptr<uchar>(i)[i];
            auto &e_nv = ci.img.ptr<uchar>(i)[sz - i - 1];

            if (i % step == 0)
            {
                vp.push_back(e_pv);
                vn.push_back(e_nv);
            }
        }

        std::vector<int> diff_p(vp.size());
        std::adjacent_difference(vp.begin(), vp.end(), diff_p.begin());

        std::vector<int> diff_n(vn.size());
        std::adjacent_difference(vn.begin(), vn.end(), diff_n.begin());

        vector<int> diff_all = ci.combine_vec(diff_p, diff_n);

        std::transform(diff_all.begin(), diff_all.end(), diff_all.begin(),
                       [](int ev)
                       { return std::abs(ev); });

        auto esum = std::accumulate(diff_all.begin(), diff_all.end(), 0);

        v_sum_diff.push_back(esum);
    }

    auto d_ts = GetTickCount64() - t0;

    // cout_("{}\n", v_sum_diff);

    // cout_("{},{}\n", v_sum_diff[206], v_sum_diff[238]);

    ci.img = ci.P(v_sum_diff, 0);

    ci.puttext(title);

    vector<cv::Mat> vimg;

    float target_rows = 512.0f;

    float f = target_rows / ci.img.rows;
    ci.resize(f);

    vimg.push_back(ci.img);

    // get max and max index for v_sum_diff
    auto maxIndex = std::distance(v_sum_diff.begin(), std::max_element(v_sum_diff.begin(), v_sum_diff.end()));

    string fn_dat_full = s_("{}/zidx_{}_{}.dat.png", fn_d, title, maxIndex);
    ci.read_img(fn_dat_full);
    f = target_rows / ci.img.rows;
    ci.resize(f);
    cv::cvtColor(ci.img, ci.img, cv::COLOR_GRAY2BGR);

    fn_dat_full = ci.replace_str(fn_dat_full, fn_d, "");
    ci.puttext(fn_dat_full);

    vimg.push_back(ci.img);

    ci.img = ci.hconcat(vimg, 0);

    cout << endl
         << "\n\ttime cost for find focus(not include load images):" << d_ts - sum_time_load << " ms" << endl;

    ci.s_i();
}

void from_dat_to_v_p_png_slow(string fn_d, string title, int max_idx, int step)
{
    // .\x64\Debug\t0.exe   d:/jd/doc/sw_monograph a 399 16
    // fn_d = "d:/jd/t/t0";
    cimg ci;
    int rows = 2056;
    int cols = 2464;

    vector<int> v_sum_diff;

    auto t0 = GetTickCount64();

    int sum_time_load = 0;
    for (int i = 0; i < max_idx; i++)
    {

        string fn_dat_full = s_("{}/zidx_{}_{}.dat.png", fn_d, title, i);
        // string fn_dat_full = fn_d + "/" + fn_dat;

        // ci.read_bin_to_mat(fn_dat_full, rows, cols, 1);
        auto t00 = GetTickCount64();

        ci.read_img(fn_dat_full);

        auto d00 = GetTickCount64() - t00;
        sum_time_load += d00;

        if (i % 20 == 0 && i != 0)
        {
            auto fn_t = ci.replace_str(fn_dat_full, fn_d + "/", "");
            cout << " | process:" << fn_t << endl;
            // cout << "process:" << fn_dat_full << endl;
        }
        else
        {
            if (i == 0)
            {
            }
            else
            {
                printf("%3d,", i);
            }
        }
        // ci.s_i();

        vector<int> vpn;
        for (int r = 0; r < ci.img.rows; r++)
        {
            for (int c = 0; c < ci.img.cols; c++)
            {
                auto &e = ci.img.ptr<uchar>(r)[c];

                if (r % step == 0 && c % step == 0)
                {
                    vpn.push_back(e);
                }
            }
        }

        std::vector<int> diff_pn(vpn.size());
        std::adjacent_difference(vpn.begin(), vpn.end(), diff_pn.begin());

#if 0
        std::vector<int> diff_p(vp.size());
        std::adjacent_difference(vp.begin(), vp.end(), diff_p.begin());

        std::vector<int> diff_n(vn.size());
        std::adjacent_difference(vn.begin(), vn.end(), diff_n.begin());

        vector<int> diff_all = ci.combine_vec(diff_p, diff_n);
#endif

        vector<int> diff_all = diff_pn;

        std::transform(diff_all.begin(), diff_all.end(), diff_all.begin(),
                       [](int ev)
                       { return std::abs(ev); });

        auto esum = std::accumulate(diff_all.begin(), diff_all.end(), 0);

        v_sum_diff.push_back(esum);
    }

    auto d_ts = GetTickCount64() - t0;

    // cout_("{}\n", v_sum_diff);

    // cout_("{},{}\n", v_sum_diff[206], v_sum_diff[238]);

    ci.img = ci.P(v_sum_diff, 0);

    ci.puttext(title);

    vector<cv::Mat> vimg;

    float target_rows = 512.0f;

    float f = target_rows / ci.img.rows;
    ci.resize(f);

    vimg.push_back(ci.img);

    // get max and max index for v_sum_diff
    auto maxIndex = std::distance(v_sum_diff.begin(), std::max_element(v_sum_diff.begin(), v_sum_diff.end()));

    string fn_dat_full = s_("{}/zidx_{}_{}.dat.png", fn_d, title, maxIndex);
    ci.read_img(fn_dat_full);
    f = target_rows / ci.img.rows;
    ci.resize(f);
    cv::cvtColor(ci.img, ci.img, cv::COLOR_GRAY2BGR);

    fn_dat_full = ci.replace_str(fn_dat_full, fn_d, "");
    ci.puttext(fn_dat_full);

    vimg.push_back(ci.img);

    ci.img = ci.hconcat(vimg, 0);

    cout << endl
         << "\n\ttime cost for find focus(not include load images):" << d_ts - sum_time_load << " ms" << endl;

    ci.s_i();
}
#endif

#endif

#if 0

std::pair<cv::Mat, std::vector<std::vector<cv::Point>>> genCervicalOverlayWithGradient(cv::Mat& rgb)
{
    cv::Mat image;
    cv::GaussianBlur(rgb.clone(), image, cv::Size(5, 5), 0, 0);

    auto cytoContours = calcCytoCervical(context);

    //将包浆外部区域的背景色加深
    cv::Mat maskCyto = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::drawContours(maskCyto, cytoContours, -1, cv::Scalar(255), cv::FILLED);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    // 对包浆之外的区域进行变浅处理
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            if (maskCyto.ptr(y)[x] == 0) {
                channels[0].ptr(y)[x] = 0;
                channels[1].ptr(y)[x] = 0;
                channels[2].ptr(y)[x] = 0;
            }
        }
    }

    // 计算每个通道的梯度幅值
    std::vector<cv::Mat> gradientMagnitudes;
    for (const auto& channel : channels) {
        cv::Mat Gx, Gy;
        cv::Sobel(channel, Gx, CV_64F, 1, 0, 3); // 水平方向梯度
        cv::Sobel(channel, Gy, CV_64F, 0, 1, 3); // 垂直方向梯度
        cv::Mat gradientMagnitude;
        cv::magnitude(Gx, Gy, gradientMagnitude);
        gradientMagnitudes.push_back(gradientMagnitude);
    }


    cv::normalize(gradientMagnitudes[0], gradientMagnitudes[0], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(gradientMagnitudes[1], gradientMagnitudes[1], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(gradientMagnitudes[2], gradientMagnitudes[2], 0, 255, cv::NORM_MINMAX, CV_8U);

    // 取梯度, 取其中最大值 * 3
    cv::Mat gradientMagnitude;
    (cv::max)(gradientMagnitudes[0], gradientMagnitudes[1], gradientMagnitude);
    (cv::max)(gradientMagnitude, gradientMagnitudes[2], gradientMagnitude);
    cv::multiply(gradientMagnitude, 3, gradientMagnitude);

    LOG_DEBUG << "gradientMagnitude.type() = " << gradientMagnitude.type() << endl;

    context->saveMiddleImg(gradientMagnitude, "gradientMagnitude.jpg");

    // 二值化
    cv::Mat binaryImage;
    cv::threshold(gradientMagnitude, binaryImage, 70, 255, cv::THRESH_BINARY);

    context->saveMiddleImg(binaryImage, "binaryImage.jpg");
    // 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 计算每个轮廓度的凸包
    std::vector<std::vector<cv::Point>> convexHulls;
    for (const auto& contour : contours) {

        if (!filterContour(contour, context)) {
            continue;
        }

        convexHulls.push_back(contour);
    }

    // 填充白色, 方便下一步继续识别细胞核
    for (const auto& contour : convexHulls) {
        cv::fillPoly(binaryImage, std::vector<std::vector<cv::Point>>{contour}, cv::Scalar(255));
    }

    context->saveMiddleImg(binaryImage, "fill_binaryImage.jpg");

    contours.clear();
    // 轮廓检测
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    LOG_DEBUG << "contours.size() = " << contours.size() << endl;

    std::vector<std::vector<cv::Point>> resultContours;
    std::vector<std::vector<cv::Point>> resultContoursAll;
    for (size_t i = 0; i < contours.size(); i++) {
        if (filterContour(contours[i], context)) {

            resultContoursAll.push_back(contours[i]);

            cv::Rect rect = cv::boundingRect(contours[i]);

            //在rect内部, 再重新寻找路径, 方法: 计算路径内的平均灰度值, 小于平均灰度值作为新的路径
            cv::Mat coreMat = context->getOrgGrey()(rect);
            cv::Mat maskMat = cv::Mat::zeros(coreMat.size(), coreMat.type());
            //mask内部填充为白色
            cv::drawContours(maskMat, std::vector<std::vector<cv::Point>>({ contours[i] }), -1, cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, cv::Point(-rect.x, -rect.y));
            auto meanScalar = cv::mean(coreMat, maskMat);

            //计算外圈的平均灰度值
            cv::bitwise_not(maskMat, maskMat);
            auto meanScalar2 = cv::mean(coreMat, maskMat);

            //细胞浆更浅, 比细胞核 灰度值>5才认为有效
            int diff = meanScalar2[0] - meanScalar[0];
            if (diff > 5)
            {
                // LOG_DEBUG << "meanScalar[0] = " << meanScalar[0] << ", meanScalar2[0] = " << meanScalar2[0] << ", meanScalar[0] - meanScalar2[0] = " << meanScalar[0] - meanScalar2[0] << endl;

                cv::Mat binaryImage;
                cv::threshold(coreMat, binaryImage, meanScalar[0] + (std::min)(15, diff / 2), 255, cv::THRESH_BINARY_INV);
                std::vector<std::vector<cv::Point>> newContours;
                cv::findContours(binaryImage, newContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

                for (int j = 0; j < newContours.size(); j++)
                {
                    for (int k = 0; k < newContours[j].size(); k++)
                    {
                        newContours[j][k].x += rect.x;
                        newContours[j][k].y += rect.y;
                    }

                    if (!filterContour(newContours[j], context))
                    {
                        continue;
                    }

                    resultContours.push_back(newContours[j]);
                }
            }
        }
    }

    LOG_DEBUG << "resultContours.size() = " << resultContours.size() << ", resultContoursAll.size() = " << resultContoursAll.size() << endl;

    cv::Mat mask = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
    cv::drawContours(mask, resultContours, -1, cv::Scalar(255), cv::FILLED);

    if (context->isSaveMiddle())
    {
        context->saveMiddleImg(mask, "org_core.jpg");

        cv::Mat org = context->getOrgRGB().clone();
        cv::drawContours(org, resultContours, -1, cv::Scalar(0, 255, 0), 1);
        context->saveMiddleImg(org, "org_core_contours.jpg");

        cv::Mat orgAll = context->getOrgRGB().clone();
        cv::drawContours(orgAll, resultContoursAll, -1, cv::Scalar(0, 255, 0), 1);
        context->saveMiddleImg(orgAll, "org_core_contours_all.jpg");
    }

    for (int y = 0; y < binaryImage.rows; ++y) {
        for (int x = 0; x < binaryImage.cols; ++x) {
            if (mask.ptr(y)[x] == 255)
            {
                binaryImage.ptr(y)[x] = CORE_VALUE;
            }
            else
            {
                binaryImage.ptr(y)[x] = 0;
            }
        }
    }

    return std::make_pair(binaryImage, resultContours);
}

#endif

#if 1

std::vector<std::vector<cv::Point>> calcCytoCervical(cv::Mat &bgr, int *bg)
{

    // use cv::imwrite to implement saveMiddleImg
    auto saveMiddleImg = [](const cv::Mat &img, const std::string &filename)
    {
        cv::imwrite(filename, img);
    };

    int backgroundThre = bg[0] - 30;

    // 使用blue通道计算包浆
    std::vector<cv::Mat> brgImg;
    cv::split(bgr, brgImg);
    cv::Mat binaryImage;
    cv::threshold(brgImg[0], binaryImage, backgroundThre, 255, cv::THRESH_BINARY);

    if (1)
    {

        saveMiddleImg(brgImg[0], "org_blue.jpg");

        saveMiddleImg(brgImg[1], "org_green.jpg");
        saveMiddleImg(brgImg[2], "org_red.jpg");

        saveMiddleImg(binaryImage, "org_background.jpg");
    }

    cv::Mat invertedGrayMask;
    cv::bitwise_not(binaryImage, invertedGrayMask);

    /////////////////////////
    cv::Size kernelSize = cv::Size(5, 5);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernelSize);

    // 应用闭运算
    cv::Mat imgClosed;
    cv::morphologyEx(invertedGrayMask, imgClosed, cv::MORPH_CLOSE, kernel);

    // 应用开运算
    cv::morphologyEx(imgClosed, invertedGrayMask, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> cypoContours;
    cv::findContours(invertedGrayMask, cypoContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 过滤无效的数据, 太大或者太小都过滤掉
    std::vector<std::vector<cv::Point>> cytoContours;

    float pixelLength = 0.565;

    for (size_t i = 0; i < cypoContours.size(); i++)
    {
        auto area = cv::contourArea(cypoContours[i]) * pixelLength * pixelLength;

#define MIN_CYTO_AREA (150 / 4)

        if (area < MIN_CYTO_AREA)
        {
            continue;
        }

        cytoContours.push_back(cypoContours[i]);
    }

    if (1)
    {
        cv::Mat org = bgr.clone();
        cv::drawContours(org, cytoContours, -1, cv::Scalar(0, 255, 0), 1);
        saveMiddleImg(org, "org_cyto_contours.jpg");
    }

    return cytoContours;
}


std::vector<std::vector<cv::Point>> doChestProcessY(const cv::Mat& img, int *bg, int bin_thres)
{

    using cvContour = std::vector<std::vector<cv::Point>>;
    cv::Mat image;


    vector<string> vtag = { "b", "g", "r", "gray" };

    // find the cell contour in all vbgr

    vector<cv::Mat> vmasks;


    cv::Mat bin_img;
    cv::Mat gray = img;
    cv::threshold(gray, bin_img, bg[3] - bin_thres, 255, cv::THRESH_BINARY);
    bin_img = ~bin_img;
    auto mask_r = bin_img.clone();
    mask_r.setTo(0);

    std::vector<std::vector<cv::Point>> contours;

    std::vector<std::vector<cv::Point>> contours_roi;

    cv::findContours(bin_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int cnt = 0;
    for (const auto& e_contour : contours)
    {
        double area = cv::contourArea(e_contour);

        if (area > 20) // Example threshold, adjust as needed
        {
            // cv::drawContours(e_chn, std::vector<std::vector<cv::Point>>{e_contour}, -1, cv::Scalar(255), 1);

            // cv::drawContours(mask, std::vector<std::vector<cv::Point>>({ e_contour }), -1, cv::Scalar(255, 0, 0), cv::FILLED); // Filled contours  
            cv::drawContours(mask_r, contours, cnt, cv::Scalar(255, 0, 0), cv::FILLED); // Filled contours  
            // Draw the contours on the mask  

            contours_roi.push_back(contours[cnt]);


            // cv::drawContours(mask, contours, static_cast<int>(cnt), cv::Scalar(255), cv::FILLED); // Filled contours  



             // write a number in contour center
            cv::Moments moments = cv::moments(e_contour);
            cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
            // 在中心位置画数字
            //std::string text = std::to_string(cnt); // 转换为字符串，使用i+1作为标号
            // cv::putText(e_chn, text, center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(222, 222, 222), 1);

        }
        cnt++;
        // cv::drawContours(channel, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), 1);
    }

    cv::Mat img_b_r = img & mask_r;

    cimg ci; 




    auto img_all = ci.hconcat({ img, img_b_r }, 0);
    //ci.s_i(img_all);





    return contours_roi; 

    // auto cytoContours = calcCytoCervical(bgr, bg);
}


void doChestProcess(cv::Mat &bgr)
{

    using cvContour = std::vector<std::vector<cv::Point>>;
    cv::Mat image;

    int cutrows = 400;
    int cutcols = 400;
    int bin_thres = 86;

    int intv = 4;

    cv::Mat gray; // Assuming you need to pass a grayscale image
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    int bg[4];
    doCurrentBackground(bgr, gray, bg);

    // print all bg
    for (int i = 0; i < 4; i++)
    {
        cout << "bg[" << i << "] = " << bg[i] << endl;
    }

    cv::GaussianBlur(bgr, image, cv::Size(5, 5), 0, 0);

    std::vector<cv::Mat> vbgr;
    cv::split(bgr, vbgr);


    auto img_r = vbgr[2]; 
    auto img_g = vbgr[1];



    // vbgr.push_back(gray);

    cimg ci;


    int x = (img_r.cols - cutcols) / 2;
    int y = (img_r.rows - cutrows) / 2;
    img_r = img_r(cv::Rect(x, y, cutcols, cutrows));


    img_g = img_g(cv::Rect(x, y, cutcols, cutrows));

    // auto img_copy = ci.hconcat(vbgr, intv).clone();

    vector<string> vtag = {"b", "g", "r", "gray"};

    // find the cell contour in all vbgr

    vector<cv::Mat> vmasks;


    cv::Mat bin_img;
    cv::threshold(img_r, bin_img, bg[3] - bin_thres, 255, cv::THRESH_BINARY);
    bin_img = ~bin_img;
    auto mask_r = bin_img.clone();
    mask_r.setTo(0);

    std::vector<std::vector<cv::Point>> contours;

    std::vector<std::vector<cv::Point>> contours_roi;

    cv::findContours(bin_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int cnt = 0;
    for (const auto& e_contour : contours)
    {
        double area = cv::contourArea(e_contour);

        if (area > 20) // Example threshold, adjust as needed
        {
            // cv::drawContours(e_chn, std::vector<std::vector<cv::Point>>{e_contour}, -1, cv::Scalar(255), 1);

            // cv::drawContours(mask, std::vector<std::vector<cv::Point>>({ e_contour }), -1, cv::Scalar(255, 0, 0), cv::FILLED); // Filled contours  
            cv::drawContours(mask_r, contours, cnt, cv::Scalar(255, 0, 0), cv::FILLED); // Filled contours  
            // Draw the contours on the mask  

            contours_roi.push_back(contours[cnt]);


            // cv::drawContours(mask, contours, static_cast<int>(cnt), cv::Scalar(255), cv::FILLED); // Filled contours  



             // write a number in contour center
            cv::Moments moments = cv::moments(e_contour);
            cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
            // 在中心位置画数字
            //std::string text = std::to_string(cnt); // 转换为字符串，使用i+1作为标号
            // cv::putText(e_chn, text, center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(222, 222, 222), 1);

        }
        cnt++;
        // cv::drawContours(channel, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), 1);
    }

    cv::Mat img_b_r = img_g & mask_r;

    



    ci.s_i(img_b_r); 





    // auto cytoContours = calcCytoCervical(bgr, bg);
}

void findScanArea(const std::vector<cv::Point>& contour, std::vector<int>& scanX, std::vector<int>& scanY)
{
    // 确保轮廓不为空
    assert(!contour.empty());

    // 按 y 坐标对轮廓点进行排序（如果尚未排序）
    std::vector<cv::Point> sortedContour = contour;
    std::sort(sortedContour.begin(), sortedContour.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    });

    int currentY = sortedContour[0].y;
    scanY.push_back(currentY);
    scanX.push_back(sortedContour[0].x);
    scanX.push_back(sortedContour[0].x);

    // 遍历排序后的轮廓点
    for (size_t i = 1; i < sortedContour.size(); ++i) {
        if (sortedContour[i].y != currentY) {
            // 我们移动到了新的一行
            currentY = sortedContour[i].y;
            scanY.push_back(currentY);
            scanX.push_back(sortedContour[i].x);
            scanX.push_back(sortedContour[i].x);
        }
        else
        {
            scanX.back() = sortedContour[i].x;
        }
    }
}


bool isCircle(const std::vector<cv::Point>& contour) {

    vector<int> ScanX;
    vector<int> ScanY;
    findScanArea(contour, ScanX, ScanY);

    int count = 0;
    for (int i = 0; i < ScanY.size(); i++)
    {
        if (ScanX[2 * i + 1] - ScanX[2 * i] > 0)
        {
            count++;
        }
    }

    return count > 3;
}


bool filterContour(const std::vector<cv::Point>& contour)
{
    cv::Rect rect = cv::boundingRect(contour);
    if (rect.width < 6 || rect.height < 6) {
        // LOG_DEBUG << "rect.width = " << rect.width << ", rect.height = " << rect.height << endl;
        return false;
    }

    if (!isCircle(contour))
    {
        return false;
    }

    float pixelLength = 0.565;
    const int MAX_CORE_AREA = 55;
    const int MIN_CORE_AREA = 10;


    auto area = cv::contourArea(contour) * pixelLength * pixelLength;
    if (area > MAX_CORE_AREA || area < MIN_CORE_AREA)
    {
        // LOG_DEBUG << "area = " << area << endl;
        return false;
    }

    //判断不接近圆, 也过滤掉
    //计算圆度, 对于非圆形的封闭图形，其圆度值会小于1。图形越接近圆形，圆度值越接近1
    double perimeter = cv::arcLength(contour, true);
    double circularity = 12.566f * cv::contourArea(contour) / (perimeter * perimeter);
    // LOG_DEBUG << "circularity = " << circularity << endl;
    if (circularity < 0.5) {
        // LOG_DEBUG << "circularity = " << circularity << endl;
        return false;
    }

    return true;

}


void deal_bbox_diff(const cv::Mat& img, const cv::Mat& mask, cv::Mat& diffimg, const cv::Rect& bbox, float smallerscale = 1.0f, int thres=100)
{
    double sum = 0.0;
    int count = 0;





    for (int r = bbox.y + 1; r < bbox.y + bbox.height - 1; r++)
    {
        for (int c = bbox.x + 0; c < bbox.x + bbox.width - 1; c++)
        {

            if (mask.ptr(r)[c] != 0)
            {

               
                auto& e = img.ptr(r)[c];  auto& e_right = img.ptr(r)[c + 1];
                auto& e_down = img.ptr(r+1)[c];
                auto& e_upright = img.ptr(r - 1)[c + 1];
                auto& e_downright = img.ptr(r + 1)[c + 1];


                auto d_x = abs(e_right - e);
                auto d_y = abs(e_down - e);
                auto d_diag = abs(e_downright - e);
                auto d_vice_diag = abs(e_upright - e);

                int d_xydiag = (d_x + d_y + d_diag) / smallerscale;


                if (d_xydiag > thres)
                {
                    d_xydiag = thres;
                }

                

                diffimg.ptr(r)[c] = d_xydiag;
            }
        }
    }






}



bool isRectInImage(const cv::Rect& rect, const cv::Mat& img) {
    return (rect.x >= 0 && rect.y >= 0 &&
        (rect.x + rect.width) <= img.cols &&
        (rect.y + rect.height) <= img.rows);
}
void deal_bbox_d_diff(const cv::Mat& img, const cv::Mat& mask, cv::Mat& diffimg, const cv::Rect& bbox, float smallerscale = 1.0f, int thres = 100)
{
    double sum = 0.0;
    int count = 0;





    for (int r = bbox.y + 1; r < bbox.y + bbox.height - 1; r++)
    {
        for (int c = bbox.x + 0; c < bbox.x + bbox.width - 1; c++)
        {

            if (mask.ptr(r)[c] != 0)
            {


                auto& e = img.ptr(r)[c];  auto& e_right = img.ptr(r)[c + 1];
                auto& e_down = img.ptr(r + 1)[c];

                auto& e_downright = img.ptr(r + 1)[c + 1];
                auto& e_upright = img.ptr(r - 1)[c + 1];


                auto d_x = abs(e_right - e);
                auto d_y = abs(e_down - e);
                auto d_diag = abs(e_downright - e);
                auto d_vice_diag = abs(e_upright - e);
                int d_xydiag = (d_x + d_y + d_diag) / smallerscale;



                if (d_xydiag > thres)
                {
                    d_xydiag = 0;
                }



                diffimg.ptr(r)[c] = d_xydiag;
            }
        }
    }






}




cv::Mat clone_contour_bb(const cv::Mat& src_r0, const std::vector<cv::Point>& e_contour)
{
    auto maskB = src_r0.clone().setTo(0);
    cv::drawContours(maskB, std::vector<std::vector<cv::Point>>{e_contour}, -1, cv::Scalar(255), cv::FILLED);
    
    cv::Rect bb = cv::boundingRect(e_contour);

    cv::Point tl = bb.tl();  
    cv::Size2i bbsz = bb.size();

    // get bb width , height to a cvSize 
    auto tl_r0 = tl;
    auto bbsz_r0 = bbsz;

    tl -= cv::Point(2, 2);  

    // boundingBox size += 4
    bbsz.width += 4;
    bbsz.height += 4;


    // to judge if a rect in img range?
    auto bb_r0 = bb; 

   
    bb = cv::Rect(tl, bbsz);

    if (isRectInImage(bb, maskB))
    {
        // in side 
    }
    else
    {
        bb = bb_r0; 
    }

    // 从原始图像中提取 ROI
    cv::Mat imgroi = src_r0(bb).clone();
    cv::Mat maskroi = maskB(bb).clone();

    return imgroi & maskroi;

}



// find min but non-zero pixel location (x0,y0)
cv::Point2i findMinNonZeroPixel(const cv::Mat& img) {
    cv::Point2i minPoint(-1, -1);
    uchar minValue = 255; // Initialize to max value for uchar

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            uchar pixelValue = img.ptr<uchar>(r)[c];
            if (pixelValue > 0 && pixelValue < minValue) {
                minValue = pixelValue;
                minPoint = cv::Point2i(c, r);
            }
        }
    }
    return minPoint;
}

struct pixelValPos
{
    uchar ep;
    cv::Point2i exy;
};


std::vector<pixelValPos> getChestCellPixelStatInfo(const cv::Mat& img, int N, int* numofNonzero)
{


    // loop img and put all pixel to a vector<uchar>, also put its x,y to vector<cv::Point2i> 
    vector<uchar> pixels;
    vector<cv::Point2i> coordinates;

    vector<pixelValPos> vpxy;

    vector<pixelValPos> vFeatures = {};


    for (int r = 0; r < img.rows; r++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            auto& ep = img.ptr<uchar>(r)[c];
            if (ep != 0)
            {
                vpxy.push_back({ ep,{c,r} });
            }
        }
    }

    auto sz = vpxy.size();
    *numofNonzero = sz;

    if (sz < N * 2)
    {
        return vFeatures;
    }


    // sort vpxy by vpxy.ep
    std::sort(vpxy.begin(), vpxy.end(), [](const pixelValPos& a, const pixelValPos& b) {
        return a.ep < b.ep; // Assuming 'ep' is a member of the elements in vpxy
    });



    double avg_l = 0;
    double avg_h = 0;
    double avg_all = 0;



    vector<uchar> vstddev;

    // strip front , tail
    for (int i = 1; i < N + 1; i++)
    {
        avg_l += vpxy[i].ep;

        avg_h += vpxy[sz - 1 - i].ep;
    }

    int cnt_mid = 1;
    string sb;
    for (int i = 0; i < sz / 5 * 2; i++)
    {
        avg_all += vpxy[i].ep;
        // sb += to_string(int(vpxy[i].ep)) +  ",";
        cnt_mid++;
    }

    // LOG_DEBUG <<sb << endl;


    avg_l /= N;
    avg_h /= N;
    avg_all /= cnt_mid;


    uchar uc_l = avg_l;
    uchar uc_h = avg_h;

    auto minp = vpxy[0].exy;

    double avg_center = 0;

    for (int r = -2; r <= 2; r++)
    {
        int row_t = minp.y + r;

        if (row_t < 0 || row_t >= img.rows)
        {
            continue;
        }
        auto* er = img.ptr<uchar>(row_t);

        for (int c = -2; c <= 2; c++)
        {
            int col_t = minp.x + c;

            if (col_t < 0 || col_t >= img.cols)
            {
                continue;
            }

            if (er[col_t] < (uchar)(uc_l + 30))
            {
                vstddev.push_back(er[col_t]);
                avg_center += er[col_t];
            }
        }
    }

    auto stddevsz = vstddev.size();
    if (stddevsz == 0)
    {
        stddevsz = 1;
    }
    avg_center /= vstddev.size();

    double squaredDiffSum = 0.0;
    for (double value : vstddev)
    {
        squaredDiffSum += (value - avg_center) * (value - avg_center);
    }

    uchar stddev = squaredDiffSum / vstddev.size();




    vFeatures = { vpxy[0], vpxy[sz - 1] , {uc_l,{0,0}},{uc_h,{0,0}},   {stddev,{0,0}}, {(uchar)avg_all,{0,0}} };

    // vector<h,l, avg_h, avg_l);
    return vFeatures;
}


// 获取coreInnerxy位置的八领域的平均值
double getAverageOfNeighborhood(const cv::Mat& img, const cv::Point& coreInnerxy) 
{
    int sum = 0;
    int count = 0;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int x = coreInnerxy.x + j;
            int y = coreInnerxy.y + i;

            if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
                sum += img.ptr<uchar>(y)[x];
                count++;
            }
        }
    }

    return count > 0 ? static_cast<double>(sum) / count : 0.0;
}


#define  CONST_LHAVGLHSTD_EXPECTED 6




cv::Mat  toGrayByStripMin(const cv::Mat & img_)
{
    
    // ci.img , {b,g,r} get min, then strip min from b,g,r, and average the other two
    //auto& img = ci.img;
    auto img = img_.clone();
    assert(img.channels() == 3); // 确保图像是三通道的
    vector<cv::Mat> vbgr;;
    cv::split(img, vbgr); // 将图像拆分为三个通道

    auto gray = vbgr[0].clone();

    int sum = 0;
    int cnt = 0;

    for (int r = 0; r < img.rows; r++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
            uchar & e_gray = gray.ptr<uchar>(r)[c];

            vector<uchar> vp = { pixel[0], pixel[1], pixel[2] };
            std::sort(vp.begin(), vp.end());
            e_gray = (vp[1] * 0.5 + vp[2] * 0.5);

            if (e_gray == 255)
            {
                e_gray = 230;
            }


            sum += e_gray;
            cnt++;

        }
    }


    auto avg = sum / cnt;

    
    return gray; 

}
struct para_con
{
    int cen_avg;
    int flag_filterout;
    std::vector<cv::Point> * p_e_c;
};



// 计算直线 y = y0 与轮廓 c0 相交的两个点  
std::vector<Point2f> getIntersectionPoints(const std::vector<Point>& contour, float y0) 
{
    vector<Point2f> intersectionPoints;

    // 遍历每条边  
    for (size_t i = 0; i < contour.size(); ++i) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()]; // 下一个点，循环返回起始点  

        // 检查两点的 y 坐标范围  
        if ((p1.y <= y0 && p2.y >= y0) || (p1.y >= y0 && p2.y <= y0)) {
            // 计算交点的 x 坐标  
            float x = p1.x + (p2.x - p1.x) * (y0 - p1.y) / (p2.y - p1.y);
            intersectionPoints.push_back(Point2f(x, y0));
        }
    }

    // 返回找到的交点  
    return intersectionPoints;
}



// 计算两个点之间的单位法向量  
Point2f getNormal(const Point& p1, const Point& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float length = std::sqrt(dx * dx + dy * dy);
    return Point2f(-dy / length, dx / length); // 逆时针旋转 90 度  
}

// 主函数：将轮廓向内收缩  
// proportion 取值范围为 (0, 1)，其中 0.5 表示收缩 50%。  
const std::vector<cv::Point> shrinkC(const std::vector<cv::Point>& contour, float proportion) {
    std::vector<cv::Point> newContour;

    if (contour.size() < 3) return newContour; // 至少需要三个点  

    for (size_t i = 0; i < contour.size(); ++i) {
        Point p1 = contour[i];
        Point p2 = contour[(i + 1) % contour.size()]; // 下一个顶点  
        Point p0 = contour[(i + contour.size() - 1) % contour.size()]; // 上一个顶点  

        // 计算法向量  
        Point2f normal1 = getNormal(p1, p2);
        Point2f normal2 = getNormal(p0, p1);

        // 计算平均法向量  
        Point2f averageNormal = (normal1 + normal2) * 0.5; // 找到平均法向量  

        // 计算到相邻点的距离，用于控制收缩  
        float distance = cv::norm(p2 - p1); // 点 p1 到 p2 之间的距离  

        // 收缩后的新点，向内收缩 proportion  
        Point newPoint(
            static_cast<int>(p1.x + averageNormal.x * proportion * distance),
            static_cast<int>(p1.y + averageNormal.y * proportion * distance)
        );

        newContour.push_back(newPoint);
    }

    return newContour; // 返回收缩后的轮廓  
}


cv::Mat img_min_nonzero(const cv::Mat &img)
{

    struct pos_pv
    {
        int pos; 
        int pv; 
    };

    vector<pos_pv> v_pos_pv(img.cols, {0, 255});
    auto img_ret = img.clone().setTo(0);

    for (int r = 0; r < img.rows; r++)
    {
        
        int sum = 0; 
        int cnt = 0; 
        int avg = 0;
        for(int c=0;c<img.cols;c++)
        {
            auto ev = img.ptr<uchar>(r)[c];
            if (ev == 0)
            {
                ev = 255;
            }
            else
            {
                sum += ev; 
                cnt++;
            }

            pos_pv e_p = {c, ev};
            
            
            v_pos_pv[c] = e_p; 
        }

        avg = sum / cnt; 



        

        std::sort(v_pos_pv.begin(), v_pos_pv.end(), [](const pos_pv& a, const pos_pv& b) {
            return a.pv < b.pv; 
        });

        cnt = 0;
        for (int i = 0; i < v_pos_pv.size(); i++)
        {
            //if (cnt < v_pos_pv.size()/10)
            //{
            //    cnt++;
            //    continue;
            //}
            if (cnt > v_pos_pv.size() / 4 )break;

            auto e = v_pos_pv[i]; 
            e.pos; 
            e.pv; 

            if (e.pv < avg *0.88)
            {
                img_ret.ptr<uchar>(r)[e.pos] = 255;
            }
            
            cnt++;
        }







     
        

    }



    return img_ret;


}



std::pair<cv::Mat, std::vector<std::vector<cv::Point>>>  findCellClusterContour(cv::Mat & gray)
{
    const int step_r = 32;
    const int step_c = step_r;
    const int overlap = step_r / 2;
    const int thres_bg = 191;
    const int thres = 80;
    const int thres_cnt_roi = 100;
    const int thres_area = 10000;
    const float resize_factor = 0.41f;  // 加上 'f' 表示该值为 float  
    const int CORE_MASK_PIXEL_VAL = 128;

    const int size_e_row = 32;




    std::pair<cv::Mat, std::vector<std::vector<cv::Point>>> p_con_roi;
    auto & v_con_roi = p_con_roi.second; 


    //  cimg ci; 

    auto img = gray;

    auto gray_r0 = gray.clone(); 


    struct bb_cnt
    {
        cv::Rect bb;  // boundingBox
        int cnt; // feature pixels cnt
        // int row; // bb row idx
        // int col; // bb col idx
    };

    vector<bb_cnt> v_bb_cnt;


    // get gray intense value bb, and draw on img as white block
    int row_cut = 0;

    for (int r = 0; r < img.rows - step_r; r += (step_r - overlap))
    {
        int col_cut = 0;
        for (int c = 0; c < img.cols - step_c; c += (step_c - overlap))
        {

            cv::Rect bb = cv::Rect(c, r, step_c, step_r);

            cv::Mat img_roi = gray_r0(bb).clone();

            int cnt = 0;
            int cnt_bg = 1;
            img_roi.forEach<uchar>(
            [&thres, &thres_bg, &cnt, &cnt_bg](uchar& pixel, const int* position)
            {
                if (pixel < thres)
                {
                    cnt++;
                }
            });


            if (cnt > thres_cnt_roi)
            {
                //  bb_cnt e_bb_cnt = { bb, cnt, row_cut, col_cut };
                bb_cnt e_bb_cnt = {bb, cnt};
                v_bb_cnt.push_back(e_bb_cnt);
            }

            col_cut++;
        }

        row_cut++;
    }

    // std::sort(v_bb_cnt.begin(), v_bb_cnt.end(), [](const bb_cnt& a, const bb_cnt& b) { return (a.cnt > b.cnt); });


    img = gray_r0.clone();


    // v_bb_cnt_roi = v_bb_cnt;

    for (auto & e_bb_cnt : v_bb_cnt)
    {
        auto & bb = e_bb_cnt.bb;
        // cv::Point center = cv::Point(bb.x + bb.width / 2 - 10, bb.y + bb.height / 2);
        cv::rectangle(img, bb, cv::Scalar(255), cv::FILLED);
    }

    cv::threshold(img, img, 254, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> v_contours;
    cv::findContours(img, v_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (auto& e_c : v_contours)
    {
        auto area = cv::contourArea(e_c);

        if (area > thres_area)
        {
            // v_con_area.push_back(e_con_area);
            std::vector<cv::Point> e_c_;
            cv::convexHull(e_c, e_c_);
            v_con_roi.push_back(e_c_);
        }

    }



    // sort by v_con_area.area
    // std::sort(v_con_area.begin(), v_con_area.end(), [](const auto& a, const auto& b) { return a.area > b.area; });

    img = gray_r0.clone();

    // draw contour on ci.img 
    //img_contour_line = cv::Mat::zeros(img.size(), CV_8UC1);;
    cv::drawContours(img, v_con_roi, -1, cv::Scalar(255), cv::FILLED);

    

    

    cv::threshold(img, img, 255 - 1, CORE_MASK_PIXEL_VAL, cv::THRESH_BINARY);  // to 128



    v_con_roi.clear();
    cv::findContours(img, v_con_roi, cv::RETR_LIST, cv::CHAIN_APPROX_NONE); // get NONE contour


    // cv::resize(img, img, cv::Size((int)(img.cols * resize_factor), int(img.rows * resize_factor)));
    

    p_con_roi.first = img.clone();

    

    cout << p_con_roi.first.rows << ":" << p_con_roi.second.size() << endl;
    return p_con_roi;
}


void test_const_contour( vector<std::vector<Point>>  & v_con)
{

    for (auto& e : v_con)
    {
        cout << e.size() << endl; 
        e.pop_back();
        cout << e.size() << endl;
    }
    
}

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

bool IsContourBeTheSame(const vector<cv::Point>& contour1, const vector<cv::Point>& contour2) {
    bool ret_same = false;

    // 计算两个轮廓的中心点
    cv::Moments m1 = cv::moments(contour1);
    cv::Moments m2 = cv::moments(contour2);

    const float lowestF = 1e-6;
    if (abs(m1.m00) < lowestF) {
        m1.m00 = lowestF;
    }
    if (abs(m2.m00) < lowestF) {
        m2.m00 = lowestF;
    }

    cv::Point2f center1(m1.m10 / m1.m00, m1.m01 / m1.m00);
    cv::Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);

    // 计算中心点的距离
    double centerDistance = cv::norm(center1 - center2);

    // 计算中心点的相对偏差
    double maxWidth = max(cv::boundingRect(contour1).width, cv::boundingRect(contour2).width);
    double centerDeviation = centerDistance / maxWidth;

    // 如果中心点偏差超过20%，提前返回
    if (centerDeviation > 0.2) {
        return ret_same;
    }

    // 计算两个轮廓的面积
    double area1 = cv::contourArea(contour1);
    double area2 = cv::contourArea(contour2);

    // 计算面积的相对偏差
    double areaDeviation = abs(area1 - area2) / max(area1, area2);

    // 判断面积的相对偏差是否在38%以内
    if (areaDeviation <= 0.38) {
        ret_same = true;
    }

    return ret_same;
}



std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }

    return tokens;
}

// Function to parse points from a string like "[x1, y1],[x2, y2],..."
std::vector<cv::Point2i> parsePoints(const std::string& pointsStr) {
    std::vector<cv::Point2i> points;
    std::string str = pointsStr;

    // Process each [x, y] pair
    size_t pos = 0;
    while (pos < str.length()) {
        // Find the start and end of the point definition
        size_t startPos = str.find('[', pos);
        size_t endPos = str.find(']', startPos);

        if (startPos == std::string::npos || endPos == std::string::npos) {
            break;
        }

        // Extract the x,y part
        std::string pointPair = str.substr(startPos + 1, endPos - startPos - 1);

        // Split by comma
        size_t commaPos = pointPair.find(',');
        if (commaPos != std::string::npos) {
            int x = std::stoi(pointPair.substr(0, commaPos));
            int y = std::stoi(pointPair.substr(commaPos + 1));
            points.push_back(cv::Point2i(x, y));
        }

        // Move to the next point
        pos = endPos + 1;
    }

    return points;
}

std::vector<std::vector<cv::Point2i>> readPointsFromFile(const std::string& filename, const std::string & roiStr) 
{
    // Vector to store all the points from the file
    std::vector<std::vector<cv::Point2i>> allPoints;

    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return allPoints; // Return empty vector on error
    }

    std::string line;
    while (std::getline(file, line)) {
        // Split the line by semicolon

        std::vector<std::string> parts = split(line, ';');

        // Make sure we have at least 4 parts (filename, two numbers, and points)
        if (parts.size() < 4) {
            std::cerr << "Warning: Invalid line format: " << line << std::endl;
            continue;
        }

        if (parts[0] != roiStr)
        {
			//std::cerr << "Warning: Invalid line format: " << line << std::endl;
			continue;
        }

        // Parse the points from the fourth part
        std::vector<cv::Point2i> linePoints = parsePoints(parts[3]);

        // Add to our collection
        allPoints.push_back(linePoints);
    }

    file.close();
    return allPoints;
}


struct TotalFieldPara {
    wchar_t BianHao[20], Unit[6];
    float PixelLength;
    int FieldNum, CellNum, CervicCellNum;
    int RefCellNumber;
    float RefCellODSum, RefCellODMean, RefCellODMeanLow;
    bool IsDNAIndexCaculated;
    char ScanType[8], UnMixModel;
    bool IsYangxing, IsMeaDAB;
    char Reserved[13];
};


bool get_IsDNAIndexCaculated(const char* filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open file");
    }

    TotalFieldPara para;
    if (!in.read(reinterpret_cast<char*>(&para), sizeof(TotalFieldPara)))
    {
        throw std::runtime_error("Failed to read header");
    }

    return para.IsDNAIndexCaculated;
}

// 修改头部IsDNAIndexCaculated, 保持尾部数据不变
bool update_IsDNAIndexCaculated(const char* filename, bool newValue)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in)
    {
        std::cerr << "Cannot open input file.\n";
        return false;
    }
    std::streamsize filesize = in.tellg();
    if (filesize < (std::streamsize)sizeof(TotalFieldPara))
    {
        std::cerr << "File too small.\n";
        return false;
    }
    in.seekg(0, std::ios::beg);

    // 读取头部
    TotalFieldPara para;
    if (!in.read(reinterpret_cast<char*>(&para), sizeof(TotalFieldPara)))
    {
        std::cerr << "Read head failed.\n";
        return false;
    }

    // 读取尾部
    std::streamsize tail_size = filesize - sizeof(TotalFieldPara);
    char* tail = new char[tail_size];
    if (!in.read(tail, tail_size))
    {
        std::cerr << "Read tail failed.\n";
        delete[] tail;
        return false;
    }
    in.close();

    // 修改
    para.IsDNAIndexCaculated = newValue;

    // 写回
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        std::cerr << "Cannot open output file for writing.\n";
        delete[] tail;
        return false;
    }
    out.write(reinterpret_cast<char*>(&para), sizeof(TotalFieldPara));
    out.write(tail, tail_size);

    delete[] tail;
    return true;
}






std::vector<int> searchAllComIdx(int max_port = 256)
{

    auto com_exists = [](int port) -> bool 
    {
           char name[20];
           snprintf(name, sizeof(name), "\\\\.\\COM%d", port);
           HANDLE h = CreateFileA(name, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
           if (h != INVALID_HANDLE_VALUE)
           {
               CloseHandle(h);
               return true;
           }
           return false;
       };


    std::vector<int> result;
    for (int i = 1; i <= max_port; ++i)
    {
        if (com_exists(i))
        {
            result.push_back(i);
        }
    }
    return result;
}


// pad image to square, center the original image, pad value as border
cv::Mat padToSquare(const cv::Mat& img)
{
    int h = img.rows;
    int w = img.cols;
    int size = max(h, w);

    assert(img.channels() == 1 || img.channels() == 3); 

	cv::Scalar border_value = cv::Scalar(215, 215, 215); // Default border value

    if (img.channels() == 1)
    {
        border_value = cv::Scalar(0);
    }

    int pad_top = (size - h) / 2;
    int pad_bottom = size - h - pad_top;
    int pad_left = (size - w) / 2;
    int pad_right = size - w - pad_left;

    cv::Mat square_img;
    cv::copyMakeBorder(img, square_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, border_value);
    return square_img;
}

// reverse pad2square: crop the center region to recover the original size
cv::Mat reversePadTosquare(const cv::Mat& square_img, const cv::Size& old_size)
{
    int size = square_img.rows; // square, so rows==cols
    int w = old_size.width;
    int h = old_size.height;

    int x = (size - w) / 2;
    int y = (size - h) / 2;

    x = max(0, x);
    y = max(0, y);

    int crop_w = min(w, size - x);
    int crop_h = min(h, size - y);

    return square_img(cv::Rect(x, y, crop_w, crop_h)).clone();
}


bool endsWith(const std::string& str)
{
    const std::string suffix = ".y.svs";
    if (str.length() < suffix.length())
    {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}


void TileImageToFill(const cv::Mat& src, cv::Mat& dst)
{
    if (src.empty() || dst.empty())
    {
        return;
    }

    int src_rows = src.rows;
    int src_cols = src.cols;
    int dst_rows = dst.rows;
    int dst_cols = dst.cols;

    // 计算水平和垂直方向需要的重复次数
    int h_repeats = (dst_cols + src_cols - 1) / src_cols;
    int v_repeats = (dst_rows + src_rows - 1) / src_rows;

    // 逐块复制图像
    for (int i = 0; i < v_repeats; ++i)
    {
        for (int j = 0; j < h_repeats; ++j)
        {
            int top = i * src_rows;
            int left = j * src_cols;

            // 计算当前区块的实际宽高（处理边界情况）
            int roi_width = min(src_cols, dst_cols - left);
            int roi_height = min(src_rows, dst_rows - top);

            if (roi_width <= 0 || roi_height <= 0)
            {
                continue;
            }

            // 创建目标区域
            cv::Rect dst_roi(left, top, roi_width, roi_height);
            cv::Mat dst_region = dst(dst_roi);

            // 创建源区域（可能需要裁剪源图像）
            cv::Rect src_roi(0, 0, roi_width, roi_height);
            cv::Mat src_region = src(src_roi);

            // 复制图像
            src_region.copyTo(dst_region);
        }
    }
}

vector<vector<Point>> convert_roi_contours_to_big(const vector<vector<Point>>& roi_contours, const Rect& roi_rect)
{
    vector<vector<Point>> big_contours;
    for (const auto& contour : roi_contours)
    {
        vector<Point> big_contour;
        for (const auto& pt : contour)
        {
            big_contour.push_back(Point(pt.x + roi_rect.x, pt.y + roi_rect.y));
        }
        big_contours.push_back(big_contour);
    }
    return big_contours;
}

float countPixelsRateOnLines(const cv::Mat& grayImage, unsigned char thresGray)
{

	if (grayImage.channels() != 1)
	{
		throw std::invalid_argument("Input image must be a grayscale image.");
	}
    int totalPixelCount = 0;
    int sumThres = 0;

    int height = grayImage.rows;
    int width = grayImage.cols;

    // Calculate positions for the 3 horizontal lines
    int hLine1 = height / 4;
    int hLine2 = height / 2;
    int hLine3 = height * 3 / 4;

    // Calculate positions for the 3 vertical lines
    int vLine1 = width / 4;
    int vLine2 = width / 2;
    int vLine3 = width * 3 / 4;

    // Process horizontal lines
    for (int line : {hLine1, hLine2, hLine3})
    {
        for (int x = 0; x < width; ++x)
        {
            uchar pixelValue = grayImage.at<uchar>(line, x);

            totalPixelCount++;
            if (pixelValue < thresGray && pixelValue > 9)
            {
                sumThres++;
            }
        }
    }

    // Process vertical lines
    for (int line : {vLine1, vLine2, vLine3})
    {
        for (int y = 0; y < height; ++y)
        {
            uchar pixelValue = grayImage.at<uchar>(y, line);

            totalPixelCount++;
            if (pixelValue < thresGray)
            {
                sumThres++;
            }
        }
    }

    return sumThres * 1.0f / totalPixelCount;
}

cv::Rect expand_rect(const cv::Rect& rect, float expand_ratio, const cv::Mat& img)
{
    // Calculate expansion amount
    int dx = static_cast<int>(rect.width * expand_ratio * 0.5);
    int dy = static_cast<int>(rect.height * expand_ratio * 0.5);

    // Create expanded rectangle
    cv::Rect expanded_rect;
    expanded_rect.x = rect.x - dx;
    expanded_rect.y = rect.y - dy;
    expanded_rect.width = rect.width + 2 * dx;
    expanded_rect.height = rect.height + 2 * dy;

    // Ensure the expanded rectangle stays within image boundaries
    expanded_rect.x = max(0, expanded_rect.x);
    expanded_rect.y = max(0, expanded_rect.y);
    expanded_rect.width = min(expanded_rect.width, img.cols - expanded_rect.x);
    expanded_rect.height = min(expanded_rect.height, img.rows - expanded_rect.y);

    return expanded_rect;
}


#include <opencv2/opencv.hpp>

template<typename T>
cv::Mat zeroMainDiagonal(const cv::Mat& mat)
{
    cv::Mat result = mat.clone();
    int n = min(result.rows, result.cols);
    for (int i = 0; i < n; ++i)
    {
        result.at<T>(i, i) = 0;
    }
    return result;
}



class bv {
public:
	virtual void f() = 0; // 纯虚函数
    virtual void f1() = 0; 

};

class sv_0 : public bv {

public:
	void f() override {
		std::cout << "Derived class implementation of f()" << std::endl;
	}

	void f1() override {
		std::cout << "Derived class implementation of f1()" << std::endl;
	}
};


#if 1

class sv_1 : public bv {

public:
    void f() override {
        std::cout << "Derived class implementation of f()" << std::endl;
    }


};
#endif 

#endif






// main_
int main(int argc, char **argv)
{

    //  global var  //

    com &ec = s_com;
    ec.glo_init(argc, argv);
    ecl(string(argv[0]) + " start__");

    // string dirname = "D:\\jd\\t\\platform_test_data\\";

    // dirname = string("D:/jd/t/img_rgb_cmp/_309/sz_big/");
    string fn_r0 = "d:/jd/t/rgb_10_12.jpg";
    cimg ci;
    cimg ci_0;
    cimg ci_1;
    cimg ci_2;

    // -------- //

#if 1

    rex  ids("AB3C");

    rex_match idm = m(R"(B)");

    bool ism = ids % idm;
    cout << ism << endl;

    ids % s(R"(\d)", R"(__)");


    cout << ids.str() << endl;


    


#endif 
#if 0

    unordered_map<int, int> mii = {
        {0,0},
        {1,11},
		{2,22},
    
    };

    cout << mii.at(3) << endl; 

#endif 

#if 0

    string a = "STRING"; 

    vector<string> vi; 
	vi.push_back(std::move(a));

    cout << "a:"<< a << endl;
    cout << vi[0] << endl; 




#endif 

#if 0

	unordered_map<int, int> m_i_i; // key: idx, value: <img, mask>

    m_i_i[0] = 0; 
    m_i_i[1] = 11; 
	m_i_i[2] = 22;
    m_i_i[1] = 111; 
// cout m_i_i
	cout << s_("m_i_i : {}", m_i_i) << endl;

    int key = 2; 
    if (m_i_i.find(key) != m_i_i.end())
	{
		cout << "m_i_i has key " << key << endl;
	}
	else
	{
		cout << "m_i_i has no key " << key << endl;
	}




	

#endif 
#if 0
    sv_0 id_sv; 

    id_sv.f1(); 
    id_sv.f(); 

#endif 
#if 0
    string fn = "D:/jd/t/t0/t0/img_hard_37_82.png"; 

    ci.read_img(fn); 

    ci.img = zeroMainDiagonal<v3b>(ci.img);
    ci.s_i();






#endif 


#if 0
    for (int i = 0; i < 4; i++)
    {
        
        bool flag0 = false;
        int input_number = i;

        bool flag_rerun = false;

    RE_RUN:
        
        if (flag0 == true)
        {

            flag0 = false; 

            cout << "- RE_RUN " << endl; 

            flag_rerun = true; 
            input_number <<= 8;

        }

        vector<int> va; 

        va.push_back(1); 

        if (flag_rerun == false && input_number % 2)
        {

            flag0 = true; 
			goto RE_RUN;

        }


		cout << "input_number:" << input_number << endl;


        cout << s_("va is: {}", va); 

    }

    
    assert(0 == 1);
#endif 
#if 0

    vector<cv::Mat> vimgall; 
    vector<string> vfn = ci.get_file_list("D:/jd/t/t0/t0/img_*.png"); 



    static int cnt = 0; 
    for (auto& efn : vfn)
    {
        string fn_img = efn; 
        string fn_mask = ci.replace_str(efn, "img_", "mask_");

        ci.read_img(fn_img); 
        ci_0.read_img(fn_mask); 
        ci_0.cvtcolor("RGB"); 


        auto himg = ci.hconcat({ ci.img, ci_0.img });
		vimgall.push_back(himg.clone());
       
        if (vimgall.size() % 30 == 0 && vimgall.size()>0)
        {
            auto imgallv = ci.vconcat(vimgall); 
            cv::imwrite(s_("d:/jd/t/t0/t0/{}.png", cnt), imgallv); 
            vimgall.clear();
        }

        cnt++;
    }




#endif 


#if 0
	string fn = "d:/jd/t/1.png";

	cv::Rect bb = cv::Rect(100, 100, 300, 400);

	ci.read_img(fn);
	auto img_roi = ci.img(bb).clone();

	auto bb_ = expand_rect(bb, 0.3, ci.img);
	auto img_roi_ = ci.img(bb_).clone();

	auto img_all = ci.hconcat({ img_roi, img_roi_ });

	ci.img = img_all;
	ci.s_i();




#endif 



#if 0


#if 0
    vector<string> vfn = {
    "img_105_34_0.png", 
    "img_104_44_0.png", 
    "img_103_46_0.png",
    "img_92_42_0.png",
    "img_90_63_0.png",
    "img_83_54_0.png",
    "img_93_46_0.png",
    "img_64_71_0.png",
    "img_52_63_0.png",
    "img_37_42_0.png",
    "img_38_34_0.png",
    "img_35_34_0.png",
    };
#endif 

	struct fn_pv
	{
		string fn;
		float rate;
	};


    vector<fn_pv> vfn_rate = {};
	vector<string> vfn = ci.get_file_list("d:/jd/t/t0/t0/*.png");

    for (auto& efn : vfn)
    {
        string fn = s_("{}", efn);

		ci.read_img(fn);

        cv::cvtColor(ci.img, ci.img, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> v_img;
        cv::split(ci.img, v_img);
		ci.img = v_img[2]; // V channel

		// ci.cvtcolor("GRAY");

        auto p_roi_sum =  countPixelsRateOnLines(ci.img, 122);

		//cout << s_("{}\t{}", efn, p_roi_sum) << endl;

		//ecl(s_("{}\t{}", efn, p_roi_sum));
    
		vfn_rate.push_back({ efn, p_roi_sum });
    }


	// sort vfn_rate by rate 
	std::sort(vfn_rate.begin(), vfn_rate.end(), [](const fn_pv& a, const fn_pv& b) {
		return a.rate < b.rate;
	});

	for (auto& e : vfn_rate)
	{
		cout << s_("{}\t{}", e.fn, e.rate) << endl;
        ecl(s_("{}\t{}", e.fn, e.rate));

	}

	
#endif 


#if 0
    string fn = "d:/jd/t/2.jpg";
    ci.read_img(fn); 

    auto img_rgb_r0 = ci.img.clone();

	ci.cvtcolor("GRAY");
	ci.threshold(100);
    ci.img = ~ci.img;


    auto img_show = ci.find_contours();

    auto con_all = ci.v_contours; 


    for (auto & econ : con_all)
    {
		auto bb = boundingRect(econ);

        if (bb.width < 10 && bb.height < 10)
        {
            continue;
        }


		auto img_roi = img_rgb_r0(bb).clone();

        ci.s_i(img_roi); 


       
        ci_0.img = img_roi.clone(); 

        ci_0.cvtcolor("GRAY"); 

        ci_0.threshold(100);
        ci_0.img = ~ci_0.img; 


		auto img_roi_contour = ci_0.find_contours();
		auto con_roi = ci_0.v_contours;


       

   


        auto con_roi_bb = convert_roi_contours_to_big(con_roi, bb);


		cout << con_roi_bb.size() << endl;


        // cout << s_("{}",con_roi) << endl;







    }


    assert(0 == 1);

    

    ci_0.img = img_show; 

    ci_0.s_i(); 



    

    // ci.s_i();

#endif 

#if 0
    ci.read_img("d:/jd/t/git/images/analysis_result/isPicStandardBenchmark/middle/8_8_overlay_after_set_to_9_0.png");

    ci.img; 
    cout << ci.img.rows << endl; 
#endif 

#if 0
    // ci.read_img("d:/jd/t/8_8_y_do_unmixy.jpg");
    ci.read_img("d:/jd/t/8_8_rgb_org.jpg");

    ci.img;
    int w = 2464;
    int h = 2056;
    int rows = h;
    int cols = w;

    auto img_dst = ci.create_img_rc_chn(rows, cols, ci.img.channels());
    img_dst.setTo(0);

    // 平铺 ci.img 直到铺满 img_dst
    TileImageToFill(ci.img, img_dst);

    ci.img = img_dst; 
    ci.s_i();

    img_dst;
	cv::imwrite("d:/jd/t/rgb.jpg", img_dst);

#endif 
#if 0
    
    int x = 0;

    int t = 1 / x; 

    map<string, string> m_s_s; 

    m_s_s = {
        {"a", "1"},
        {"b", "2"},
		{"c", "3"},
		{"d", "4"}
               
    };
    

    cout << m_s_s.at("a") << endl;;

#endif 
#if 0

	cv::Mat img = ci.create_img_rc_chn(600, 400, 1);

    for (int i = 0; i < 22; i++)
    {
        for (int j = 0; j < 22; j++)
        {
            img.ptr<uchar3>(i)[j] = uchar3(0, 0, 0); 
        }
    }

    auto img_old_size = img.size(); 

    (img_old_size.width, img_old_size.height);
    

    auto pad_img = padToSquare(img); 


	auto img_new =  reversePadTosquare(pad_img, img_old_size);

    cout << 1 << endl;



    // pad img to square 
	// img_square = pad2square(img, cv::Scalar(0, 0, 0)); //TODO
    // reverse size from img_square to old img 
	// img = rev_pad2square(img_square,img_old_size); //TODO

    
  





#endif 
#if 0

    ci.img = ci.create_img_rc_chn(200, 500, 3); 




    ci.s_i(); 


    for (int i=0;i<ci.img.dims;i++)
	{
		cout << ci.img.size[i] << " ";
	}

    cout << ci.img.size() << endl; 


#endif 



#if 0
    string fn = "d:/jd/t/git/dna-analysis/images/analysis_result/DAcervicHard/middle/8_8_fill_binaryImage.jpg"; 

    ci.read_img(fn);

    ci.img = ci.img({ 300,300+568 }, { 600,600+1024 }); 

    ci.resize(0.4); 
    ci.s_i(); 


    cv::imwrite("d:/jd/t/6.jpg", ci.img); 




#if 0

    auto img_rgb_r0 = ci.img.clone();  


    ci.cvtcolor("GRAY");

    auto img_gray = ci.img.clone(); 

	auto img_avg_h2 = img_gray.clone();

    ci.img = img_rgb_r0.clone(); 

    std::vector<cv::Mat> v_img;
    cv::split(ci.img, v_img);

    for (int r = 0; r < ci.img.rows; r++)
    {
        for (int c = 0; c < ci.img.cols; c++)
        {

			vector<uchar> vuc = { v_img[0].ptr<uchar>(r)[c], v_img[1].ptr<uchar>(r)[c], v_img[2].ptr<uchar>(r)[c] };
			std::sort(vuc.begin(), vuc.end());

            auto max_p = vuc[2];
            auto mid_p = vuc[1];
            auto pavg = (max_p + mid_p) / 2;

			

			// 设置灰度图像的像素值
            img_avg_h2.ptr<uchar>(r)[c] = pavg;

        }
    
    }


	// img_rgb_r0, img_gray, img_avg_h2

    ci.img = img_rgb_r0.clone(); 
    ci_0.img = img_gray.clone();  ci_0.cvtcolor("RGB"); 


	auto img_av2_h2_c1 = img_avg_h2.clone();

    



	ci_1.img = img_avg_h2.clone(); ci_1.cvtcolor("RGB");





    ci.resize(0.4);
    ci_0.resize(0.4);
    ci_1.resize(0.4);


    vector<cv::Mat> vmats = { ci.img, ci_0.img, ci_1.img };
    auto img_all = ci.vconcat(vmats, 11);

	ci_2.img = img_all;
    //ci_2.resize(0.6); 
    ci_2.s_i(); 



    int cnt = 0;
    for (auto eimg : vmats)
    {
        ci.img = eimg; 
        //ci.resize(0.5);
		eimg = ci.img.clone();
        // cv::imwrite(s_("d:/jd/t/{}.jpg", cnt), eimg);

        cnt++;
    }



    ci.img = img_av2_h2_c1.clone(); 

    ci.threshold(188); 

    ci.img = ~ci.img;


    auto img_cyto_mask = ci.img.clone(); 

	ci_1.img = img_cyto_mask.clone();
    ci_1.resize(0.4); 

    cv::imwrite("d:/jd/t/4.jpg", ci_1.img); 



    ci.img = ci.img & img_av2_h2_c1;


    ci.resize(0.4);
    cv::imwrite("d:/jd/t/5.jpg", ci.img);
    


    assert(0 == 1);








    ci.resize(0.4);   ci.s_i();

	cv::imwrite("d:/jd/t/3.jpg", ci.img);

 




#endif 











#endif 
#if 0
    string sa = "DTP2504092_00_04.jpg;10291;1;[3626, 392],[3624, 394],[3622, 394]"; 

    auto loc = sa.find(";"); 
    cout << loc << endl; 

    cout << sa.substr(0, loc); 

    auto t0 = GetTickCount64(); 
    auto arr = searchAllComIdx();
	float d = GetTickCount64() - t0;

    cout_("{}, {}", arr, d/1000); 


#endif 
#if 0

    string fn = "d:/jd/t/t0/20241223005_08_09_OUT.png"; 

    ci.read_img(fn);

    cout << ci.img << endl;




#endif 
#if 0

     update_IsDNAIndexCaculated("D:/dnaana/meadata/Mea08010004.fea", false); 


	 cout << get_IsDNAIndexCaculated("D:/dnaana/meadata/Mea08010004.fea") << endl;


#endif 

#if 0
    string fn = "d:/jd/t/1.bmp";
    ci.read_img(fn); 

    ci.cvtcolor("GRAY");
    
    ci.img; 



    cv::imwrite("d:/jd/t/1.png", ci.img);


#endif 
#if 0
    string fn = "d:/jd/t/08_08_OUT.png"; 

	ci.read_img(fn);
	ci.img;
    cout << 1 << endl; 

    // count ci.img non-zero pixel number
	int non_zero_cnt = 0;
	for (int r = 0; r < ci.img.rows; r++)
	{
		for (int c = 0; c < ci.img.cols; c++)
		{
			auto& e = ci.img.ptr<uchar>(r)[c];
			if (e > 0)
			{
				non_zero_cnt++;
			}
		}
	}

    cout << non_zero_cnt << endl; 


#endif 
#if 0
	string fn = "d:/jd/t/t1/1.jpg";


   
	ci.read_img(fn);

    ci.img;


    ci.cvtcolor("GRAY");




    ci.threshold(88, 255); 


    

	


    uchar old = ci.img.ptr<uchar>(0)[0];
// loop ci.img 
    int cnt = 0;
    int sum = 0;

    int cnt_break = 0;
	for (int r = 0; r < ci.img.rows; r++)
	{
        for (int c = 0; c < ci.img.cols; c++)
        {

            sum++;
            cnt++;
            if (r == 0 && c == 0)
            {
                continue;
            }


            auto& e = ci.img.ptr<uchar>(r)[c];

            if (e != old)
            {
                // cout << r << ":" << c << endl; 
                if (e > 0 )
                {
                    cout << sum << ",";
                }
                else
                {
                    cout << cnt << ",";

                }

                cnt_break++;


                
               
                cnt = 0;
                old = e;


          
            }

      

            if (cnt_break > 10) break;
        }

        if (cnt_break > 10) break;
			

   
	

	}

       

	

	

	//ci.s_i(); 
	//ci.img; 
	//cout << ci.img.size << endl;




#endif 
#if 0

    string view_jpg = "07_08.jpg";

    string fn = "d:/jd/t/t0/tilejpg/" + view_jpg;
    ci.read_img(fn);
    ci.img;


    string fn_list = "d:/jd/t/t0/tilejpg/fn_list.txt";


    std::vector<std::vector<cv::Point2i>> vvp = readPointsFromFile(fn_list, view_jpg);


    // ci.img.setTo(0); 

    // ci.img *= 0.25;
    // draw vvp on ci.img 
    for (auto& e : vvp)
    {
        for (auto& e1 : e)
        {
            cv::circle(ci.img, e1, 2, cv::Scalar(255, 0, 0), -1);
        }
    }

    // 










    //ci.resize(0.231);
    cv::imwrite("d:/jd/t/t0/1.jpg", ci.img); 


    //ci.s_i(); 

    ci.img;
    //cout << ci.img.size << endl; 


#endif 

#if 0
    // read fn 


    string tfn = "d:/jd/t/lena.bmp";

    ci.read_img(tfn); 

    ci.img; 



    int N = 10;

       vector<Mat> vimg; // 修复：将函数声明改为变量定义

       for (int e : ci.R(0, N)) // 使用 std::views::iota 生成范围
       {
           cout << e << endl;
           auto e_fn = s_("d:/jd/t/p/masked_{}.png", e);

           ci.read_img(e_fn); 

           vimg.push_back(ci.img.clone()); // 确保 vimg 是一个有效的 vector
       }



       // merge all vimg
	   cv::Mat img_sum = cv::Mat::zeros(ci.img.size(), ci.img.type());
	   for (int i = 0; i < vimg.size(); i++)
	   {
		   img_sum += vimg[i]/N;
	   }

	   // img_sum = img_sum * split_num; 

       img_sum = ~img_sum; 


       // use min_max to change img_sum to be 0~255
	   double minVal, maxVal;
	   cv::minMaxLoc(img_sum, &minVal, &maxVal);
	   // img_sum.convertTo(img_sum, CV_8UC3, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	   img_sum = img_sum * 255.0 / (maxVal - minVal);


	   ci.s_i(img_sum);

	   // ci.s_i(); 
	   // ci.img; 
	   // cout << ci.img.size << endl;

#endif 
#if 0
    string fn = "d:/jd/t/lena.bmp"; 

    ci.read_img(fn); 


    auto img_r1 = ci.img.clone(); 

	// rotate img_r1 180 degree 
	cv::Mat img_r2;
	cv::rotate(img_r1, img_r2, cv::ROTATE_180);


    ci.s_i(img_r2); 
    assert(0 == 1);

    
    vector<cv::Mat> vmat; 

    int split_num = 11;
	cv::Mat img_r0 = ci.img.clone();

    for (int id = 0; id < split_num; id++)
    {
        // for each pixel in ci.img , gray++;
        for (int r = 0; r < ci.img.rows; r++)
        {
            for (int c = 0; c < ci.img.cols; c++)
            {
                auto& e = ci.img.ptr<cv::Vec3b>(r)[c];

                if ((r * ci.img.step + c) % split_num == id)
                {
                    
                    
                }
                else
                {
                    e = { 0,0,0 };
                }

            }
        }

        vmat.push_back(ci.img.clone()); 


		cv::imwrite(to_string(id) + ".jpg", ci.img);

        ci.img = img_r0.clone();;


    }




	// add all imgs in vmat
	cv::Mat img_sum = cv::Mat::zeros(ci.img.size(), ci.img.type());
	for (int i = 0; i < vmat.size(); i++)
	{
		img_sum += vmat[i];
	}

    //img_sum = img_sum * split_num; 

    ci.s_i(img_sum); 





#endif 




#if 0

    // create a example of fArray_DiPloid, float type
	std::vector<float> fArray_DiPloid = { 1.1, 2.1,  };
	fArray_DiPloid.push_back(3.1);
	fArray_DiPloid.push_back(4.1);
	fArray_DiPloid.push_back(5.1);
	fArray_DiPloid.push_back(6.1);
	fArray_DiPloid.push_back(7.1);
	fArray_DiPloid.push_back(8.1);
	fArray_DiPloid.push_back(9.1);
	fArray_DiPloid.push_back(10.1);

	// calculate mean of fArray_DiPloid
	float sum = 0;
	for (auto& e : fArray_DiPloid)
	{
		sum += e;
	}
	float mean = sum / fArray_DiPloid.size();

	
	



#endif 
#if 0

    enum CellClassType
    {
        UNKNOW_CELL = 0,
        NORMAL_CELL = 1,
        PYKNOTIC_CELL = 2,
        WBC_CELL = 3,
        IMPURITY_CELL = 4,
        NAKED_CELL = -1
    };

	

    int t0 = -11;

    CellClassType e = (CellClassType)t0; 

	cout << e << endl;

    


#endif 
#if 0
	auto img = ci.create_img_rc_chn(101, 101, 1);
auto step = img.step[0]; // 修复为获取第一通道的步幅

cout << step << endl;

// get widthstep for img 
int widthstep = img.step; // 修复为获取第一通道的步幅

int ws = img.step[0];


uchar* pdata = img.data; 


for (int i = 0; i < 101; i++)
{
	for(int j=0;j<101;j++)
	{
		pdata[i * ws + j] = i + j; 
	}
}





#endif 
#if 0
    string fn = "d:/jd/t/git/dna-analysis/images/analysis_result/DTA0004/middle/10_7_rgb_org.jpg"; 
	ci.read_img(fn);

    // smooth ci.img 
	cv::GaussianBlur(ci.img, ci.img, cv::Size(5, 5), 0, 0);
    

    auto bgr = ci.img.clone(); 


	

	cv::Mat img_gray = toGrayByStripMin(bgr);

	//ci.s_i(img_gray);

	
	auto p_con_roi = findCellClusterContour(img_gray);

	auto img_contour = p_con_roi.first.clone();

	auto bgrcp = bgr.clone();
	// show contour on bgr
	cv::drawContours(bgrcp, p_con_roi.second, -1, cv::Scalar(0, 0, 255), 1);
    // save bgrcp
	cv::imwrite("d:/jd/t/t0/10_7_rgb_org_contour.jpg", bgrcp);

    int bg[4] = { 0 };
    doCurrentBackground(img_gray, bg);


    ci.img = img_gray.clone();
	
    cv::threshold(ci.img, ci.img, 161, 255, cv::THRESH_BINARY);

    

    ci.img = ~ci.img; 


	ci.find_contours();

    ci.v_contours;
	auto ci_v_contours_cp = ci.v_contours;

	// filter ci.v_contours by area
	ci.v_contours.clear();
	for (auto& e : ci_v_contours_cp)
	{
		auto area = cv::contourArea(e);
		if (area > 600)
		{

			// push convexhull of e to ci.v_contours
			std::vector<cv::Point> e_;
			cv::convexHull(e, e_);
			ci.v_contours.push_back(e_);
		}
	}

	// draw p_con_roi.second  and ci.v_contours on bgrcp , use different color
	



    bgrcp = bgr.clone();

    // draw p_con_roi.second  and ci.v_contours on bgrcp , use different color 
	cv::drawContours(bgrcp, p_con_roi.second, -1, cv::Scalar(0, 0, 255), 3);
	cv::drawContours(bgrcp, ci.v_contours, -1, cv::Scalar(0, 255, 0), 3);

	// save bgrcp
	cv::imwrite("d:/jd/t/t0/10_7_rgb_org_contour_ci.jpg", bgrcp);




	
 
    std::vector<std::vector<cv::Point>> vsmall;
    std::vector<std::vector<cv::Point>> vbig;


	//// for each pair ele in p_con_roi.second   ci.v_contours, assess the overlap ratio
    for (auto& esmall : p_con_roi.second)
    {
        for (auto& ebig : ci.v_contours)
        {
            auto ifSame = IsContourBeTheSame(esmall, ebig);

            if (ifSame == true)
			{
				vsmall.push_back(esmall);
				vbig.push_back(ebig);
            }
        }
    }

    bgrcp = bgr.clone();

    // draw p_con_roi.second  and ci.v_contours on bgrcp , use different color 
    cv::drawContours(bgrcp, vsmall, -1, cv::Scalar(0, 0, 255), 3);
    cv::drawContours(bgrcp, vbig, -1, cv::Scalar(0, 255, 0), 3);

    // save bgrcp
    cv::imwrite("d:/jd/t/t0/thesame_big_small.jpg", bgrcp);
	// ci.s_i();








       










#endif 
#if 0

    char id_buf[7] = { 0 };
    auto len_ = sprintf_s(id_buf, sizeof(id_buf), "%06d", atoi("12"));
    // sprintf(id_buf, "%06d", "1");

    cout << id_buf << endl; 


#endif 

#if 0
    vector<int> vi{ 1,2,34 }; 
    auto vicp = vi; 
    vi[0] += 999; 
    vi.push_back(-999); 



    cout_("{}, {}", vicp, vi); 

    vi.pop_back(); 

    cout_("{}, {}", vicp, vi);

#endif 
#if 0

    int x = 3000; 
    int y = 3000; 

    auto xy = (x << 10) + y;
    cout << xy << endl; 
    assert(0 == 1);
#endif 


#if 0
    int rows = 1232;
    int cols = 1760;

#if 1
    ci.read_bin_to_mat("d:/jd/t/t0/1.dat", rows, cols, 4);
    cv::imwrite("d:/jd/t/t0/1.png", ci.img); 

    auto img_bak = ci.img.clone(); 

    ci.s_i();

#endif 

  ci.read_bin_to_mat("d:/jd/t/t0/2.dat", rows, cols, 4);





    cv::imwrite("d:/jd/t/t0/2.png", ci.img);

    ci.s_i();


#endif

#if 0

    vector<cv::Mat> v_img; 


    assert(argc == 2); 

    string fn_all_jpg = string(argv[1]);



    auto vstr = ci.read_txt_to_vec_str(fn_all_jpg);


    

    for (auto fn : ci.filter_out_vstr(vstr, R"(^#.*)"))
    {

        // string fn = "d:/jd/t/t0/7_12_rgb_org_group.jpg";


        cout << fn << endl;




        ci.read_img(fn);

        std::filesystem::path path(fn);
        string fn_new = "d:/jd/t/t0/cervic_" + path.filename().string();

        cv::GaussianBlur(ci.img, ci.img, cv::Size(3, 3), 0, 0);

        auto bgr_r0 = ci.img.clone();


        auto gray = toGrayByStripMin(ci.img);

        auto p_con_roi = findCellClusterContour(gray);

        cout << p_con_roi.first.rows << ":" << p_con_roi.second.size() << endl;

        gray = p_con_roi.first; 
        v_img.push_back(gray.clone()); 



        // Return the filename without the extension (if needed)  
        // You can also use path.filename().string() if you want the extension  
       

       

        cv::imwrite(fn_new, gray); 


    }








#endif 
#if 0

     // 示例轮廓  
    std::vector<Point> contour = {
        Point(100, 100), Point(200, 100),
        Point(200, 200), Point(100, 222),  Point(100, 100),
    };

    vector< std::vector<Point>> v_contours {contour}; 


    test_const_contour(v_contours); 


    // 进行轮廓收缩  
    std::vector<Point> shrunkenContour = shrinkC(contour,0.3);

    // 在图像上绘制原始和收缩后的轮廓  
    Mat image = Mat::zeros(300, 300, CV_8UC3);


    
    cv::drawContours(image, v_contours, -1, Scalar(255, 0, 0), cv::FILLED); // 原始轮廓  

    // 绘制收缩后的轮廓  
    // polylines(image, shrunkenContour, true, Scalar(0, 255, 0), 2); // 收缩后轮廓  


    ci.s_i(image);

#endif 

#if 0
    string fn = "d:/jd/t/t0/7_12_rgb_org_group.jpg";
    ci.read_img(fn);

    cv::GaussianBlur(ci.img, ci.img, cv::Size(3, 3), 0, 0);

    auto bgr_r0 = ci.img.clone();


    auto gray = toGrayByStripMin(ci.img);
 
    auto gray_r0 = gray.clone(); 

    ci.img = gray;

    int step_r = 32; 
    
    int step_c = step_r;

    int overlap = step_r / 2; 
    int thres_bg = 191; 
    int thres = 80; 
    int thres_cnt_roi = 100; 
    int thres_area = 10000; 

    

    int size_e_row = 32; 

    struct bb_cnt
    {
        cv::Rect bb; 
        int cnt; 
        int row;
        int col;

    }; 

    vector< bb_cnt > v_bb_cnt; 


    int row_cut = 0;

    for (int r = 0; r < ci.img.rows - step_r; r += (step_r - overlap))
    {
        int col_cut = 0; 
        for (int c = 0; c < ci.img.cols-step_c; c += (step_c - overlap))
        {


            cv::Rect bb = cv::Rect(c, r, step_c, step_r); 

            cv::Mat img_roi = gray_r0(bb).clone(); 

            

     
            int cnt = 0;
            int cnt_bg = 1; 
            img_roi.forEach<uchar>(
            [&thres,&thres_bg, &cnt, &cnt_bg](uchar& pixel, const int* position)
            {
                //if (pixel < thres_bg) 
                //{
                //    cnt_bg++;
                //}

                if (pixel < thres)
                {
                    cnt++;
                }


                
            });

            // cnt = cnt;

            if (cnt > thres_cnt_roi)
            {

                bb_cnt e_bb_cnt = { bb, cnt, row_cut, col_cut };
                v_bb_cnt.push_back(e_bb_cnt);
                
            }


            cv::Point center = cv::Point(bb.x + bb.width / 2-10, bb.y + bb.height / 2);

            putText(ci.img, to_string(cnt),
                center, FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(0), 2);


            // draw a rectange on bb 
            // 在适当的位置添加绘制矩形的代码
            // cv::rectangle(ci.img, bb, cv::Scalar(0, 0, 0), 1); // 使用红色绘制矩形

            col_cut++;
        }

        row_cut++;
    }


    

    cv::imwrite("d:/jd/t/t0/1.png", ci.img); 

   


    std::sort(v_bb_cnt.begin(), v_bb_cnt.end(), [](const bb_cnt& a, const bb_cnt& b) {
        return (a.cnt>b.cnt);
    });


    //vector<cv::Mat> vimg; 


    //vector<cv::Mat> v_res; 

    //for (auto e_bb_cnt: v_bb_cnt)
    //{

    //    cv::Mat img_roi = ci.img(e_bb_cnt.bb);

    //    vimg.push_back(img_roi); 

    //    if (vimg.size() == size_e_row)
    //    {
    //        auto img_all = ci.hconcat(vimg, 4); 
    //        // ci.s_i(img_all); 

    //        v_res.push_back(img_all); 


    //        vimg.clear(); 
    //    }

    //}


    //auto img_res = ci.vconcat(v_res); 

    //cv::imwrite("d:/jd/t/t0/2.png", img_res); 

    ci.img = gray_r0.clone();

    vector< bb_cnt > v_bb_cnt_roi;

    v_bb_cnt_roi = v_bb_cnt; 

    //for (auto e_bb_cnt : v_bb_cnt)
    //{
    //    if (e_bb_cnt.cnt > thres_cnt_roi)
    //    {
    //        v_bb_cnt_roi.push_back(e_bb_cnt);
    //    }
    //}


    for (auto e_bb_cnt : v_bb_cnt_roi)
    {
        auto bb = e_bb_cnt.bb; 
        cv::Point center = cv::Point(bb.x + bb.width / 2 - 10, bb.y + bb.height / 2);

        // putText(ci.img, to_string(e_bb_cnt.cnt), center, FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(0), 2);

        // cv::rectangle(ci.img, bb, cv::Scalar(0, 0, 0), 1); // 使用红色绘制矩形
        // draw rectangle and filled with 255
        cv::rectangle(ci.img, bb, cv::Scalar(255), cv::FILLED);


    }


    // ci.threshold(254); 

    cv::threshold(img, img, 254, 255, cv::THRESH_BINARY);

    // ci.img; 

    // 对图ci.img 做一次开运算。
    //int kernelSize = 7;
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    //// 开运算
    //cv::morphologyEx(ci.img, ci.img, cv::MORPH_OPEN, element);
    //cv::morphologyEx(ci.img, ci.img, cv::MORPH_OPEN, element);
    //cv::morphologyEx(ci.img, ci.img, cv::MORPH_OPEN, element);

    cv::findContours(img, v_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    ci.find_contours(255);
    ci.v_contours;

    struct con_area
    {
        std::vector<cv::Point> e_c;
        int area;
    }; 


    vector<con_area> v_con_area;
    vector<std::vector<cv::Point>> v_con_roi;

    for (auto& e_c : ci.v_contours)
    {
        auto area = cv::contourArea(e_c);
        con_area e_con_area = {e_c, area};

        v_con_area.push_back(e_con_area); 

    }


    // sort by v_con_area.area
    std::sort(v_con_area.begin(), v_con_area.end(), [](const auto& a, const auto& b) {
        return a.area > b.area; // or any other comparison logic based on area
    });

    
    for (auto e_con_area : v_con_area)
    {
        if (e_con_area.area > thres_area)
        {
            std::vector<cv::Point> e_c;
            
            cv::convexHull(e_con_area.e_c, e_c);

            v_con_roi.push_back(e_c);
        }
    }




    ci.img = gray_r0.clone(); 

    // draw contour on ci.img 
    //img_contour_line = cv::Mat::zeros(img.size(), CV_8UC1);;
    cv::drawContours(ci.img, v_con_roi, -1, cv::Scalar(255), 3);
    

    cv::imwrite("d:/jd/t/t0/3.png", ci.img); 

    ci.resize(0.5);
    ci.s_i();
    









#endif 
#if 0
    string fn = "d:/jd/t/t0/7_12_rgb_org_group.jpg";
    ci.read_img(fn);

	// auto bgr_r0 = ci.img.clone();


    cv::GaussianBlur(ci.img, ci.img, cv::Size(5, 5), 0, 0);

    auto bgr_r0 = ci.img.clone();


    auto gray = toGrayByStripMin(ci.img);
    ci.img = gray;




    // get mean of ci.img
    auto mean_img = cv::mean(ci.img);



    auto gray_r0 = ci.img.clone();
    int thres = mean_img[0];
    ci.threshold(thres, 255);

    ci.img = ~ci.img;


    auto img_copy = ci.img.clone(); 

  

    ci.find_contours();

    ci.v_contours; 

    ci.img_contour_mask; 
    

    struct con_feature
    {
        std::vector<cv::Point>* p_ec;
        int area;
    };


	vector<con_feature> v_con_feature;
    for (auto& ec : ci.v_contours)
    {
		con_feature e_con_feature = { &ec, cv::contourArea(ec) };

		v_con_feature.push_back(e_con_feature);

    
    }


	std::sort(v_con_feature.begin(), v_con_feature.end(), [](const con_feature& a, const con_feature& b) {
		return a.area > b.area; // Assuming 'ep' is a member of the elements in vpxy
	});




    vector<int> v_area;
	for (auto& e : v_con_feature)
	{
        if (e.area>9)
        {
            v_area.push_back(e.area);
        }
	}

    vector<std::vector<cv::Point>> v_con_roi;

    for (int i = 0; i < v_con_feature.size()/2; i++)
    {
        v_con_roi.push_back(*(v_con_feature[i].p_ec));
    }


    vector<cv::Mat> vimg; 
    vector<cv::Mat> vimg_r0; 

    
    for (auto& ec : v_con_roi)
    {


		// clone_contour_bb(gray_r0, ec);
        auto img_mask_t = ci.img_contour_mask.clone().setTo(0);
		cv::drawContours(img_mask_t, std::vector<std::vector<cv::Point>>{ec}, -1, cv::Scalar(255), cv::FILLED);

		auto bb = cv::boundingRect(ec);


        cv::Mat eimg_roi = gray_r0(bb) & img_mask_t(bb);

        cv::GaussianBlur(eimg_roi, eimg_roi, cv::Size(5, 5), 0, 0);

       // cv::GaussianBlur(eimg_roi, eimg_roi, cv::Size(5, 5), 0, 0);

        cv::GaussianBlur(eimg_roi, eimg_roi, cv::Size(5, 5), 0, 0);

        



        auto eimg_roi_ = img_min_nonzero(eimg_roi); 
        cv::Mat dst;

        cv::Mat eimg_roi__;

        cv::rotate(eimg_roi, dst, cv::ROTATE_90_CLOCKWISE);
        dst = img_min_nonzero(dst);
        cv::rotate(dst, eimg_roi__, cv::ROTATE_90_COUNTERCLOCKWISE);

        eimg_roi_ = eimg_roi_ & eimg_roi__; 

        // rotate eimg_roi 90 degree 


        cv::Mat img_t = eimg_roi * 0.7 + eimg_roi_ * 0.3;

        vimg.push_back(img_t); 

        
        vimg_r0.push_back(gray_r0(bb));




        if (vimg.size() == 8)
        {
            ci.img = ci.hconcat(vimg, 2); 
            auto ci_img_edit = ci.img.clone(); 
            ci.img = ci.hconcat(vimg_r0, 2); 

            ci.img = ci.vconcat({ ci_img_edit, ci.img }, 0);

            ci.s_i();

            

            vimg.clear(); 
            vimg_r0.clear(); 
        }





            

    }



	//draw v_con_roi
    

    cv::Mat img_mask = cv::Mat::zeros(ci.img.size(), CV_8UC1);
    cv::drawContours(img_mask, v_con_roi, -1, cv::Scalar(255), cv::FILLED);

	cv::imwrite("d:/jd/t/t0/img_mask.png", img_mask);
    



 //   cv::imwrite("d:/jd/t/t0/bin.png", ci.img);


 //   auto binimg = ci.img.clone(); 



 //   

 //   
 //   vector<cv::Mat> vchn = { binimg, binimg, binimg};
 //   cv::merge(vchn, binimg);


	//cv::Mat img_r0_bin = bgr_r0*0.5 +  binimg *0.5;

	//cv::imwrite("d:/jd/t/t0/img_r0_bin.png", img_r0_bin);







    

#endif 

#if 0
    float f = 0.35;
    string fn = "d:/jd/t/git/dna-analysis/images/analysis_result/DAcervicHard/middle/1.txt"; 

    auto vfn = ci.read_txt_to_vec_str(fn);
    for (auto efn : vfn)
    {
		auto efn_new = ci.replace_str(efn, "middle", "middle_new");   
     
        ci.read_img(efn);
        auto idx = efn.find_last_of("\\");
		auto puttext_ =  efn.substr(idx+1);

		if (puttext_.find("10_9") == string::npos)
		{
            continue;
		}

        ci.resize(f);
        ci.puttext(puttext_+ "_old");
       
		auto img = ci.img.clone();
		ci.read_img(efn_new);
        ci.resize(f);
        ci.puttext(puttext_ + "new");
		auto img_new = ci.img.clone();

	
		ci.img = ci.hconcat({ img, img_new }, 4);

    
        cv::imwrite("d:/jd/t/t0/1.png",ci.img);
        ci.s_i();
       



        
    }


#endif 
#if 0
    std::vector<Point> c0 = {
          Point(100, 100), Point(300, 100), Point(350, 200), Point(50, 200)
    };

    // 定义直线 y0 的值  
    float y0 = 150.0; // 传入的 y 坐标值  

    // 计算交点  
    std::vector<Point2f> intersectionPoints = getIntersectionPoints(c0, y0);

    // 输出交点  
    std::cout << "Intersection Points with line y = " << y0 << ":\n";
    for (const auto& point : intersectionPoints) {
        std::cout << "Point: (" << point.x << ", " << point.y << ")\n";
    }

#endif 


#if 0

	string fn_bgr = "d:/jd/t/t0/cervicHard.jpg";
    ci.read_img(fn_bgr);

    auto bgr_r0 = ci.img.clone(); 

	string fn = "d:/jd/t/t0/img_r0_gray.png";

	ci.read_img(fn);

    auto img_r0 = ci.img.clone(); 

	string fn_mask = "d:/jd/t/t0/img_contour_mask_ok.png";
	ci.read_img(fn_mask);
    ci.threshold(254);
    ci.find_contours(); 

    ci.v_contours;

	auto imgmean = ci.create_img_rc_chn(1,ci.v_contours.size(), 1);
    vector<int> vmean;
    int cnt = 0; 




	vector<para_con> v_para_con;

    

    for (auto & e_c : ci.v_contours)
    {
        cv::Moments m = cv::moments(e_c);
        cv::Point center(int(m.m10 / m.m00), int(m.m01 / m.m00));

        int cenp_i = 0;
        for (int k = -2; k < 2; k++)
        {
            cenp_i += img_r0.ptr<uchar>(center.y)[center.x + k];

        }
        cenp_i /= 5;
		uchar cenp = cenp_i;

		para_con e_para_con = { cenp_i, 0, &e_c };
		v_para_con.push_back(e_para_con);
        
		imgmean.ptr<uchar>(0)[cnt] = cenp;
		vmean.push_back(cenp);
        cnt++;
    }


   


    auto meanv = cv::mean(imgmean);

    cout << meanv << endl; 

    vector<int> hist(256, 0);

    for (auto& e : vmean)
    {
        hist[e]++;
    }

    auto pimg = ci.P(hist, 1); 


    pimg; 

    // find max element location  in hist 
    int max_loc = 0;
    int maxv = 0;
    for (int i = 0; i < hist.size(); i++)
    {
		if (hist[i] > maxv)
		{
			maxv = hist[i];
			max_loc = i;
		}
	}

	
    cout << maxv << ":" << max_loc << endl; 

    
    int filter_cenv_thres = max_loc + 65;  // about 107 ?


    std::vector<std::vector<cv::Point>> v_contour_t;
	for (auto& e : v_para_con)
	{
		if (e.cen_avg > filter_cenv_thres)
		{
			e.flag_filterout = 1;
            continue; 
		}

		v_contour_t.push_back(*e.p_e_c);
	}

    
	cv::Mat img_contour_line = cv::Mat::zeros(ci.img.size(), CV_8UC1);

	

    
    cv::drawContours(img_contour_line, v_contour_t, -1, cv::Scalar(255), 1);
    

	cv::Mat img_contour_line_bgr;

	vector<cv::Mat> vchn = { img_contour_line, img_contour_line, img_contour_line };
    cv::merge(vchn, img_contour_line_bgr);


    img_contour_line_bgr = img_contour_line_bgr * 0.7 + bgr_r0 * 0.3;
	cv::imwrite("d:/jd/t/t0/img_contour_line_filter_c.png", img_contour_line_bgr);



#endif 

#if 0
    string fn = "d:/jd/t/t0/cervicHard.jpg";
    ci.read_img(fn);


    cv::GaussianBlur(ci.img, ci.img, cv::Size(3, 3), 0, 0);

    auto bgr_r0 = ci.img.clone();


    auto gray = toGrayByStripMin(ci.img);
    ci.img = gray;




    // get mean of ci.img
    auto mean_img = cv::mean(ci.img);



    auto img_r0 = ci.img.clone();
    int thres = mean_img[0];
    ci.threshold(thres, 255);

    ci.img = ~ci.img;


    cv::imwrite("d:/jd/t/t0/bin.png", ci.img);




    ci.img = ci.find_contours();

    ci.v_contours;
    ci.img_contour_mask;


    std::vector<cv::Mat> channels;
    cv::split(bgr_r0, channels);

    for (auto& echn : channels)
    {
        echn = echn & ci.img_contour_mask;
    }
    // 计算每个通道的梯度幅值
    std::vector<cv::Mat> gradientMagnitudes;
    for (const auto& channel : channels) {
        cv::Mat Gx, Gy;
        cv::Sobel(channel, Gx, CV_64F, 1, 0, 9, 5.0); // 水平方向梯度
        cv::Sobel(channel, Gy, CV_64F, 0, 1, 9, 5.0); // 垂直方向梯度
        cv::Mat gradientMagnitude;
        cv::magnitude(Gx, Gy, gradientMagnitude);
        gradientMagnitudes.push_back(gradientMagnitude);
    }



    cv::normalize(gradientMagnitudes[0], gradientMagnitudes[0], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(gradientMagnitudes[1], gradientMagnitudes[1], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(gradientMagnitudes[2], gradientMagnitudes[2], 0, 255, cv::NORM_MINMAX, CV_8U);



    // 取梯度, 取其中最大值 * 3
    cv::Mat gradientMagnitude;
    (cv::max)(gradientMagnitudes[0], gradientMagnitudes[1], gradientMagnitude);
    (cv::max)(gradientMagnitude, gradientMagnitudes[2], gradientMagnitude);
    cv::multiply(gradientMagnitude, 3, gradientMagnitude);


    cv::imwrite("d:/jd/t/t0/gradientMagnitude.png", gradientMagnitude);



    std::vector<std::vector<cv::Point>> contours;

    cv::Mat binaryImage;

    for (int i = 70; i <= 150; i += 20)
    {
        // 二值化
        cv::threshold(gradientMagnitude, binaryImage, i, 255, cv::THRESH_BINARY);


        cv::imwrite("d:/jd/t/t0/binaryImage" + to_string(i) + ".png", binaryImage);

        // context->saveMiddleImg(binaryImage, "binaryImage" + TC_Common::tostr(i) + ".jpg");
        // 轮廓检测
        std::vector<std::vector<cv::Point>> contoursThreshold;

        cv::Mat img_contour_mask = cv::Mat::zeros(binaryImage.size(), CV_8UC1);


        cv::findContours(binaryImage, contoursThreshold, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);




        std::vector<std::vector<cv::Point>> convexFilter;
        for (const auto& e_contour : contoursThreshold) 
        {

            
			if (e_contour.size() < 6)
			{
				continue;
			}

			auto area = cv::contourArea(e_contour);

			auto bb = cv::boundingRect(e_contour);

            if (area > 400 || area< 17 || bb.width < 9 || bb.height < 9 || bb.width > 35 || bb.height > 35)
            {
                continue;
            }

            double perimeter = cv::arcLength(e_contour, true);
            double circularity = 12.566f * area / (perimeter * perimeter);
            if (circularity < 0.55)
            {
                continue;
            }

            cv::Mat contourHull;
            cv::convexHull(e_contour, contourHull);
            convexFilter.push_back(contourHull);
        }


        contours.insert(contours.end(), convexFilter.begin(), convexFilter.end());
    }






	cv::Mat img_contour_mask = cv::Mat::zeros(binaryImage.size(), CV_8UC1);


	// drawContours
	// cv::drawContours(img_contour_mask, contours, -1, cv::Scalar(255), cv::FILLED);


    int cnt = 0;
    vector<uchar> color_lines = {
	180, 200, 220 ,240, 255
    };
	std::vector<std::vector<cv::Point>> contours_roi;
    for (auto& e_c : contours)
    {

		//if (cnt > 100)
		//{
		//	break;
		//}
	
		contours_roi.push_back(e_c);
        cv::drawContours(img_contour_mask, contours_roi, -1, cv::Scalar(color_lines[cnt % color_lines.size()]), 1);
        contours_roi.clear();


        
        cnt++;
    }
    
    img_contour_mask = img_contour_mask * 0.6 + img_r0 * 0.4;

	cv::imwrite("d:/jd/t/t0/img_contour_mask.png", img_contour_mask);


	

	
#if 1

    // 过滤重复的轮廓, 使用map优化查找效率
    std::vector<std::vector<cv::Point>> uniqueContours;
    for (const auto& contour : contours) {
        // 计算当前轮廓的中心点
        cv::Moments m = cv::moments(contour);
        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);

        bool isDuplicate = false;
        // 与已保存的轮廓比较中心点距离
        for (size_t i = 0; i < uniqueContours.size(); i++) {
            cv::Moments m2 = cv::moments(uniqueContours[i]);
            cv::Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);

            // 如果中心点距离小于10个像素,认为是重复的轮廓, 并保留面积大的轮廓
            if (cv::norm(center - center2) < 15) {
                isDuplicate = true;
                if (cv::contourArea(contour) > cv::contourArea(uniqueContours[i]))
                {
                    uniqueContours[i] = contour;
                }
                break;
            }
        }

        if (!isDuplicate) {
            uniqueContours.push_back(contour);
        }
    }



    contours = uniqueContours;


    img_contour_mask.setTo(0); 
    cnt = 0;
    for (auto& e_c : contours)
    {


        contours_roi.push_back(e_c);
        cv::drawContours(img_contour_mask, contours_roi, -1, cv::Scalar(color_lines[cnt % color_lines.size()]), 1);
        contours_roi.clear();

        cnt++;
    }

    img_contour_mask = img_contour_mask * 0.6 + img_r0 * 0.4;

    cv::imwrite("d:/jd/t/t0/img_rm_dup.png", img_contour_mask);



	img_contour_mask.setTo(0);

    cv::drawContours(img_contour_mask, contours, -1, cv::Scalar(255), cv::FILLED);


	cv::imwrite("d:/jd/t/t0/img_contour_mask_ok.png", img_contour_mask);


	cv::imwrite("d:/jd/t/t0/img_r0_gray.png", img_r0); 



    




#endif 




#endif 
#if 0

	string fn = "d:/jd/t/t0/cervicHard.jpg";
    vector<cv::Mat> v_img; 

	ci.read_img(fn);
	ci.cvtcolor("GRAY");

    auto ciimg_clone = ci.img.clone(); 

    
	v_img.push_back(ciimg_clone);

    ci.img.setTo(0);


	v_img.push_back(ci.img);
	
	ci.img = ci.hconcat(v_img, 12);


    ci.resize(0.2);

    ci.s_i();
	// cout << dstimg;







#endif 


#if 0
    string fn = "d:/jd/t/t0/findcontour.png";
    ci.read_img(fn); 

    ci.cvtcolor("GRAY"); 

    auto gray = ci.img.clone(); 


	cv::GaussianBlur(ci.img, ci.img, cv::Size(3, 3), 0, 0);
    ci.find_contours();

    ci.v_contours; 

    ci.img = gray; 

    auto dstimg = ci.img.clone(); 

    int cnt = 0;

    vector<cv::Mat> vimg; 
    for (auto e : ci.v_contours)
    {
        
        std::vector<std::vector<cv::Point>> v_contours_ = { e };

        cv::drawContours(dstimg, v_contours_, -1, cv::Scalar(122), 4);
        vimg.push_back(dstimg.clone());
        cnt++;
        

    }

    // cout << dstimg; 


	ci.img = ci.vconcat({ gray, dstimg }, 12);
	ci.s_i(ci.img);


	ci.img = ci.vconcat(vimg, 12);


	cv::imwrite("d:/jd/t/t0/findcontour_result.png", ci.img);
    ci.resize(.4);
    ci.s_i(); 



#endif 


#if 0

    string fn = "d:/jd/t/t0/hardcell.png";
	ci.read_img(fn);


    cv::GaussianBlur(ci.img, ci.img, cv::Size(3, 3), 0, 0);


    // get mean of ci.img
	auto mean_img = cv::mean(ci.img);


	auto img_r0 = ci.img.clone();
    int thres = mean_img[0];
	ci.threshold(thres, 255);

    ci.img = ~ci.img;


    cv::imwrite("d:/jd/t/t0/bin.png", ci.img); 




    ci.find_contours();
    ci.v_contours;
    ci.img_contour_mask;


    int cnt = 0;
    int KER_SZ = 5;

    auto img_r1 = img_r0.clone();

    for (auto& econ : ci.v_contours)
    {
        auto bb = cv::boundingRect(econ); 

        cv::Mat imgroi = img_r0(bb) & ci.img_contour_mask(bb);



        



        int numOfNonZero = 0;



        cout << cnt << endl; 
        
        {
            // {0:l, 1:h, 2:avg_l, 3:avg_h, 4:stddev, 5: avg_all}
            vector<pixelValPos> vFeatureImgroi = getChestCellPixelStatInfo(imgroi, KER_SZ, &numOfNonZero);


            if (vFeatureImgroi.size() != CONST_LHAVGLHSTD_EXPECTED)
            {
                continue;
            }
            cv::Point2i coreMinPos = vFeatureImgroi[0].exy;


			uchar coreMinAvgVal = vFeatureImgroi[5].ep;

            auto thres = coreMinAvgVal;
			if (thres > 255)
			{
				thres = 255;
			}

            for (int y = 0; y < imgroi.rows; ++y) {

                for (int x = 0; x < imgroi.cols; ++x) {

					auto& e = imgroi.ptr<uchar>(y)[x];
					if (e < thres && e != 0) {
						e = 255;
						// coreMinPos = cv::Point2i(x, y);
					}

                }
            }
			

           // imgroi.copyTo(img_r1(bb));




            uchar updiff = (vFeatureImgroi[3].ep - vFeatureImgroi[2].ep) / 3.0f * 1.1f; // floodfill diff 上界,调整1.1f这个参数,调小会导致找得cell core偏小，反之

            // cv::floodFill(imgroi, coreMinPos, 255, 0, cv::Scalar::all(10), cv::Scalar::all(updiff), FLOODFILL_FIXED_RANGE);


            // 将imgroi中像素值为255的部分拷贝到img_r1的bb区域  
            for (int y = 0; y < imgroi.rows; ++y) {
                for (int x = 0; x < imgroi.cols; ++x) {
                    if (imgroi.ptr<uchar>(y)[x] == 255 && img_r1.ptr<uchar>(bb.y + y)[bb.x + x]!=255) {
                        img_r1.ptr<uchar>(bb.y + y)[bb.x + x] = imgroi.ptr<uchar>(y)[x];
                    }
                }
            }



            // ci.s_i(imgroi);
        }
      
        cnt++;
    }


    ci.img = img_r1;

    ci.resize(0.5);
    ci.s_i();


    


   
	



    

    
    

#endif 

#if 0
    string fn = "d:/jd/t/git/dna-analysis/images/analysis_result/DAchestdaHard/middle/10_9_rgb_org.jpg"; 
	ci.read_img(fn);
    // ci.img , {b,g,r} get min, then strip min from b,g,r, and average the other two
    auto& img = ci.img;
        assert(img.channels() == 3); // 确保图像是三通道的
        vector<cv::Mat> vbgr;;
		cv::split(img, vbgr); // 将图像拆分为三个通道

        auto gray = vbgr[0].clone();

        int sum = 0;
        int cnt = 0;

        for (int r = 0; r < img.rows; r++) 
        {
            for (int c = 0; c < img.cols; c++) 
            {
                cv::Vec3b& pixel = img.at<cv::Vec3b>(r, c);
                auto& e_gray = gray.ptr<uchar>(r)[c];

                vector<uint> vp = { pixel[0], pixel[1], pixel[2] };
                std::sort(vp.begin(), vp.end());
                e_gray = (vp[1] * 0.5 + vp[2] * 0.5);

                sum += e_gray;
                cnt++;

            }
        }
   

		auto avg = sum / cnt;

        ci.img = gray; 

        ci.s_i();

        cv::imwrite("d:/jd/t/t0/hardcell.png", ci.img);







#endif 
#if 0

    vector< string > v_fn = { "d:/jd/t/t0/background.bmp","d:/jd/t/t0/backgroundf.bmp", }; 

    vector<cv::Mat> v_img;

    for (auto efn : v_fn)
    {
        ci.read_img(efn);

 

        ci.resize(0.3);


        if (ci.img.channels() != 1)
        {
            ci.cvtcolor("GRAY");
            ci.puttext("BGR");

        }
        else
        {
            ci.puttext("Y");
        }
        v_img.push_back(ci.img);


    }


    
    ci.img = ci.hconcat(v_img, 12);


    // split ci.img to v_img
	/*vector<cv::Mat> v_img;
	cv::split(ci.img, v_img);

    ci.img = ci.hconcat(v_img, 12);*/

    //ci.resize(0.3);
    ci.s_i();

    





#endif 


#if 0
    vector<string> v_fn_img = {

            "D:\\jd\\t\\git\\analysis-ui\\out\\AnalysisData\\analysis\\DAchest2024121304\\middle\\7_7_rgb_org.jpg",
            "D:\\jd\\t\\git\\analysis-ui\\out\\AnalysisData\\analysis\\DAchest2024082810\\middle\\7_7_rgb_org.jpg",
            "D:\\jd\\t\\git\\analysis-ui\\out\\AnalysisData\\analysis\\DAchest2024121314\\middle\\7_7_rgb_org.jpg"
        };


	vector<cv::Mat> v_img;
    for (auto efn : v_fn_img)
    {
        ci.read_img(efn); 
		v_img.push_back(ci.img.clone());

    }

    auto v_img_ = vector<cv::Mat>();

    for (auto& eimg : v_img)
    {
        // split eimg to bgr
		//vector<cv::Mat> vbgr;
		// cv::split(eimg, vbgr);

        // loop eimg rows, cols, get r,g,b 
        for (int r = 0; r < eimg.rows; r++) {
            for (int c = 0; c < eimg.cols; c++) {
                cv::Vec3b & pixel = eimg.ptr<cv::Vec3b>(r)[c];
                uchar r_val = pixel[2]; // Red channel
                uchar g_val = pixel[1]; // Green channel
                uchar b_val = pixel[0]; // Blue channel


				vector<uchar> vp = { r_val, g_val, b_val };

				std::sort(vp.begin(), vp.end());

				auto max_p = vp[2];
				auto mid_p = vp[1];
				auto min_p = vp[0];
				auto pavg = (max_p + mid_p) / 2;


				pixel = cv::Vec3b(pavg, pavg, pavg);
                



                // auto m = max(r_val, std::max(g_val, b_val));

                // strip max from b,g,r, and averge the other two


            }
        }





        // hconcat vbgr
		//auto img = ci.hconcat(vbgr, 4);
		auto img = eimg.clone();
		v_img_.push_back(img);

		
    }



	auto img = ci.vconcat(v_img_, 14);
    //ci.img = img; 
    //ci.resize(0.18); 

    //ci.s_i();

	 cv::imwrite("d:/jd/t/t0/t.jpg", img);




#endif 
#if 0
// get cyto


    ci.create_img_rc_chn(2222, 2222, 1); 


    ci.img.setTo(0);
vector<cv::Point> points_r0 = {  
   cv::Point(1785, 903),  
   cv::Point(1785, 903),  
   cv::Point(1785, 904),  
   cv::Point(1785, 904),  
   cv::Point(1785, 905),  
   cv::Point(1785, 905),  
   cv::Point(1785, 906),  
   cv::Point(1785, 906),  
   cv::Point(1785, 907),  
   cv::Point(1785, 907),  
   cv::Point(1785, 908),  
   cv::Point(1785, 908),  
   cv::Point(1785, 909),  
   cv::Point(1785, 909)  
};



    vector<cv::Point> points = {
        {1442, 621},
        {1444, 623},
        {1444, 625},
        {1447, 628},
        {1447, 631},
        {1444, 634},
        {1444, 635},
        {1442, 637}
    };


    for (auto e : points_r0)
    {
        ci.img.at<uchar>(e) = 255;
    }


    ci.img;


    std::vector<std::vector<cv::Point>> contours(1);
    contours[0] = points;

    auto img_r1 = ci.img.clone().setTo(0);
    cv::drawContours(img_r1, contours, -1, cv::Scalar(255), cv::FILLED);
    


    // cv::findContours(img_r1, contours); 

    // cv::findContours(img_r1, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(img_r1, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);




    ci.s_i(img_r1);




#endif 

#if 0
    // get cyto
    string fn = "d:/jd/t/git/dna-analysis/images/analysis_result/DAchest2024082810/middle/11_3_imgroi.png";
    ci.read_img(fn); 

    int numofnonzero = 0;
    ci.img;
    auto LhAvglhStd = getChestCellPixelStatInfo(ci.img, 5, &numofnonzero);




#endif 

#if 0

    string fn = "d:/jd/t/t0/gray_r0.png";
    string fn_mask = "d:/jd/t/t0/mask_r1.png";
    
    ci.read_img(fn_mask);
    auto mask_r1 = ci.img.clone();
    ci.read_img(fn);
    auto gray = ci.img.clone();
    auto gray_r0 = gray.clone();

    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
        

    auto bb = cv::Rect(703, 1829, 82 , 64); 


    cv::Mat imgroi = gray(bb) & mask_r1(bb);

    int numofnonzero = 0;
    auto LhAvglhStd = getChestCellPixelStatInfo(imgroi, 5, &numofnonzero);

    auto coreMinPos = LhAvglhStd[0].exy;

    auto updiff = LhAvglhStd[3].ep - LhAvglhStd[2].ep;
    updiff = updiff / 3 * 1.1f;

    cv::floodFill(imgroi, coreMinPos, 255, 0, cv::Scalar::all(10), cv::Scalar::all(updiff), FLOODFILL_FIXED_RANGE);

    imgroi.copyTo(gray(bb));


    auto bb_ = cv::Rect(600, 1800, 222, 100);
    ci.img = ci.hconcat({ gray_r0(bb_), gray(bb_) },4); 

    ci.img;


    cout << gray.size() << endl;



#endif 
#if 0

    string fn = "d:/jd/t/t0/chest_1304_grey.jpg";
    ci.read_img(fn);

    auto gray_r0 = ci.img.clone();

    cv::GaussianBlur(ci.img, ci.img, cv::Size(5, 5), 0, 0);

    cv::threshold(ci.img, ci.img, 177, 255, ci.img.type());

    ci.img = ~ci.img;



    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(ci.img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    auto mask = ci.img.clone().setTo(0);

    cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);


    for (auto& ec : contours)
    {

        auto bb = cv::boundingRect(ec);

        cout << bb << endl; 
        cv::Mat imgroi = ci.img(bb) & mask(bb);

        // imgroi.setTo(255);

        for (int r = 0; r < imgroi.rows / 2; r++)
        {
            for (int c = 0; c < imgroi.cols / 2; c++)
            {
                imgroi.ptr<uchar>(r)[c] = 0;
            }
        }

        imgroi.copyTo(gray_r0(bb));

    }





    cv::Rect rect(22, 22, 400, 400);



    //cv::threshold(mask, mask, 1, 255, mask.type()); 



    ci.img = ci.hconcat({ gray_r0(rect),  ci.img(rect), mask(rect) }, 8);





    ci.s_i();

#endif 

#if 0
    string fn = "d:/jd/t/t0/chest_2810_grey.jpg";

    ci.read_img(fn);


    
    




    cv::Rect bb(0, 44, 400, 500); 

    auto bbimg = ci.img(bb); 

   


    // loop bbimg rows, cols 
    for (int r = 0; r < bbimg_copy.rows; r++) {
        for (int c = 0; c < bbimg_copy.cols; c++) {
            // 处理每个像素
            uchar pixel_value = bbimg_copy.ptr<uchar>(r)[c];
            // 进行某种操作，例如阈值处理
            if (pixel_value > 66) {
                bbimg_copy.ptr<uchar>(r)[c] = 255; // 设置为白色
            } else {
                bbimg_copy.ptr<uchar>(r)[c] = 0;   // 设置为黑色
            }
        }
    }


    // ci.img(bb) = bbimg.clone();

    

    bbimg_copy.copyTo(ci.img(bb)); 


    cout << ci.img << endl; 


#endif 

#if 0
     
    string fn_d = string("d:/jd/t/t0") + "/";

    int LEN = 36;
    vector<cv::Mat> vmat(LEN);


    for (auto idx : ci.R(LEN))
    {
        auto fn = s_("{}{}.png", fn_d, idx);

        //cout << fn << endl; 

        ci.read_img(fn);

        // ci.resize(11);
       // ci.s_i();


        vmat[idx] = ci.img;

    }


      vector<int> vidx_roi= ci.R(LEN); 
    //vector<int> vidx_roi = {7,17,14};


    vector<cv::Mat> v_3g(0);
    vector<string> v_3gstr(0);

    for (auto idx : vidx_roi)
    {
        auto gray = vmat[idx];


        int numofnonzero = 0; 
        auto LhAvglhStd = getChestCellPixelStatInfo(gray,5, &numofnonzero);


        // auto coreInnerxy = findMinNonZeroPixel(gray);
        auto coreMinPos = LhAvglhStd[0].exy;


        
        
        




        auto mask = gray.clone(); 

        cv::threshold(mask, mask, 1, 255, mask.type()); 



        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
        gray = gray & mask; 
        // cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);


   
       




        //ci_0.img = gray.clone(); 
        ////ci_0.resize(11);
        //ci_0.img;
        //ci_0.s_i();




        //cv::Mat Gx, Gy;
        //cv::Sobel(gray, Gx, CV_64F, 1, 0, 3); // 水平方向梯度
        //cv::Sobel(gray, Gy, CV_64F, 0, 1, 3); // 垂直方向梯度
        //cv::Mat gradientMagnitude;
        //cv::magnitude(Gx, Gy, gradientMagnitude);

        //cv::normalize(gradientMagnitude, gradientMagnitude, 0, 255, cv::NORM_MINMAX, CV_8U);

   

        //gradientMagnitude = gradientMagnitude & mask; 

        //auto gradientMagnitude_r0 = gradientMagnitude.clone();



        // auto coreInnerxy = findMinNonZeroPixel(gradientMagnitude);

        //auto minval = gradientMagnitude.ptr(coreInnerxy.y)[coreInnerxy.x];


        //cout << coreInnerxy << endl;



        uchar minval = gray.at<uchar>(LhAvglhStd[0].exy);

        
       


        auto gray_r1 = gray.clone();


#if 0
        int flags = 4:

        填充操作的标志，控制填充方式。常用的标志包括：
            4 : 边界填充，4 - 邻接。
            8 : 边界填充，8 - 邻接。
            FLOODFILL_FIXED_RANGE : 颜色匹配基于颜色范围，固定种子点颜色。
            FLOODFILL_MASK_ONLY : 只更新掩模，不填充图像。
#endif 


        auto updiff = LhAvglhStd[3].ep - LhAvglhStd[2].ep;

        
        updiff = updiff / 3 * 1.1f;
        // updiff = updiff / 10 * 1.0f;


        cout << "Index: " << idx 
            << ", Non-zero count: " << numofnonzero 
            << ", Core Inner Coordinates: " << coreMinPos 
            << ", Minimum Value: " << int(minval) 
            << ", Up Difference: " << int(updiff) 
            << ", Average Std Dev: " << int(LhAvglhStd[4].ep) 
            << endl;
        
        // cv::floodFill(gray_r1, coreInnerxy, 255, 0, 0, cv::Scalar(minval + 3), 8);
        cv::floodFill(gray_r1, coreMinPos, 255, 0, cv::Scalar::all(10), cv::Scalar::all(updiff), FLOODFILL_FIXED_RANGE);
        // cv::floodFill(gray_r1, coreInnerxy, 255, 0, cv::Scalar::all(10), cv::Scalar::all(updiff), 4);


        auto s_text = s_("coreInnerxy:[{}{}],minval:{},updiff:{}", coreMinPos.x, coreMinPos.y , int(minval), int(updiff));


       ci.img = ci.hconcat({ gray, gray_r1 }, 4);


       // ci.puttext(to_string(idx), cv::Scalar::all(222));

       v_3gstr.push_back(s_text);

        //ci.img = gradientMagnitude;


        //ci.resize(11); 
        ci.img;
        
        






        v_3g.push_back(ci.img); 

        //ci.s_i();
        v_3g.size();

       
    }




    
  

    ci.img = ci.vconcat(v_3g,8);

    auto img_r0 = ci.img.clone();

    cv::threshold(ci.img, ci.img, 254, 255, CV_THRESH_BINARY);


    std::vector<std::vector<cv::Point>> v_contours;

    cv::findContours(ci.img, v_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    auto mask = ci.img.clone().setTo(0);
    //cv::drawContours(mask, v_contours, -1, cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, cv::Point(-rect.x, -rect.y));
    cv::drawContours(mask, v_contours, -1, cv::Scalar(255,0,0), 1);


    

    ci.img = ci.hconcat({ img_r0, mask });


    ci.img_copy = ci.img; 




#endif 


#if 0



    string fn = "d:/jd/t/lena.bmp";

    ci.read_img(fn);
    ci.cvtcolor("GRAY");


    // ci.s_i(); 


    cv::Rect2i roi(22, 22, 120, 150);
    // 将 imgroi2_mat 设置回 ci.img 的相同 ROI

    auto imgroi = ci.img(roi);
    auto imgroi_copy = imgroi.clone();

    imgroi_copy.at<uchar>(0, 11) = 0;
    imgroi_copy.at<uchar>(1, 11) = 0;
    imgroi_copy.at<uchar>(2, 11) = 0;
    imgroi_copy.at<uchar>(3, 11) = 0;


    // 使用 cv::Mat 代替 cv::MatExpr
    cv::Mat imgroi2_mat = imgroi & imgroi_copy;

    // 使用 .at<uchar>() 方法访问像素值
    cout << int(imgroi2_mat.at<uchar>(1, 10)) << endl;

    // imgroi2_mat.setTo(222);


    imgroi.at<uchar>(0, 1) = 0;








    //imgroi = imgroi2_mat; 



    // set imgroi2_mat back to ci.img 



    // imgroi.setTo(222); 


    for (int r = 0; r < imgroi.rows; r++)
    {
        // loop cols 
        // loop cols 
        for (int c = 0; c < imgroi.cols; c++)
        {
            // Perform operations on each column
            // Example: Access pixel value
            auto& pixelValue = imgroi.ptr<v3b>(r)[c];

            if (r < 10 && c < 11)
            {
                pixelValue = v3b(0, 0, 0);
            }
            else
            {
                pixelValue = v3b(0, 255, 255);
            }
            // Add your processing logic here
        }
    }

    ci.img;

    Sleep(1);
#endif 
#if 0



//auto fn = string("d:/jd/t/t0/chest_5_7.jpg");
//auto fn = string("d:/jd/t/t0/cervic_6_7.jpg");
//auto fn = string("d:/jd/t/t0/chest_1304.jpg");
    auto fn = string("d:/jd/t/t0/chest_1304_grey.jpg");

    //auto fn = string("d:/jd/t/t0/chest_1304.jpg");
    //auto fn = string("d:/jd/t/t0/chest_1314_grey.jpg");

     //auto fn = string("d:/jd/t/t0/chest_2810.jpg");
     //auto fn = string("d:/jd/t/t0/chest_2810_grey.jpg");
    ci.read_img(fn);


 



#if 0
    vector<cv::Mat> v_bgry;
    auto bgr = ci.img.clone();

    ci.cvtcolor("gray");
    auto gray = ci.img.clone();


    cv::split(bgr, v_bgry);
    v_bgry.push_back(gray);

    for (auto& e : v_bgry)
    {
        // cut e to be center rectangle  w x h: 400x500
        auto w = 400;
        auto h = 500;
        cv::Rect centerRect((e.cols - w) / 2, (e.rows - h) / 2, w, h);
        e = e(centerRect).clone();

    }

    ci.img = ci.hconcat(v_bgry, 4);

    // ci.resize(0.3);
    ci.s_i();

#endif 




    using cvContour = std::vector<std::vector<cv::Point>>;
    cv::Mat image;

    int flag_need_cut = 1;
    int cutrows = 400;
    int cutcols = 400;
    
    const int BIN_THRES = 41;
    const int DIFF_THRES = 100;
    const int D_DIFF_THRES = 88;

    int intv = 4;



    cv::Mat bgr;
    auto img_r0 = ci.img.clone();
    cv::Mat & gray = ci.img; 
    int bg[4];
    doCurrentBackground(gray, bg);

    if (flag_need_cut == 1)
    {
        int x = (gray.cols - cutcols) / 2;
        int y = (gray.rows - cutrows) / 2;
        gray = gray(cv::Rect(x, y, cutcols, cutrows));
    }





    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0, 0);


    cv::imwrite("d:/jd/t/t0/center.png", gray); 



    auto contours_roi = doChestProcessY(ci.img, bg, BIN_THRES);


  


    vector<cv::Mat> vmat; 


    







    for (size_t i = 0; i < contours_roi.size(); i++) 
    {
        // cv::drawContours(mask, contours_roi, static_cast<int>(i), cv::Scalar(255), cv::FILLED); // 填充轮廓


        auto graybb = clone_contour_bb(gray, contours_roi[i]);

        vmat.push_back(graybb); 
    }



    int cnt = 0; 
    string fn_d = string("d:/jd/t/t0") + "/";
    for (auto e : vmat)
    {
        cv::imwrite(fn_d + to_string(cnt) + ".png", e);
        cnt++;
    }


    //diffimg; 




#if 0
    cv::Mat d_diffimg = mask.clone();
    d_diffimg.setTo(0);

    for (size_t i = 0; i < contours_roi.size(); i++)
    {
        cv::drawContours(mask, contours_roi, static_cast<int>(i), cv::Scalar(255), cv::FILLED); // 填充轮廓

        // get boundbox for contours_roi[i]
        cv::Rect bbox = cv::boundingRect(contours_roi[i]);



        deal_bbox_diff(diffimg, mask, d_diffimg, bbox,2.0f ,D_DIFF_THRES);

        // deal_bbox_diff(diffimg, mask, diffimg, bbox);



        // loop all bbox's pixel value and count the average 



    }
    //ci.s_i(diffimg);

    //diffimg += 100;


    
    cv::Mat min_kernel; // 用于存储最小值的图像
    cv::Mat temp; // 临时图像

    // 使用 3x3 的卷积核
    cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

    // 初始化 min_kernel
    min_kernel = cv::Mat::zeros(d_diffimg.size(), d_diffimg.type());

    // 遍历图像
    for (int r = 1; r < d_diffimg.rows - 1; r++) {
       for (int c = 1; c < d_diffimg.cols - 1; c++) {
           // 提取 3x3 区域
           temp = d_diffimg(cv::Rect(c - 1, r - 1, 3, 3));
           // 计算最小值
           double min_val;
           cv::minMaxLoc(temp, &min_val);
           min_kernel.ptr(r)[c] = static_cast<uchar>(min_val);
       }
    }


    d_diffimg = min_kernel; 


    cv::threshold(d_diffimg, d_diffimg, 1, 255, cv::THRESH_BINARY);
#endif 
    //ci.img = ci.hconcat({ diffimg, gray,/* d_diffimg*/ }, 4);

    //ci.s_i();


    //cv::imwrite("d:/jd/t/t0/diffimg.png", diffimg); 






    

    



    
    









    




    //ci.resize(0.3);
    // ci.s_i();

#endif
#if 0
    
    

    // 使用标准库中的 M_PI
    double pi_value = M_PI; // 或者直接使用 3.14159265358979323846
#endif 

#if 0

    ci.read_img("image_g_r.png"); 
    ci.img;

    cv::Mat binimg; 
    cv::threshold(ci.img, binimg,  155, 255, CV_)


    ci.s_i();


    



   





#if 0

    vector<cv::Mat> channels = { ci.img }; 

    // 计算每个通道的梯度幅值
    std::vector<cv::Mat> gradientMagnitudes;
    for (const auto& channel : channels) {
        cv::Mat Gx, Gy;
        cv::Sobel(channel, Gx, CV_64F, 1, 0, 3); // 水平方向梯度
        cv::Sobel(channel, Gy, CV_64F, 0, 1, 3); // 垂直方向梯度
        cv::Mat gradientMagnitude;
        cv::magnitude(Gx, Gy, gradientMagnitude);
        gradientMagnitudes.push_back(gradientMagnitude);
    }


    cv::normalize(gradientMagnitudes[0], gradientMagnitudes[0], 0, 255, cv::NORM_MINMAX, CV_8U);


    // 取梯度, 取其中最大值 * 3
    cv::Mat gradientMagnitude = gradientMagnitudes[0];
  
    cv::multiply(gradientMagnitude, 3, gradientMagnitude);

    cout << "gradientMagnitude.type() = " << gradientMagnitude.type() << endl;

    cv::imwrite("gradientMagnitude.jpg", gradientMagnitude);

    cv::Mat binaryImage;
    cv::threshold(gradientMagnitude, binaryImage, 70, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);


    // 计算每个轮廓度的凸包
    std::vector<std::vector<cv::Point>> convexHulls;
    for (const auto& contour : contours) {

        if (!filterContour(contour)) {
            continue;
        }

        convexHulls.push_back(contour);
    }

    // 填充白色, 方便下一步继续识别细胞核
    for (const auto& contour : convexHulls) {
        cv::fillPoly(binaryImage, std::vector<std::vector<cv::Point>>{contour}, cv::Scalar(255));
    }


    
    cv::imwrite("fill_binaryImage.jpg", binaryImage);


    contours.clear(); 

#endif 
    // cv::adaptiveThreshold(ci.img, ci.img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 31, 3);












#endif 



#if 0

    const unordered_map<string, string> vss{
        {"a","b"},
        {"a0","b0"},
    
    };



    auto e = vss.at("a");


    cout << e << endl;

#endif
#if 0
    cout_("{}",ci.linspace(2, 3, 3));
#endif
#if 0

    for (auto e : ci.R(1,9,3))
    {
        cout << e << endl; 
    }

#endif

#if 0
    string fn_d = "d:/jd/doc/sw_monograph"; 
    string title = "a"; 
    int max_idx = 399;
    int step = 16;


    if (argc == 2)
    {
        fn_d = string(argv[1]);
    }

    if (argc == 3)
    {
        fn_d = string(argv[1]);
        title = string(argv[2]);
        
    }

    if (argc == 4)
    {
        fn_d = string(argv[1]);
        title = string(argv[2]);
        max_idx = atoi(argv[3]);
    }

    if (argc == 5)
    {
        fn_d = string(argv[1]);
        title = string(argv[2]);
        max_idx = atoi(argv[3]);
        step = atoi(argv[4]);
    }


    from_dat_to_v_p_png_slow(fn_d, title, max_idx, step);

#endif
#if 0
    int rows = 2056;
    int cols = 2464;

    string fn_d = "d:/jd/doc/sw_monograph";
    vector<int> vi(399,0);
    std::iota(vi.begin(), vi.end(), 0);    // generate array list 0 to N

    int i = 0; 
    for (auto e : vi)
    {
        auto e_fn = s_("{}/zidx_a_{}.dat", fn_d, e);

        ci.read_bin_to_mat(e_fn, rows, cols, 1); 
        // ci.resize(1/4.0);
        auto fn_png = s_("{}.png", e_fn);

        if (i % 10 == 0)
        {
            cout << fn_png << endl; 
        }
        cv::imwrite(fn_png, ci.img); 
        i++;
    }

#endif

#if 0
    float multiplyFactor = 0.35; 
    cv::Mat img4C(3330, 5500, CV_8UC4);
    cv::Mat img_resized;

    cv::resize(img4C, img_resized, cv::Size((int)(img4C.cols * multiplyFactor), int(img4C.rows * multiplyFactor)));


    ci.s_i(img_resized);

#endif
#if 0

  
    char* s = ret_test(); 
    for (uint i = 0; i < 4; i++)
    {
        cout << s[i] << endl;
    }

#endif
#if 0
    int w = 9;
    int h = 13;

    vector<uchar> vuc(w*h, 111); 
    cv::Mat id_mat(h, w, CV_8UC1, vuc.data()); 
    cout << id_mat.isContinuous() << endl; 





    ci.create_img_rc_chn(h, w, 2); 
    cout << ci.img.isContinuous() << endl; 


    // ci.s_i(id_mat); 


    //vector<char> vc; 

    //auto t0 = GetTickCount64();
    //for(int i=0;i<1000;i++)
    //    vc = vector<char>(vuc.begin(), vuc.end());
    //auto d = GetTickCount64() - t0; 

    //cout << d << endl;

#endif
#if 0
    ci.create_img_rc_chn(999, 888, 4); 
    cv::Mat dstImg; 
    cv::resize(ci.img, dstImg, cv::Size(ci.img.rows * 2, ci.img.cols * 2), 1);

    ci.s_i(dstImg);

#endif
#if 0

    // jpg bytes array to mat
    auto id_s = ci.read_bin_to_string("d:/jd/t/t0/1.jpg"); 
    vector<char> vchar = vector<char>(id_s.begin(), id_s.end()); 
    auto mat = cv::imdecode(vchar, cv::IMREAD_COLOR); 

    ci.s_i(mat);

#endif
#if 0
    int rows = 512;
    int cols = 512;
    ci.create_img_rc_chn(rows, cols, 1);

    ci.img.setTo(0);

    auto gray = 250;
    cv::drawContours(ci.img, std::vector<std::vector<cv::Point>>({ {{22,33},{333,444}} }), -1, cv::Scalar(gray), 1, cv::LINE_8, cv::noArray(), INT_MAX);

    ci.s_i();

#endif
#if 0


    vector<int> va(55, 0);

    std::iota(va.begin(), va.end(), 0);

    cout_("{}", va);

#endif
#if 0

    int rows = 4;
    int cols = 5;

    // Create a byte array containing data (should be rows * cols in size)  
    // For example: a grayscale image of 4x5 pixels  
    uint8_t byteArray[20] = { 0, 50, 100, 150, 200,
                               25, 75, 125, 175, 225,
                               10, 20, 30, 40, 50,
                               5, 15, 25, 35, 45 };

    cv::Mat id_mat(rows, cols, CV_8UC1, byteArray);

    cout << id_mat << endl; 

    byteArray[4] = 111; 
    cout << id_mat << endl;

#endif

#if 0
    string fn = "d:/jd/t/git/sq2dz/sqrayslide_demo/sqrayslide_demo/sqrayslide_20240202_x64/windows/bin/1.bin";
    int cols = 5120;
    int rows = 2560;
    ci.read_bin_to_mat(fn, rows, cols, 4);
    ci.resize(1.0f / 4);
    ci.s_i();

#endif
#if 0
    string fn = "d:/jd/t/t0/test.png";
    ci.read_img(fn); 

    ci.s_i();

#endif
#if 0
    string fn_d = "d:/jd/t";
    vector<cv::String> v_fn;
    cv::glob(fn_d, v_fn, false);

    for (auto efn : v_fn)
    {
        string efn_s = efn;
        cout << efn_s + ":"+ efn_s << endl;
    }

#endif
#if 0
    vector<double> vi{ 2.2,3,4,5,6,7,8 };

    vi.resize(3);
    cout_("{}", vi);
    cout << endl;

#endif

#if 0
    string fn_d = "d:/jd/t/t0";
    vector<string> v_fn{
    s_("{}/zidx_e_{}.dat", fn_d, 206),
    s_("{}/zidx_e_{}.dat", fn_d, 238),

    };

    vector<cv::Mat> v_mat(v_fn.size());


    int rows = 2056;
    int cols = 2464;
    int step = 1;
    int cnt = 0;
    for (auto e_fn : v_fn)
    {
        ci.read_bin_to_mat(e_fn, rows, cols, 1);


        v_mat[cnt++] = ci.img;

    }

    ci.img = ci.hconcat(v_mat, 9);

    ci.resize(0.3f);
    ci.s_i();

#endif

#if 0
    vector<float> vf{ 2.0f,3.0f,4.5f }; 
    auto ev =  ci.sum_vec(vf);

    cout << ev << endl;

    cout << ci.mean_vec(vf) << endl; 

    cout_("{}\n", ci.diff_v(vf)) ;

#endif

#if 0

    string fn_d = "d:/jd/t/t0";
    vector<string> v_fn{
    s_("{}/zidx_e_{}.dat", fn_d, 206),
    s_("{}/zidx_e_{}.dat", fn_d, 238),

    };

    vector<cv::Mat> v_mat(v_fn.size()); 


    int rows = 2056;
    int cols = 2464;
    int step = 1;
    int cnt = 0;
    for (auto e_fn : v_fn)
    {
        ci.read_bin_to_mat(e_fn, rows, cols, 1);

        cv::Mat mean, stddev;

        cv::meanStdDev(ci.img, mean, stddev); 
        cout << e_fn << endl; 
        cout << mean << stddev << endl; 


        auto sz = min(rows, cols);
        vector<int> vp;
        vector<int> vn;



        for (int i = 0; i < sz; i++)
        {
            auto& e_pv = ci.img.ptr<uchar>(i)[i];
            auto& e_nv = ci.img.ptr<uchar>(i)[sz - i - 1];



            if (i % step == 0)
            {
                vp.push_back(e_pv);
                vn.push_back(e_nv);
            }
        }

        auto vp_vn = ci.combine_vec(vp, vn);

        
        ci.create_img_rc_chn(1, vp_vn.size(), 1); 
       
        std::copy(vp_vn.begin(), vp_vn.end(), ci.img.data);

        cv::meanStdDev(ci.img, mean, stddev); 

        cout << stddev << endl; 

        

        unordered_map<int, int> m_v_c;

        for (auto e : vp_vn)
        {
            m_v_c[e]++;
        }

        vector<int> vi(256, 0);
        
        for (int i = 0; i < vi.size(); i++)
        {
            vi[i] = m_v_c[i];
        }

        auto img = ci.P(vi,0);

        cout_("x={}\n", vi);

        v_mat[cnt] = img; 

        cnt++;
    }

    auto v_merge = ci.img_copy;
    cv::hconcat(v_mat, v_merge);

    ci.s_i(v_merge);

#endif

#if 0

    from_dat_to_v_p("e", 399, 16);

#endif

#if 0
    vector<int> vi(100,0);
    int cnt = 0;
    for (auto& e : vi)
    {
        e = cnt++;
    }


    ci.P(vi);

#endif

#if 0

    ci.img = ci.P(vector<int>{ 1,2,3,4 },0);
    ci.puttext("img:1");
    
    ci.s_i();

#endif
#if 0

    vector<int> vi{ 2,3,4,7,8,9 }; 
    vector<int> diff(vi.size(), 0);
    std::adjacent_difference(vi.begin(), vi.end(), diff.begin()); 
    cout_("{}", diff);

#endif

#if 0

    vector<string> fn_all{
    "d:/jd/t/t0/log_diffsum_a.csv.png",
    "d:/jd/t/t0/log_diffsum_b.csv.png",
    "d:/jd/t/t0/log_diffsum_c.csv.png",
    "d:/jd/t/t0/log_diffsum_d.csv.png",
    "d:/jd/t/t0/log_diffsum_e.csv.png",
    };

    int step = 16;

    vector<int> vstep = { 16,20,25,30,35 }; 
    for (auto step : vstep)
    {

       

        //     for (auto fn : fn_all)
        {
            ci.read_img(fn_all[1]);

            vector<int> vsum;
            vsum.resize(ci.img.rows);


            for (int r = 0; r < ci.img.rows; r++)
            {

                uchar* tr = ci.img.row(r).data;

                int sum = 0;
                for (int c = 0; c < ci.img.cols - step; c += step)
                {

                    //auto prev = tr[c]; 
                    //auto next = tr[c + step];


                    auto diff = abs(tr[c] - tr[c + step]);

                    sum += diff;


                }


                vsum[r] = sum;
            }


            int flag_show = 0;
            cout << step << endl; 
            auto img = ci.P(vsum, flag_show);
            ci.puttext(img, to_string(step));
            ci.s_i(img);

            //ci.s_i(png);
        }





    }


    // visualizeVector(vi, "abs");

#endif

#if 0

    vector<string> fn_all{
    
    "d:/jd/t/t0/log_diffsum_a.csv",
    "d:/jd/t/t0/log_diffsum_b.csv",
    "d:/jd/t/t0/log_diffsum_c.csv",

    "d:/jd/t/t0/log_diffsum_d.csv",
    "d:/jd/t/t0/log_diffsum_e.csv",
    
    };


    for (auto fn : fn_all)
    {
        // string fn = "d:/jd/t/t0/log_diffsum_a.csv";
        vector<string> vs = ci.read_txt_to_vec_str(fn);
        // cout << vs[vs.size() - 1] <<SP <<vs.size()<< endl;

        // assert(0 == 1);
        int rows = vs.size();
        int cols = 0;
        for (auto es : vs)
        {
            vector<string> vp;
            vector<string> vn;

            deal_eline(ci, es, vp, vn);


            cols = vp.size();
            break;
        }


        int intervsz = 40;
        cv::Mat img(rows, cols + intervsz + cols, CV_8UC1);



        int r = 0;


        for (auto es : vs)
        {
            vector<string> vp;
            vector<string> vn;
            deal_eline(ci, es, vp, vn);

            assert(vp.size() == vn.size());

            int c = 0;



            for (auto eps : vp)
            {
                img.ptr<uchar>(r)[c] = atoi(eps.c_str());
                //img.ptr<uchar>(r)[c] = 100;
                c++;
            }

            for (int i = 0; i < intervsz; i++)
            {
                img.ptr<uchar>(r)[c] = 0;
                c++;
            }

            for (auto eps : vn)
            {
                img.ptr<uchar>(r)[c] = atoi(eps.c_str());
                //img.ptr<uchar>(r)[c] = 200;
                c++;
            }


            r++;
        }



        ci.img = img;

        cv::imwrite(fn + ".png", ci.img); 
    }

#endif

#if 0
    ci.create_img_rc_chn(1, 2, 3);


    ci.img.ptr<v3b>(0)[0] = { 1,2,3 }; 

    ci.img.ptr<v3b>(0)[1] = { 1,21,33 };

    ci.serial_to_mat("1.mat");

    ci.deserial_from_mat("1.mat"); 

    
    cv::imwrite("1.png", ci.img); 


    ci.read_img("1.png"); 

    cout << ci.img << endl;

#endif

#if 0

    if (argc >= 2)
    {
        int comidx = opencom(); 

        string sendmsg = "0x11,0x22,0x33,0x44"; 

        while (true)
        {

            string input;

            string response;
            int send_status;
            cout << "send>";

            std::cin >> input;
            if (input == "exit")
            {
                break;
            }
            sendmsg = input;

            sendbytes(comidx, sendmsg, send_status, response);
            
            cout << response << endl;



        }

    }

#endif

#if 0
    ci.create_img_rc_chn(4, 8 + 4 + 8, 1); 

    
    auto rows = ci.img.rows; 
    for (int r = 0; r < rows; r++)
    {
        int c = 0;
        for (int j = 0; j < 8; j++)
        {
            ci.img.ptr(r)[c] = 100; 
            c++;
        }


        for (int j = 0; j < 4; j++)
        {
            ci.img.ptr(r)[c] = 0;
            c++;
        }

        for (int j = 0; j < 8; j++)
        {
            ci.img.ptr(r)[c] = 200;
            c++;
        }
       

    }

    cout << ci.img << endl;

    assert(0 == 1);

#endif
#if 0
    vector<string> vfn = {

        "d:/jd/t/focus_img/zidx_a_104.dat.jpg",
"d:/jd/t/focus_img/zidx_a_139.dat.jpg",
"d:/jd/t/focus_img/zidx_a_54.dat.jpg",

"d:/jd/t/focus_img/zidx_e_399.dat.jpg",
"d:/jd/t/focus_img/zidx_a_73.dat.jpg",


    };


    vector<cv::Mat> vimg; 

    int rows = 2056;
    int cols = 2464;

    vector<cv::Mat> vimg_full; 

    for (auto efn : vfn)
    {
        ci.read_img(efn); 

        
        vimg_full.push_back(ci.img); 

        ci.img = cv::Mat(ci.img, { rows / 2, rows / 2 + rows / 4 }, { rows / 2, rows / 2 + rows / 4 });

        vimg.push_back(ci.img);


       

    }


    for (auto eimg : vimg_full)
    {
        auto id_p = diag_diffsum(eimg,11);

        cout << id_p.first << ", " << id_p.second << endl; 
    }

    cv::Mat catimg;
    cv::hconcat(vimg, catimg);
    ci.s_i(catimg);

#endif
#if 0

    string a(120*120, 'a'); 

    a[10] = 0x1a;

    ci.str_to_bin_file("1.bin",a); 

    ci.read_bin_to_mat("1.bin", 120, 120, 1);

    ci.resize(4);

    ci.s_i();

#endif
#if 0

    string fn = "d:/jd/t/t0/zidx_10.dat"; 

    int cols = 2464;
    int rows = 2056; 



    ci.read_bin_to_mat(fn, rows, cols, 1);
    ci.resize(0.4);
    
    ci.s_i();
    
    auto idbin = ci.read_bin_to_string(fn);
    for (auto e : idbin)
    {
        if (e == 0)
        {
            cout << e << endl; 
        }
    }


    cv::Mat img(rows, cols, CV_8UC1);
    int cnt = 0;

    for (auto r = 0; r < rows; r++)
    {
        for (auto c = 0; c < cols; c++)
        {
            img.ptr<uchar>(r)[c] = idbin[cnt++];
        }
    }


    ci.img = img; 
    ci.resize(0.4);
    ci.s_i();

#endif

#if 0


    ci_0.read_img("d:/jd/t/da101490605.bmp");
    ci_1.read_img("d:/jd/t/t0/da101410605.bmp");


    // ci_0.resize(0.3);

    //ci_1.resize(0.3);


    //ci_0.cvtcolor("GRAY"); 
    //ci_1.cvtcolor("GRAY");

    //cv::imwrite("d:/jd/t/da101490605.bmp", ci_0.img); 
    //cv::imwrite("d:/jd/t/t0/da101410605.bmp", ci_1.img);





    int maxv_thres = 133;
    int minv_thres = 49; 


    vector<uint64_t> cnt_pixels = { 0,0 };


    //ci_0.resize(0.3);
    //ci_1.resize(0.3);


  




    for (int r = 0; r < ci_0.img.rows; r++)
    {
        for (int c = 0; c < ci_0.img.cols; c++)
        {
            auto& ep_0 = ci_0.img.ptr<uchar>(r)[c];
            auto& ep_1 = ci_1.img.ptr<uchar>(r)[c];

            if (ep_0 > minv_thres && ep_0 < maxv_thres)
            {
                cnt_pixels[0]++;
            }
            
            if (ep_1 > minv_thres && ep_1 < maxv_thres)
            {
                cnt_pixels[1]++;
            }

        }
    }





    cout << s_("{}", cnt_pixels);

#endif

#if 0

    vector<double> vd{ 1984,1970,1981,1975,1973,1975,1979,1976,1982,1970,1976,1978,1968,1975,1973,1969,1972,1969,1973,1970,1964,1976,1969,1974,1981,1978,1979,1965,1977,1975,1976,1971,1970,1981,1982,1981,1979,1975,1979,1973,1976,1981,1982,1983,1983,1980,1983,1990,1983,1991,1989,1992,1988,1985,2000,2001,1997,2003,2008,2011,2017,2024,2025,2033,2040,2045,2053,2052,2069,2071,2088,2099,2105,2117,2124,2140,2158,2176,2202,2223,2235,2263,2279,2304,2328,2359,2389,2418,2449,2484,2519,2566,2604,2653,2694,2745,2799,2864,2927,2990,3062,3137,3208,3278,3342,3407,3457,3512,3571,3625,3692,3780,3893,4050,4219,4400,4561,4671,4726,4759,4764,4718,4652,4535,4415,4291,4156,4027,3895,3770,3641,3505,3391,3290,3192,3107,3030,2960,2892,2826,2770,2714,2673,2639,2603,2569,2540,2510,2482,2462,2433,2407,2384,2358,2333,2307,2284,2274,2248,2237,2217,2201,2182,2166,2153,2147,2136,2129,2113,2099,2076,2062,2050,2044,2025,2019,2019,2006,2005,2000,2002,1998,1996,1990,1997,1988,1987,1990,1986,1987,1983,1990,1985,1987,1992,1990,1989,1990,1989,1994,1997,2002,2005,2015,2021,2034,2041,2056,2061,2080,2092,2128,2148,2185,2211,2245,2280,2315,2345,2374,2403,2435,2465,2504,2546,2589,2637,2687,2729,2783,2847,2927,3027,3151,3297,3448,3585,3715,3815,3887,3930,3941,3920,3866,3791,3696,3585,3462,3350,3225,3098,2980,2884,2790,2705,2616,2556,2502,2467,2431,2406,2375,2353,2318,2286,2254,2229,2213,2197,2177,2153,2133,2103,2072,2052,2040,2031,2018,1997 };


    int min_distance = 10; // 最小距离  
    int margin = 10;
    std::vector<Peak> top_peaks = detectTopThreePeaks(vd, min_distance, margin);

    system("Sleep 1000"); 
    //cout << s_("{}", top_peaks);

#endif

#if 0

    struct sa {
        int x;
        float y;
        double z;
    

};


    vector<sa> vsa = {

        {1,2,3.1},
        {1,2,3.1},
        {1,2,0.3},

    
    };

    unordered_map<uint64_t, int> m_;
    
    for (auto& e : vsa)
    {
        BYTE* pb = (BYTE*)(&e);

        uint64_t sum = 0; 

        for (int i = 0; i < sizeof(sa); i++)
        {
            sum += pb[i];
        }

        
        m_[sum]++;
    }


    cout << s_("{}", m_) << endl; 

    cout << m_.size() << endl;
#endif

#if 0

    cout << std::round(1.2351 * 1000.0) / 1000.0 << endl;;

#endif
#if 0
    unordered_map<int*, int> ma;
    vector<int> vi(100, 999); 

    for (auto& e : vi)
    {
        ma[&e] = 1;
    }

    cout << ma.size() << endl;

#endif
#if 0

    unordered_map<string, std::vector<cv::Point2i>> ma;

    ma["cv::Point(3, 4)"] = { {2,3} };

#endif
#if 0

string fn = "d:/jd/t/t0/e6_pos.jpg"; 
ci.read_img(fn); 

SQE6E7Common  id_SQE6E7Common;

id_SQE6E7Common.img = ci.img;


id_SQE6E7Common.toChnHSV(); 
id_SQE6E7Common.toBinaryImg(20);
cv::Mat ci_clear = id_SQE6E7Common.filterContour();

ci.img = ci_clear;
ci.resize(0.2f);

ci.s_i();

#endif

#if 0
string fn = "d:/jd/t/t0/e6_pos.jpg"; 
ci.read_img(fn);

#if 0
for(int r = 0; r < ci.img.rows; r++)
{
    for(int c = 0; c < ci.img.cols; c++)
    {
        // if it is edge 10 pixel, clear to 0
        if(r < 33 || r > ci.img.rows - 33 || c < 33 || c > ci.img.cols - 33)
        {
            ci.img.ptr<cv::Vec3b>(r)[c] = cv::Vec3b(0, 0, 0);
        }
    }
}
#endif 


//ci.resize(0.2f); 
//ci.s_i(); 

cv::Mat HSV;
cv::cvtColor(ci.img, HSV, cv::COLOR_BGR2HSV);

    // 分离 HSV 通道
    std::vector<cv::Mat> v_hsv;
    cv::split(HSV, v_hsv);





auto gray = v_hsv[2].clone();
ci.img = gray; 
ci.normalized();


ci.img = v_hsv[1];
ci.normalized();




// if ci.img value > 10, set to 255, else set to 0
for(int r = 0; r < ci.img.rows; r++)
{
    for(int c = 0; c < ci.img.cols; c++)
    {
        if(ci.img.ptr<uchar>(r)[c] > 10)
        {
            ci.img.ptr<uchar>(r)[c] = 255;
        }
        else
        {
            ci.img.ptr<uchar>(r)[c] = 0;
        }
    }
}

vector<cv::Mat> v_dilate;


cv::imwrite("d:/jd/t/t0/1.jpg", ci.img);






    std::vector<std::vector<cv::Point>> contours;
   

cv::findContours(ci.img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


auto ci_clear = ci.img.clone();
ci_clear.setTo(0);

vector<int> v_contour_size = {};

// draw contours

int cnt_all = 0;

int cnt_positive = 0; 
for(size_t i = 0; i < contours.size(); i++)
{
   
// static all contours size


    v_contour_size.push_back(contours[i].size());


    if (contours[i].size()< 32) { continue; }
  

    cv::Rect out_rectbox = cv::boundingRect(contours[i]);
    
    vector<std::vector<cv::Point>> vc;

    if (out_rectbox.width > 100 && out_rectbox.height > 100)
    {
        cv::drawContours(ci_clear, contours, i, cv::Scalar(128), 3);

        vc.push_back(contours[i]);

        cnt_positive++;
    }
    else
    {
        
        cv::Moments moments = cv::moments(contours[i], true);
        auto center_xy = cv::Point2i(moments.m10 / moments.m00, moments.m01 / moments.m00);
        auto & c = center_xy;
  
        auto& p = gray;

        int sum = 0;
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {

               sum +=  p.ptr(c.y + i)[c.x + j];
            }
        }

        auto mean = sum / 9.0f;

        if (mean < 211)
        {
            cv::drawContours(ci_clear, contours, i, cv::Scalar(255), 2);
        }

        
        
    }

    cnt_all++;
}




std::cout << cnt_positive << " / " << cnt_all  << endl;



ci.img = ci_clear;


cv::imwrite("d:/jd/t/t0/2.jpg", ci_clear);

ci.resize(0.3f);

ci.s_i();



//ci.s_i();




// cv::GaussianBlur(hsvChannels[2], hsvChannels[2], cv::Size(3, 3), 0);

// cv::threshold(hsvChannels[2], hsvChannels[2], 188, 255, cv::THRESH_BINARY);


  //cv::adaptiveThreshold(hsvChannels[2], hsvChannels[2], 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 15);  

  //cv::adaptiveThreshold(hsvChannels[1], hsvChannels[1], 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 11);

#if 0

  cv::imwrite("d:/jd/t/t0/1.jpg", hsvChannels[1]);
  
  auto ecd_cmd = R"(ecd d:\jd\t\t0\1.jpg)";


  system(ecd_cmd);


cv::hconcat(hsvChannels, ci.img);
ci.resize(0.21f);
#endif 


// ci.s_i();

#if 0

vector<cv::Mat> v_img;
cv::Mat v_img_all;

// the edge of ci.img , clear to 0




cv::split(ci.img, v_img);

auto g_img = v_img[1]; 

   cv::GaussianBlur(g_img, g_img, cv::Size(5, 5), 0);  
    cv::GaussianBlur(g_img, g_img, cv::Size(3, 3), 0);  

  cv::adaptiveThreshold(g_img, g_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 3);  




ci.img = g_img; 


cv::imwrite("d:/jd/t/t0/1.jpg", g_img);

#endif

#if 0
cv::hconcat(v_img, ci.img);

ci.resize(0.15f);

ci.s_i();
#endif

#if 0
cv::threshold(img_g, img_g, 144, 255, cv::THRESH_BINARY); // get binary overlay picture

ci.img = img_g;

cv::imwrite("d:/jd/t/t0/1.jpg", ci.img);

ci.s_i();
#endif

#endif

#if 0
cv::Mat img(300,500,CV_8UC1); 

for(int r = 0; r < img.rows; r++)
{
    for(int c = 0; c < img.cols; c++)
    {
        img.ptr<uchar>(r)[c] = 0;
    }
}

// row < 22, col <44, set to 9

for(int r = 0; r < img.rows; r++)
{
    for(int c = 0; c < img.cols; c++)
    {
        if(r < 22 || c < 44)
        {
            img.ptr<uchar>(r)[c] = 9;
        }
    }
}

ci.img = img; 
// ci.s_i();


ci.normalized();
ci.s_i();

#endif

#if 0
    string fn = "d:/jd/t/t0/p16test.png"; 


    ci.read_img(fn); 

    //ci.s_i();

    cv::Mat hsv;
    cvtColor(ci.img, hsv, COLOR_BGR2HSV);
    cv::Mat mask1;
    cv::Mat mask2;

    inRange(hsv, Scalar(0, 150, 70), Scalar(3, 255, 255), mask1);
    inRange(hsv, Scalar(150, 150, 70), Scalar(180, 255, 255), mask2);

    cv::Mat mask;
    mask = mask1 | mask2;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (contourArea(contours[i]) < 700)
        {
            continue;
        }
        double pixelRedThreshold = area / 50;
        Rect rect = boundingRect(contours[i]);
        Mat roi = ci.img(rect);
        Mat hsv_roi;
        cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
        Mat mask_roi;
#if 0			
        inRange(hsv_roi, Scalar(0, 70, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv_roi, Scalar(180, 70, 70), Scalar(180, 255, 255), mask2);
#endif

#if 1
        inRange(hsv_roi, Scalar(0, 150, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv_roi, Scalar(180, 150, 70), Scalar(180, 255, 255), mask2);
#endif 

        mask_roi = mask1 | mask2;

        int cropSize = 256;

        int count = countNonZero(mask_roi);
        if (count > pixelRedThreshold) {
            Moments m = moments(contours[i]);
            Point centroid(m.m10 / m.m00, m.m01 / m.m00);
            int x = centroid.x - cropSize / 2;
            int y = centroid.y - cropSize / 2;
            x = max(x, 0);
            y = max(y, 0);
            x = min(x, ci.img.cols - cropSize);
            y = min(y, ci.img.rows - cropSize);
            Mat crop = ci.img(Rect(x, y, cropSize, cropSize));
            /*      int globalX = globalPos.split("_").first().toUInt();
                  int globalY = globalPos.split("_").last().toUInt();*/
            Mat brightnessMat;
            extractChannel(hsv_roi, brightnessMat, 2);
            Scalar averageBrightness = mean(brightnessMat);
            double value = 255 - averageBrightness.val[0];
            if (value > 100)
            {
                value = 100;
            }
            // resultMatMap.insert(QString("%1_%2_%3").arg(centroid.x + globalX).arg(centroid.y + globalY).arg(value), crop);
        }
    }


    int t = 0;
#endif
#if 0


   

    auto run_test = [](char *aa) {
        for (int i = 0; i < 4; i++)
        {
            cout << (int)aa[i] << endl; 
        }
        };

    char a[88] = { 0x12,0x44,0x55 }; 

    char* pa = a; 
    for (int i = 0; i < 4; i++)
    {
        cout << (int)pa[i] << endl;
    }
    run_test(a);

#endif
#if 0
     ci.read_img("d:/jd/t/t0/id_mat_mask0.jpg"); 
   // ci.read_img("d:/jd/t/t0/g_img.jpg"); 

     cout << ci.img << endl; 

     ci.s_i(ci.img);


     bool t = ifMaskIsFilled(ci.img);
     cout << t << endl; 



     system("pause");

#endif
#if 0
// get g img
    ci.read_img("D:/jd/t/smb_share/t/test_rgb2y/边缘分割/01_03.jpg");


    ci.img = cv::Mat(ci.img, cv::Rect(1000, 1000, 1500, 1400));

    cv::Mat HSV;
  cv::cvtColor(ci.img, HSV, cv::COLOR_BGR2HSV);

  vector<cv::Mat> chn_all;
  cv::split(HSV, chn_all);
  auto v_chn = chn_all[2];

  
  v_chn.convertTo(v_chn, CV_8U);



  // v_chn = ~v_chn; 


  ci.img = v_chn; 
    ci.s_i();

#endif

#if 0
    // from g to contour

    ci.read_img("d:/jd/t/t0/g_img.jpg"); 



    cv::threshold(ci.img, ci.img, 77, 9, 0);

    
    //cv::imwrite("d:/jd/t/t0/binary.jpg", ci.img); 



    ci.s_i();



    int hist_info[256] = { 0 };

    //ci.img.forEach<uchar>([&](uchar& pixel, const int* position) {
    //            
    //    hist_info[pixel]++;


    //            });


//    string ss = s_("{}", hist_info);
    //std::cout << ss << std::endl;

#endif
#if 0
    // get g img
    ci.read_img("D:/jd/t/smb_share/t/test_rgb2y/边缘分割/01_03.jpg"); 


    ci.img = cv::Mat(ci.img, cv::Rect(1000, 1000, 1500, 1400));


    std::vector<cv::Mat> v_img;
    cv::split(ci.img, v_img);


    cv::hconcat(v_img, ci.img);

    



    auto& g_img = v_img[1]; 

    cv::imwrite("d:/jd/t/t0/g_img.jpg", g_img);
    

    ci.s_i();

#endif
#if 0

    uint64_t u64_addr = 782695920232LL;

#endif

#if 0

    const char* shm_name = "MySharedMemory"; // 使用 ANSI 字符串  
    const int SIZE = 4096;

    // 创建共享内存  
    HANDLE shm = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, SIZE, shm_name);
    if (shm == NULL) {
        std::cerr << "Could not create file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    // 将共享内存映射到当前进程的地址空间  
    void* ptr = MapViewOfFile(shm, FILE_MAP_ALL_ACCESS, 0, 0, SIZE);
    if (ptr == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        return 1;
    }

    // 写入数据到共享内存  
    const char* message = "Hello, Shared Memory!";

    // 使用 strcpy_s 替代 strcpy  
    // 目标缓冲区是 SIZE，所以使用 SIZE 作为参数  
    errno_t err = strcpy_s(static_cast<char*>(ptr), SIZE, message);
    if (err != 0) {
        std::cerr << "Error copying to shared memory: " << err << std::endl;
        return 1;
    }

    // 读取共享内存中的数据  
    std::cout << static_cast<char*>(ptr) << std::endl;

    // 清理  
    UnmapViewOfFile(ptr);
    CloseHandle(shm);

    system("pause");

    return 0;

#endif
#if 0


    const char* shm_name = "MySharedMemory";
    const int SIZE = 4096;

    // 打开共享内存  
    HANDLE shm = OpenFileMappingA(FILE_MAP_READ, FALSE, shm_name);
    if (shm == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    // 映射共享内存到这个进程的地址空间  
    void* ptr = MapViewOfFile(shm, FILE_MAP_READ, 0, 0, SIZE);
    if (ptr == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(shm);
        return 1;
    }

    // 读取共享内存中的数据  
    std::cout << "Data from shared memory: " << static_cast<char*>(ptr) << std::endl;

    // 清理  
    UnmapViewOfFile(ptr);
    CloseHandle(shm);

    system("pause");

#endif

#if 0

    int * tt = new int(); 
    *tt = 9999;
    uint64_t u64_addr = (uint64_t) (tt);

    cout <<  to_string(u64_addr) << endl; 

    auto & int_copy =  *(int*)u64_addr;

    cout << int_copy << endl;
    ci.fcout("1.txt", to_string(u64_addr));
            
   system("pause");

#endif

#if 0

    ci.read_img("d:/jd/t/rgb_03_06.jpg"); 
    ci.resize(0.3);


  


    auto img_z = cv::Mat(ci.img.size(), CV_8UC1).setTo(111); 


    ci.s_i(img_z);

#endif

#if 0


    string fn_rgb_r0 = "D:/jd/t/t0/comp_sq_24a/t0/rgb_old_12_04.jpg"; 
    string fn_y_r0 = "D:/jd/t/t0/comp_sq_24a/t0/y_old_12_04.jpg";
    string fn_y_mix_r0 = "D:/jd/t/t0/comp_sq_24a/t0/y_old_mix_12_04.jpg";

    vector<string> fn_arr_src = {
        "12_04",  "10_05",  "09_08",  "09_02",  "07_05",  "05_03",  "05_01",  "04_05",  "04_03",  "02_06",
    }; 

   
   vector<float> vf =  {-0.01f,-0.11f,-0.47f,1.39f};

    for (auto efn: fn_arr_src)
    {
        auto fn_rgb = fn_rgb_r0;
        auto fn_y = fn_y_r0;
        auto fn_y_mix = fn_y_mix_r0;

        fn_rgb = replace_str(fn_rgb, "12_04", efn); 
        fn_y = replace_str(fn_y, "12_04", efn); 
        fn_y_mix = replace_str(fn_y_mix, "12_04", efn); 


        auto & ci_rgb = ci_0; 
        auto & ci_y = ci_1; 

        ci_rgb.read_img(fn_rgb);
        ci_y.read_img(fn_y);

        auto f0 = vf[0];
        auto f1 = vf[1];
        auto f2 = vf[2];
        auto f3 = vf[3];

        processCurrentBackground(ci_rgb.img, ci_y.img, f0, f1, f2,f3);
        cv::imwrite(fn_y_mix, ci_y.img); 

        cout << fn_y_mix << endl; 

         ci_y.resize(0.2); ci_y.s_i();

        
    
    }

#endif

#if 0


    string cml = ci.run_cmd("dir /b D:\\jd\\t\\t0\\comp_sq_24a\\*.jpg");
    vector<string> fc = ci.split_str_2_vec(cml, '\n');

    string dir_r = "D:/jd/t/t0/comp_sq_24a/";
    string dir_w = "D:/jd/t/t0/comp_sq_24a/t0/";


    for (auto e : fc)
    {
        if (e.find(".jpg") != string::npos)
        {
            string fn_r = dir_r + e;
            string fn_w = dir_w + e; 

            cout << fn_r << endl;


           

            
            ci.read_img(fn_r); 
            
            const int edge_fill_sz = 30; 

            for (auto r = 0; r < ci.img.rows; r++)
            {
                for (auto c = 0; c < ci.img.cols; c++)
                {
                    if (r < edge_fill_sz || c<edge_fill_sz || r >  ci.img.rows - edge_fill_sz || c>ci.img.cols- edge_fill_sz)
                    {

                        if (ci.img.channels() == 3)
                        {
                            ci.img.ptr<cv::Vec3b>(r)[c] = cv::Vec3b(255, 255, 255);
                        }
                        else
                        {
                            ci.img.ptr<uchar>(r)[c] = 255;
                        }

                    }
                }
            }

            //ci.resize(0.3);  ci.s_i();

            //break;


            cv::imwrite(fn_w, ci.img); 
            cout << fn_w << endl;



        }
    }

    cout << "END" << endl;

#endif

#if 0

vector<string> v_fn = {
    "y_old_10_05.jpg","y_old_09_02.jpg","y_old_09_08.jpg","y_old_07_05.jpg",
    "y_old_05_01.jpg","y_old_05_03.jpg","y_old_04_03.jpg","y_old_04_05.jpg","y_old_02_06.jpg"
};

string dir_ = "D:/jd/t/t0/comp_sq_24a/";


for (auto &e : v_fn)
{
    e = dir_ + e;
    ci.read_img(e);  

    //ci.resize(0.31);

    cv::Mat rotatedImage;

    //ci.s_i();
    cv::rotate(ci.img, rotatedImage, cv::ROTATE_180);

    //cv::hconcat(ci.img, rotatedImage, ci.img); 

    //ci.s_i();

    cv::imwrite(e, rotatedImage); 



    //break;
}

#endif

#if 0

    int t = 99;
    /*
    comment;
    int t = 8; 

    */

    int a = 9; 
    int b = 99;

#endif

#if 0

    vector<cv::Vec2i> vp = {
        {179,99},
        {171,125},
        {181,93},
        {173,95},

        {171,93},
        {180,95},
        {182,98},
        {180,93},

        {175,97},
        {283,74},
    };

    int w = 2464;
    int h = 2056; 

    for (int i = 0; i < 10; i++)
    {

        string fn_new = "d:/jd/t/t0/tile_cut_new_" + to_string(i) + ".jpg";


        ci.read_img(fn_new);

        cv::Mat mat_cut = ci.img(cv::Rect(vp[i][0], vp[i][1], w, h));
        string fn_new_new = "d:/jd/t/t0/tile_cut_new_new_" + to_string(i) + ".jpg";

		cv::imwrite(fn_new_new, mat_cut);

    }
#endif

#if 0
    int cnt = 0;

    for (int i = 0; i <= 9; i++)
    {
        if (i != 9) continue;
        int idx = i;
        string fn = "d:/jd/t/t0/tile_cut_" + to_string(idx) + ".jpg";

        ci.read_img(fn);


        ci.resize(0.37);


        ci.img;

        string fn_new = "d:/jd/t/t0/tile_cut_new_" + to_string(idx) + ".jpg";
        cv::imwrite(fn_new,ci.img);
        cnt++;
    }

    assert(0 == 1);

#endif

#if 0
    unordered_map<int, int> um = { {1,2} };

    vector<cv::Rect> vp;

	vp.push_back(cv::Rect(0, 0, 10, 10));
	vp.push_back(cv::Rect(10, 0, 10, 10));
	vp.push_back(cv::Rect(20, 0, 10, 10));
    
    vp[0].width;


    cout << um.size() <<endl;

#endif

#if 0
    string fff = R"(d:\jd\t\rgb_y_comp.png)";
    ci.read_img(fff);
    ci.s_i();
#endif

#if 0

    string fn = R"(d:\jd\t\rgb_y_comp.png)";

    ci.read_img(fn);
    int hist[256] = { 0 };

   


    ci.resize(0.33);
    ci.s_i();




    
    //cv::resize(ci.img, ci.img, cv::Size( ci.img.cols / 4, ci.img.rows / 4));
    //ci.s_i();

#endif
#if 0
    fn_r0 = string("D:/jd/t/git/dna-analysis/images/img/comp/new_rgb.jpg");  
    ci.read_img(fn_r0);

    int sliderValue = 0; // ??始值为0
    namedWindow("Image", WINDOW_AUTOSIZE);
    cv::createTrackbar("Coefficient", "Image", &sliderValue, 100, on_trackbar); // 61 = 31 - (-30)
    while (true)
    {
        cout << sliderValue << endl;

        cv::Mat img_small;
        cv::resize(ci.img, img_small, cv::Size(ci.img.cols /3, ci.img.rows / 3));

        imshow("Image", img_small);


        ci.td_sleep(0.3);

        if (waitKey(30) == 'q') {
            break;
        }

        
    }
    // ?头???源
    destroyAllWindows();

#endif

#if 0
    int scale_f = 4;
    float f0, f1, f2, f3;
    int i0, i1, i2, i3;
    f0 = f1 = f2 = f3=0;

    vector<float> v_f0123 = vector<float>({ -0.01f,-0.11f,-0.47f,1.39f });

    f0 = v_f0123[0] ;
	f1 = v_f0123[1] ;
    f2 = v_f0123[2] ;
    f3 = v_f0123[3] ;

    //i0=i1=i2=i3=0;
    i0 = factor_f_to_i(f0);
	i1 = factor_f_to_i(f1);
	i2 = factor_f_to_i(f2);
	i3 = factor_f_to_i(f3);

    string fn_standard_y = "d:/jd/t/t0/y_07_07.new_standard.jpg";

    fn_r0 = string("D:/jd/t/git/dna-analysis/images/img/comp/new_rgb.jpg");
    ci_0.read_img(fn_r0);
    auto img_r0 = ci_0.img.clone();

    ci_1.read_img(fn_standard_y);
    cv::resize(ci_1.img, ci_1.img, cv::Size(ci_1.img.cols/scale_f, ci_1.img.rows/ scale_f));
    cv::Mat img_y_standard_3c;
    cv::cvtColor(ci_1.img, img_y_standard_3c, cv::COLOR_GRAY2BGR);


    auto  img_y_standard = img_y_standard_3c.clone();







    namedWindow("img", WINDOW_AUTOSIZE);
    cv::createTrackbar("f0_b", "img", &i0, 100, on_trackbar_f0); // 61 = 31 - (-30)
    cv::createTrackbar("f1_g", "img", &i1, 100, on_trackbar_f1); // 61 = 31 - (-30)
    cv::createTrackbar("f2_r", "img", &i2, 100, on_trackbar_f2); // 61 = 31 - (-30)
    cv::createTrackbar("f3_y", "img", &i3, 100, on_trackbar_f3); // 61 = 31 - (-30)



    while (true)
    {

        ci_0.img = img_r0.clone();

        auto& _orgRGB = ci_0.img;

        cv::Mat _orgGrey;
        Rgb2yV2(_orgRGB, _orgGrey);
       

        f0 = factor_i_to_f(i0); 
		f1 = factor_i_to_f(i1);
		f2 = factor_i_to_f(i2);
		f3 = factor_i_to_f(i3);


        cout << "i0:" << i0 << ", i1:" << i1 << ", i2:" << i2 << ", i3:" << i3 << endl;
        cout << "f0:" << f0 << ", f1:" << f1 << ", f2:" << f2 << ", f3:" << f3 << endl;

#if 1
        // f0:0.0799999, f1:-0.24, f2:-0.12, f3:1.24
f0 = 0.0799999f;
f1 = -0.24f;
f2 = -0.12f;
f3 = 1.24f;
#endif 

        processCurrentBackground(_orgRGB, _orgGrey, f0, f1, f2,f3);
        cv::imwrite("d:/jd/t/t0/_orggrey.bmp", _orgGrey);
        break;




        // show image 

        cv::Mat img_small_color;
        
        cv::resize(_orgRGB, img_small_color, cv::Size(_orgRGB.cols / scale_f, _orgRGB.rows / scale_f));

		cv::Mat img_small_grey;
		cv::resize(_orgGrey, img_small_grey, cv::Size(_orgGrey.cols / scale_f, _orgGrey.rows / scale_f));

        vector<cv::Mat> v_img = { img_small_grey, img_small_grey, img_small_grey };
        cv::Mat img_3c_for_grey = cv::Mat(img_small_grey.rows, img_small_grey.cols, CV_8UC(3));
        cv::merge(v_img, img_3c_for_grey);

        cv::Mat img_small_color_grey;
        cv::hconcat(img_small_color, img_3c_for_grey, img_small_color_grey);
        cv::hconcat(img_small_color_grey, img_y_standard, img_small_color_grey);

        imshow("img", img_small_color_grey);

        //ci.td_sleep(0.3);

		if (waitKey(30) == 'q') {
			break;
		}

        //break;

    }

    destroyAllWindows();




    int t = 999;

#endif

#if 0
    unordered_map<string, pair<string, string>> fn_map
    {
        {"a0",{"123.jpg","a0.jpg"}},
        {"a1",{"123.jpg","a1.jpg"}},
            
    };

    auto pfs = fn_map["a0"]; 
	cout << pfs.first << endl; 
	cout << pfs.second << endl;

#endif
#if 0

    cv::Scalar mean, stddev; 
    ci.read_img("d:/jd/t/rgb_y_comp.png");;
    cv::meanStdDev(ci.img, mean, stddev);

    cout << mean << endl; 
    cout << stddev << endl;

#endif

#if 0

    struct ObjectParaSmall {
        float Area, Diameter, Perimeter, ShapeFactor;
        float DnaIndex;
        BYTE CellType, TypeReserved[3];
        int CoordinateX, CoordinateY;
        float CentX, CentY, FieldX, FieldY;
        BOOL IsSelected;
        ULONGLONG  CellImageOffset;
        int CellWidth, CellHeight, CellChannel;
        float IOD, NucDABIOD, Hejiangnbi, CytoMeanDAB;
        float Reserved;
        ULONGLONG ParaOffset;
    };

    string obj = ci.read_bin_to_string("D:/jd/t/di_4431.bin");

    
    ObjectParaSmall* obj_ptr = (ObjectParaSmall*)obj.data();

    cout << obj_ptr->DnaIndex << endl;

#endif

#if 0

    string fn_string = "D000011411_y.jpg D000011410_y.jpg D000011409_y.jpg D000011408_y.jpg D000011407_y.jpg D000011406_y.jpg D000011405_y.jpg D000011404_y.jpg D000011403_y.jpg D000011402_y.jpg D000011401_y.jpg D000011400_y.jpg D000011311_y.jpg D000011310_y.jpg D000011309_y.jpg D000011308_y.jpg D000011307_y.jpg D000011306_y.jpg D000011305_y.jpg D000011304_y.jpg D000011303_y.jpg D000011302_y.jpg D000011301_y.jpg D000011300_y.jpg D000011211_y.jpg D000011210_y.jpg D000011209_y.jpg D000011208_y.jpg D000011207_y.jpg D000011206_y.jpg D000011205_y.jpg D000011204_y.jpg D000011203_y.jpg D000011202_y.jpg D000011201_y.jpg D000011200_y.jpg D000011111_y.jpg D000011110_y.jpg D000011109_y.jpg D000011108_y.jpg D000011107_y.jpg D000011106_y.jpg D000011105_y.jpg D000011104_y.jpg D000011103_y.jpg D000011102_y.jpg D000011101_y.jpg D000011100_y.jpg D000011011_y.jpg D000011010_y.jpg D000011008_y.jpg D000011007_y.jpg D000011005_y.jpg D000011004_y.jpg D000011002_y.jpg D000011001_y.jpg D000011000_y.jpg D000010911_y.jpg D000010910_y.jpg D000010909_y.jpg D000010908_y.jpg D000010907_y.jpg D000010906_y.jpg D000010905_y.jpg D000010904_y.jpg D000010903_y.jpg D000010902_y.jpg D000010901_y.jpg D000010900_y.jpg D000010811_y.jpg D000010810_y.jpg D000010809_y.jpg D000010808_y.jpg D000010807_y.jpg D000010806_y.jpg D000010805_y.jpg D000010804_y.jpg D000010803_y.jpg D000010802_y.jpg D000010801_y.jpg D000010800_y.jpg D000010711_y.jpg D000010710_y.jpg D000010708_y.jpg D000010707_y.jpg D000010705_y.jpg D000010704_y.jpg D000010702_y.jpg D000010701_y.jpg D000010700_y.jpg D000010611_y.jpg D000010610_y.jpg D000010609_y.jpg D000010608_y.jpg D000010607_y.jpg D000010606_y.jpg D000010605_y.jpg D000010604_y.jpg D000010603_y.jpg D000010602_y.jpg D000010601_y.jpg D000010600_y.jpg D000010511_y.jpg D000010510_y.jpg D000010509_y.jpg D000010508_y.jpg D000010507_y.jpg D000010506_y.jpg D000010505_y.jpg D000010504_y.jpg D000010503_y.jpg D000010502_y.jpg D000010501_y.jpg D000010500_y.jpg D000010411_y.jpg D000010410_y.jpg D000010408_y.jpg D000010407_y.jpg D000010405_y.jpg D000010404_y.jpg D000010402_y.jpg D000010401_y.jpg D000010400_y.jpg D000010311_y.jpg D000010310_y.jpg D000010309_y.jpg D000010308_y.jpg D000010307_y.jpg D000010306_y.jpg D000010305_y.jpg D000010304_y.jpg D000010303_y.jpg D000010302_y.jpg D000010301_y.jpg D000010300_y.jpg D000010211_y.jpg D000010210_y.jpg D000010209_y.jpg D000010208_y.jpg D000010207_y.jpg D000010206_y.jpg D000010205_y.jpg D000010204_y.jpg D000010203_y.jpg D000010202_y.jpg D000010201_y.jpg D000010200_y.jpg D000010111_y.jpg D000010110_y.jpg D000010109_y.jpg D000010108_y.jpg D000010107_y.jpg D000010106_y.jpg D000010105_y.jpg D000010104_y.jpg D000010103_y.jpg D000010102_y.jpg D000010101_y.jpg D000010100_y.jpg D000010011_y.jpg D000010010_y.jpg D000010009_y.jpg D000010008_y.jpg D000010007_y.jpg D000010006_y.jpg D000010005_y.jpg D000010004_y.jpg D000010003_y.jpg D000010002_y.jpg D000010001_y.jpg D000010000_y.jpg D000010403_y.jpg D000010703_y.jpg D000011003_y.jpg D000011006_y.jpg D000011009_y.jpg D000010709_y.jpg D000010409_y.jpg D000010406_y.jpg D000010706_y.jpg";

    auto v_fn = ci.split_str_2_vec(fn_string, ' ');




    auto do_chn1_to_chn3 = [&ci](const string & e_fn) {

        _chdir("Y:/t0/");;

        string fn = "d:/jd/t/D000010107_y.jpg";

        fn = string(e_fn);

        ci.read_img(fn);

        vector<cv::Mat> v_img = { ci.img, ci.img, ci.img };

        cv::Mat img_3c = cv::Mat(ci.img.rows, ci.img.cols, CV_8UC(3));

        cv::merge(v_img, img_3c);


        auto fn_new = replace_str(fn, "_y", "");
        cout << fn_new << endl;

        cv::imwrite(fn_new, img_3c);
        
        
        
        };
    

    for (auto e_fn: v_fn)
    {
        do_chn1_to_chn3(e_fn);
    }


    
   return 0;

    //ci.s_i(ci.img);

#endif
#if 0

    string fn = "d:/jd/t/git/dna-analysis/images/img/comp/new_rgb.jpg";

	ci.read_img(fn);

    cv::Mat value;
    rgb2y_v2(ci.img, value);




    ci.s_i(value); 
    cv::imwrite("d:/jd/t/1.jpg", value);

#endif
#if 0

    string fn = "d:/jd/t/git/dna-analysis/images/img/comp/new_y.bmp"; 

    ci.read_img(fn); 



    ci.s_i(); 


    cv::imwrite("d:/jd/t/git/dna-analysis/images/img/comp/new_y.jpg", ci.img);

#endif
#if 0
    string fn = "d:/jd/t/wu_uc3.bmp";

    ci.read_img(fn); 

    for (auto i = 0; i < ci.img.rows; i++)
    {
        for (auto j = 0; j < ci.img.cols; j++)
        {

            auto& ep = ci.img.ptr<cv::Vec3b>(i)[j];
            

            if (i < 100 && j < 100)
            {
                ep = cv::Vec3b(0,255,0);

            }
        }
    }


    ci.s_i(); 

    for (auto i = 0; i < ci.img.rows; ++i)
    {
        for (auto j = 0; j < ci.img.cols / 2; ++j)
        {
            // swap [j] with [cols-j]

            auto& front = ci.img.ptr<cv::Vec3b>(i)[j];
            auto& tail = ci.img.ptr<cv::Vec3b>(i)[ci.img.cols-j];

            swap(front, tail); 

        }

    }


// for i
    // for j
    // get pixel of ci.img 
    for (auto i = 0; i < ci.img.rows; ++i)
    {
        for (auto j = 0; j < ci.img.cols; ++j)
        {
			auto& ep = ci.img.ptr<cv::Vec3b>(i)[j];
		

        }
    }

#endif

#if 0
    string fn = "d:/jd/t/git/dna-analysis/images/img/comp/new_rgb.jpg"; 
    ci.read_img(fn); 

    auto img_y =  cv::Mat(ci.img.rows, ci.img.cols, CV_8UC(1));
    // loop ci.img , get a pixel
    for(auto i = 0; i < ci.img.rows; ++i)
    {
    
        for(auto j=0; j < ci.img.cols; ++j)
		{
			auto & p = ci.img.ptr<cv::Vec3b>(i)[j];
            auto& p_y = img_y.ptr<uchar>(i)[j]; 

            
            auto& b = p[0]; 
            auto& g = p[1]; 
            auto& r = p[2];

            float x  = r * 0.5 + g * 0.41 + b * 0.09;
            if (x > 255) x = 255; 
            if (x < 0) x = 0; 
            p_y = (uchar)x;



		}
    }




    ci.s_i(img_y);

#endif
#if 0

    string fn = "d:/jd/t/rgb_y_comp.png"; 

    unordered_map<string, pair<cv::Rect2i,cv::Rect2i>> map_s_rect = {
        {"red_main", {cv::Rect2i(164, 6, 9, 10),cv::Rect2i(192,8, 5,7)}},

        {"red_center", {cv::Rect2i(166, 24, 3, 3),cv::Rect2i(193,24, 4,3)}},

        {"blue_main", {cv::Rect2i(163, 40, 6, 4),cv::Rect2i(189,39, 8,5)}},
        {"blue_center", {cv::Rect2i(163, 53, 5, 4),cv::Rect2i(190,53, 5,4)}},

        {"bg_light", {cv::Rect2i(163, 69, 8, 7),cv::Rect2i(189,68, 10,6)}},
        {"bg_dark", {cv::Rect2i(155, 83, 22, 26),cv::Rect2i(185,81, 18,27)}},

        //{"total", {cv::Rect2i(151,122,228,84), cv::Rect2i(393, 125, 231, 81)} },
        {"rand", {cv::Rect2i(160,128,12,7), cv::Rect2i(191, 125, 10, 6)}}

        
    };

    unordered_map<string, pair<cv::Scalar, cv::Scalar>> map_s_mean = {
    };

    ci.read_img(fn); 

    auto img_bgra = ci.img.clone(); 

    for (auto& kv : map_s_rect)
    {
        auto sk = kv.first; 
        auto rect_rgb = kv.second.first; 
        auto rect_y = kv.second.second;

        auto rgb = ci.img(rect_rgb);
        auto y = ci.img(rect_y); 

        auto mean_rgb = cv::mean(rgb); 
        auto mean_y = cv::mean(y);

        //cout << mean_rgb << endl; 
        //cout << mean_y << endl; 

        map_s_mean[sk] = { mean_rgb,mean_y};
        
        



    }

    // total 7 factors
    vector<string> factors = {
    "f0","f1",
	"f2","f3",
	"f4","f5",
	"f6",
	};

    for (auto& ekv : map_s_mean)
    {
		auto sk = ekv.first;
		auto mean_rgb = ekv.second.first;
		auto mean_y = ekv.second.second;
        //cout << "[" << sk << "]" <<  endl;

        auto b = mean_rgb[0];   
		auto g = mean_rgb[1];   
		auto r = mean_rgb[2];
        auto y = mean_y[0];
        
        vector<double> vals = {
        b* b, b,
        g * g, g,
        r* r, r,
        y
        };

        int i = 0;
			//cout << factors[i] << ":" << vals[i] << endl;
        printf("%s*%.2f + %s*%.2f + %s*%.2f + %s*%.2f + %s*%.2f + %s*%.2f + %s == %.2f,\n",
            factors[0].c_str(), vals[0], factors[1].c_str(), vals[1], factors[2].c_str(), vals[2],
            factors[3].c_str(), vals[3], factors[4].c_str(), vals[4], factors[5].c_str(), vals[5],
            factors[6].c_str(), vals[6]);
              
  
        //cout << mean_rgb << endl; 
        //cout << mean_y << endl;
    }


    vector<double> solver = { -0.054119469,-141.1252,-1.1801796,462.12546,0.97459562,-293.98101,8480.568 };

    auto f0 = solver[0];
    auto f1 = solver[1];
    auto f2 = solver[2]; 
    auto f3 = solver[3]; 
    auto f4 = solver[4]; 
    auto f5 = solver[5];
    auto f6 = solver[6]; 

    
    double  vals[] = {108.70,79.31, 176.88}; 
    for (auto& ekv : map_s_mean)
    {
        auto kv = ekv.second;
auto b = kv.first[0];   
		auto g = kv.first[1];   
		auto r = kv.first[2];

    }
    auto b = vals[0]; 
    auto g = vals[1];
    auto r = vals[2];

    auto e = f0 * b * b + f1 * b +
        f2 * g * g + f3 * g +
        f4 * r * r + f5 * r +
        f6; 
    cout << e << endl; 


    // traverse ci.img

 

    string fn_0 = "d:/jd/t/git/dna-analysis/images/img/comp/new_rgb.jpg"; 
    ci_0.read_img(fn_0); 
    img_bgra = ci_0.img; 

    //ci.s_i(img_bgra);


    auto img_y = cv::Mat(img_bgra.rows, img_bgra.cols, CV_32F);
    img_y.setTo(0);


    for (auto i = 0; i < img_bgra.rows; i++)
    {
        for (auto j = 0; j < img_bgra.cols; j++)
        {
            auto & bgra = img_bgra.ptr<cv::Vec3b>(i)[j];
            auto& y_val = img_y.ptr<float>(i)[j];

            auto b = bgra[0]; 
            auto g = bgra[1]; 
            auto r = bgra[2]; 

			auto cal_y = f0 * b * b + f1 * b +
				f2 * g * g + f3 * g +
				f4 * r * r + f5 * r +
				f6; 


            static uint64_t s_i_cout = 0; 
            if (5000 < s_i_cout && s_i_cout < 6000)
            {
                //cout << bgra << ",";
                //cout << cal_y << ", ";
            }
            s_i_cout++;


            y_val = cal_y; 

        }
    }



    // ??一???? 0-255 ??围??
    cv::Mat normalizedImage;
    cv::normalize(img_y, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    //cv::normalize(img_y, normalizedImage, 0, 255, cv::NORM_RELATIVE, CV_8U);

    // 转??为 uchar ???偷?图??
    cv::Mat img_yuc1;
    normalizedImage.convertTo(img_yuc1, CV_8UC1);


    ci.s_i(img_yuc1);


    cv::imwrite("d:/jd/t/git/dna-analysis/images/img/comp/new_y_test.jpg", img_yuc1);

#endif

#if 0
    string fn = "d:/jd/t/git/dna-analysis/images/img/comp/old_y.jpg";
    string fn_new = "d:/jd/t/git/dna-analysis/images/img/comp/old_rgb.jpg";

    auto x = 235; 
    auto y = 435; 



    // replace fn suffix with jpg
#endif

#if 0
    string fn = "d:/jd/t/3y.jpg";
    string fn_rgb = "d:/jd/t/3.jpg";
    ci_0.read_img(fn_rgb);
	ci.read_img(fn);
    ci.img; 

    


    for (int i = 0; i < ci.img.rows; i++)
    {
        for (int j = 0; j < ci.img.cols; j++)
        {
            auto& grey = ci.img.at<uchar>(i, j);


            if (grey > 8)
            {
                grey = 255;
            }
        }
    }

    ci.img; 
    ci_0.img; 

    ci.s_i();

#endif
#if 0
    string fn = "d:/jd/t/test_mask_72.jpg";

    ci.read_img(fn);

    ci.img; 



    





    auto & overlay_rect = ci.img;
    // loop ci.img
    const int cell_kernel_val = 128;


    // if overlay_rec pixel is not zero, then set to cell_kernel_val
	for (int i = 0; i < overlay_rect.rows; i++)
	{
		for (int j = 0; j < overlay_rect.cols; j++)
		{
            auto &grey = overlay_rect.at<uchar>(i, j);

			if (grey >100)
			{
				grey = cell_kernel_val;
			}
		}
	}




    cv::Mat overlay_rect_binary;
    cv::threshold(overlay_rect, overlay_rect_binary, 120, 255, cv::THRESH_BINARY);

    



    cv::Canny(overlay_rect_binary, overlay_rect_binary, 100, 200 );




    ci.s_i(overlay_rect_binary);



    //ci.s_i();

    //cout << ci.img << endl; 


    

 


    auto get_horizon = [](cv::Mat &  overlay_rect)
    {
            auto overlay_rect_cp = overlay_rect.clone();

            overlay_rect_cp.setTo(0);

            const int cell_kernel_val = 128;
            for (int i = 0; i < overlay_rect.rows; i++)
            {
                int sloc_non_zero = 0;
                int eloc_non_zero = 0;

                for (int j = 0; j < overlay_rect.cols; j++)
                {
                    if (sloc_non_zero == 0 && overlay_rect.at<uchar>(i, j) == cell_kernel_val)
                    {
                        sloc_non_zero = j;
                        continue;
                    }

                    if (sloc_non_zero != 0 && overlay_rect.at<uchar>(i, j) != cell_kernel_val)
                    {

                        eloc_non_zero = j - 1;
                        break;
                    }
                }

                if (eloc_non_zero == 0 || sloc_non_zero == 0)
                {

                }
                else
                {
                    //memset(overlay_rect.ptr<uchar>(i) + sloc_non_zero, 0, eloc_non_zero - sloc_non_zero + 1);

                    //uchar* ps = overlay_rect.ptr<uchar>(i) + sloc_non_zero;
                    //uchar* pe = overlay_rect.ptr<uchar>(i) + eloc_non_zero;
                    //*ps = cell_kernel_val;
                    //*pe = cell_kernel_val;

                    overlay_rect_cp.at<uchar>(i, sloc_non_zero) = cell_kernel_val;
                    overlay_rect_cp.at<uchar>(i, eloc_non_zero-1) = cell_kernel_val;



                }

            }

            return overlay_rect_cp;
    };




    auto get_vert = [](cv::Mat & overlay_rect)
    {

            cv::Mat overlay_rect_cp = overlay_rect.clone();
            overlay_rect_cp.setTo(0);

            const int cell_kernel_val = 128;
            for (int i = 0; i < overlay_rect.cols; i++)
            {
                int sloc_non_zero = 0;
                int eloc_non_zero = 0;

                for (int j = 0; j < overlay_rect.rows; j++)
                {
                    if (sloc_non_zero == 0 && overlay_rect.at<uchar>(j, i) == cell_kernel_val)
                    {
                        sloc_non_zero = j;
                        continue;
                    }

                    if (sloc_non_zero != 0 && overlay_rect.at<uchar>(j, i) != cell_kernel_val)
                    {

                        eloc_non_zero = j - 1;
                        break;
                    }
                }

                if (eloc_non_zero == 0 || sloc_non_zero == 0)
                {

                }
                else
                {
                    overlay_rect_cp.at<uchar>(sloc_non_zero, i) = cell_kernel_val;
                    overlay_rect_cp.at<uchar>(eloc_non_zero-1, i) = cell_kernel_val;
                }

            }

            return overlay_rect_cp;
    };



    
    
    auto overlay_rect_cp_h = get_horizon(overlay_rect);
    auto overlay_rect_cp_v = get_vert(overlay_rect);

    
    //all pixel value in overlay_rect_cp_h or overlay_rect_cp_v;

    auto overlay_rect_merge = overlay_rect_cp_h += overlay_rect_cp_v;

    // if pixel in overlay_rect_merge >= cell_kernel_val, then set to cell_kernel_val

    for (int i = 0; i < overlay_rect_merge.rows; i++)
    {
        for (int j = 0; j < overlay_rect_merge.cols; j++)
        {
            auto& grey = overlay_rect_merge.at<uchar>(i, j);
            if (grey >= cell_kernel_val)
            {
                grey = cell_kernel_val;
            }
        }
    }



    ci.img = overlay_rect_merge;


    


    ci.img;


    ci.s_i();



    




    

    // judge whether ci_d_img_cut is the same with ci_d_img_cut2


    // count the non-zero num in ci_d_img_cut

#endif
#if 0

    using cmap = unordered_map<int, vector<int>>; 

    cmap id_map{
        {1,{1,2,3,1}},
		{2,{1,2,3,0}},
        {3,{1,2,3,11}},
		{4,{1,2,3,0}},  
    };

    auto & ev = id_map[5];
    ev.resize(9);
    memset(id_map[5].data(), 0, sizeof(int) * 9);

   

    // print id_map

    // create a lambda function to print id_map

    auto print_id_map = [&id_map]() {

        for (auto& ep : id_map)
        {
            cout << ep.first << ": ";
            for (auto& ee : ep.second)
            {
                cout << ee << " ";
            }
            cout << endl;
        }
        };

    
    print_id_map();

    vector<int> keys_arr{};

	for (auto& ep : id_map)
	{
		keys_arr.push_back(ep.first);
	}

    for(auto & ek: keys_arr)
	{
	// if vector last element is 0, then erask map[ep.first]

		if (id_map[ek].back() == 0)
		{
			id_map.erase(ek);
		}

	}

    print_id_map();

#endif

#if 0
    struct a {
        int x;
        int y; 
        int z; 

    };
    class b {
    public:
		int x;
		int y;
		int z;
    
    };

    a id_a; 
    id_a.z = 99; 


    b id_b;
    id_b.z = 999;
    cout << id_a.z << " " << id_b.z << endl;

#endif
#if 0
    vector<obj_data> vec_obj_data = {
        {2.2,2},
        {2.3,3},
        {2.4,1},
                {2.4,1},
                        {2.4,1},
                        {3,1},
    };


    for (auto& e : vec_obj_data)
    {
        cout<< e.dnaindex << " " << e.celltype << endl;

    }

    hist_di(vec_obj_data);

#endif

#if 0

    std::vector<float> x_value = { 1.0, 2.0, 3.0, 4.0, 7.0 };
    std::vector<float> y_value = { 10.0, 20.0, 15.0, 30.0, 88.0 };

    tosvg("d:/jd/t/rgb_10_12.svg", x_value, y_value);

#endif
#if 0
    //string fn = "d:/jd/t/lena.bmp";

    string fn = "d:/jd/t/rgb_10_12.jpg";


    ci.read_img(fn);



   auto img_cut = ci.img(cv::Rect(88, 88, 512, 512));
   ci.s_i(img_cut);

    //ci.s_i();

 

    auto& image = img_cut;
    float custom_weights_bgr[] = { 0.1, 0.45, 0.45 };

    cv::Mat gray_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

    // 使???远?????权系??????色图??转??为?叶?图??
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            float gray_value = custom_weights_bgr[0] * image.at<cv::Vec3b>(y, x)[0] +
                custom_weights_bgr[1] * image.at<cv::Vec3b>(y, x)[1] +
                custom_weights_bgr[2] * image.at<cv::Vec3b>(y, x)[2];

            gray_image.at<uchar>(y, x) = static_cast<uchar>(gray_value);
        }
    }






    ci.s_i(gray_image);


    gray_image *= 1.333; 
    ci.s_i(gray_image);

#endif
#if 0
    string fn = "d:/jd/t/color_test.jpg";
    ci.read_img(fn);

	



auto result = get_overlap_img(ci.img, cv::Rect(250, 100, 288, 298)); 


int  t = 0;

#endif
#if 0

    string dirna="aaa";
    auto len = dirna.length();

    cout << len << endl; 
    assert(0 == 1);


    cv::Mat id_mat(3, 3, CV_8UC(1));
    vector<uint8_t> vi{0, 1,2,3,4,5,6,7,8 };

    std::copy(vi.begin(), vi.end(), id_mat.data);

    auto t = id_mat.ptr(1);

    cout << (int)(t[1]) << endl;

#endif

#if 0


    std::shared_ptr<int> make_pa = std::make_shared<int>(99);

	

    std::shared_ptr<int> m_a; 
    std::shared_ptr<int> tmp = std::shared_ptr<int>(new int(99));
    m_a = tmp; 
    

    int rows = 111;
    std::shared_ptr<uchar> p(new uchar[rows], std::default_delete<uchar[]>());
    //p[0] = 99;  // error

    *p = 99;  // ok
    uchar * c_p = p.get();

    p.get()[0] = 99;
    p.get()[1] = 99;
    c_p[2] = 111;

#endif
#if 0

    char* cic = (char*)&ci; 

    auto sz = sizeof(ci);

    string ci_str = ci.serial_p_2_str(&ci, sz); 
    ci.str_to_bin_file("1.bin", ci_str); 

    auto ci_str_ = ci.read_bin_to_string("1.bin");
    int cnt = 0;

    for (auto e : ci_str_)
    {
        assert(e == ci_str[cnt]);
        cnt++;
    }

#endif

#if 0
    //cout << std::fmax(9L, 9.11)<<endl; 

    const char* a = "ABCD"; 

    string va(a, a + 2); 
    cout << va << endl;

#endif
#if 0
    struct sA
    {
    
        int x; 
        int y; 
        
        sA()
        {
            x = 99;
            y = 999;
        }
    };

    sA id_sa;
    cout << id_sa.x << endl;

#endif
#if 0
    cv::Mat id_mat(3, 3, CV_8UC(1)); 
    vector<uint8_t> vi{ 1,2,3,4,5,6,7,8,9 }; 

    std::copy(vi.begin(), vi.end(), id_mat.data);

    uint8_t &a=  id_mat.at<uint8_t>(1, 1);
    uint8_t* pa = &a; 
    cout << (int)(pa[0]) << " " << (int)(pa[1]) << endl;

#endif
#if 0

    string  dir = "D:/jd/t/dl/?嵌?/g0/";

    //vector<string> v_rgb_y = { "b.bmp", "g.bmp", "r.bmp" };

    vector<string> v_rgb_y = { "rgb.bmp", "y.bmp"};



    vector<cv::Mat> v_img = {};

    for (auto& e : v_rgb_y)
    {
        e = dir + e;
        ci.read_img(e);
        v_img.push_back(ci.img);

        //ci.s_i();
    }
    

    //cv::merge(v_img, img_rgb);


    //ci.s_i(img_rgb);

    cv::imwrite(dir + "rgb.bmp", img_rgb);

#endif

#if 0

    string  dir = "D:/jd/t/dl/?嵌?/g0/";

    vector<string> v_rgb_y = { "b.bmp", "g.bmp", "r.bmp"};

    //vector<string> v_rgb_y = { "rgb.bmp", "y.bmp"};



    vector<cv::Mat> v_img = {}; 

    for (auto& e : v_rgb_y)
    {
        e = dir + e;
        ci.read_img(e); 
        v_img.push_back(ci.img);

        //ci.s_i();
    }
    cv::Mat img_rgb;

    cv::merge(v_img, img_rgb);


    //ci.s_i(img_rgb);

    cv::imwrite(dir + "rgb.bmp", img_rgb);

#endif
#if 0
    float fa[9] = { 2,3,5,7, 9,8,1,0, 88 }; 

    float& f1 = ret0(fa, 3);
    f1 = f1 * 88; 

    for (auto e : fa)
    {
        cout << e << endl; 
    }

#endif

#if 0
    cv::Rect2i rec(1, 2, 325, 77);
    cout << rec.x << endl;
    cout << rec.y << endl; 
    cout << rec.width << endl;
#endif

#if 0
    // at is faster than row.col

 

    int fpArray[3][3] = { {1,3,1},{3,9,3},{1,3,1} };
    cv::Mat kernel(3, 3, CV_32S, fpArray); // ??????转??为 OpenCV ?? Mat ??式

    cout << sizeof(float) << endl; 

    auto & e = kernel.at<float>(1, 2);
    e = e * 88; 


       


    uint32_t * pt = (uint32_t*) kernel.data;

    for(int i=0;i<9;i++)

    {
        cout << (float)pt[i] << endl;
    }




    for (int i = 0; i < kernel.rows; i++) {
        for (int j = 0; j < kernel.cols; j++) {
            int element = kernel.at<int32_t>(i, j);
            std::cout << "Kernel element at (" << i << ", " << j << "): " << element << std::endl;
        }
    }


    for (auto i = 0; i < kernel.rows; i++)
    {
        for (auto j = 0; j < kernel.cols; j++)
        {
            int * p = (int*) kernel.row(i).col(j).data;
            cout << *p << endl; 

        }
    }

#endif
#if 0

    //using cv_abs_fun = cv::abs; 

    int arr[2][3] = { {3,2,1},{33,22,11} };

    int* parr = &(arr[0][0]); 

    for (int i = 0; i < 2; i++)
        
    {
        auto cols = 3;
        for(int j=0;j<3;j++)
        cout << parr[i*cols+j] << endl; 
        
    }

#endif
#if 0



    string fn = "D:/jd/t/smb_share/t/img/t0/_rgb_edit.bmp";

    ci.read_img(fn);



    auto chn = ci.img.channels();
    for (auto r = 0; r < ci.img.rows; r++)
    {
        for (auto c = 0; c < ci.img.cols; c++)
        {



            if (22<r && r <ci.img.rows-22 &&  22<c && c< ci.img.cols-22)
            {
      
}
            else
            {
                uchar* ep = ci.img.col(c).row(r).data;
                if (chn == 1)
                {
                    ep[0] = 255;
                }
                else
                {
                    ep[0] = 255;
                    ep[1] = 255;
                    ep[2] = 255;
                }
                
            }

        }
    }




    //ci.s_i();

    auto ci_d_img_cut = cv::Mat(ci.img, { 0,2056 }, { 0,2464 });  //{row},{col}
   
    cv::imwrite(fn + ".bmp", ci_d_img_cut);

#endif

#if 0
    string fn_0 = "d:/jd/t/img_rgb_cmp/_042_back_w.bmp"; 
    string fn_1 = "d:/jd/t/img_rgb_cmp/_042_back_rgb.jpg";

    ci_0.read_img(fn_0);
    ci_1.read_img(fn_1); 

    cv::Mat  ci_0_img;
    ci_0.img.convertTo(ci_0_img, CV_16S); 
    cv::Mat  ci_1_img;
    ci_1.img.convertTo(ci_1_img, CV_16S);

    auto ci_d_img = ci_1_img - ci_0_img;

    auto ci_d_img_cut = cv::Mat(ci_d_img, { 1333,1355 }, { 1155,1277 });
    cout << ci_d_img_cut << endl;

#endif
#if 0
    string fn = "d:/jd/t/img_rgb_cmp/_007/_007_ok/d_rgb_w.jpg";
    ci.hist_img(fn);
#endif

#if 0
    string fn = "d:/jd/t/img_rgb_cmp/_007/_007_ok/d_rgb_w.jpg"; 
    ci.read_img(fn); 
    std::vector<cv::Mat> channels;
    cv::split(ci.img, channels);

    // ????每??通?赖?直??图
    std::vector<cv::Mat> hist(3);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    for (int i = 0; i < 3; i++) {
        cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist[i], 1, &histSize, &histRange);
    }

    // ????直??图
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::normalize(hist[0], hist[0], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist[1], hist[1], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist[2], hist[2], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[0].at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(hist[0].at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[1].at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(hist[1].at<float>(i))),
            cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[2].at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(hist[2].at<float>(i))),
            cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // ???雍???????
    cv::line(histImage, cv::Point(0, hist_h), cv::Point(hist_w, hist_h), cv::Scalar(0, 0, 0), 1, 8, 0);
    cv::line(histImage, cv::Point(0, hist_h), cv::Point(0, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
    for (int i = 0; i < histSize; i += 32) {
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, hist_h - 5), cv::Scalar(0, 0, 0), 1, 8, 0);
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
        std::stringstream ss;
        ss << i;
        cv::putText(histImage, ss.str(), cv::Point(bin_w * i, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // ????直??图图??
    cv::imwrite("histogram.png", histImage);


    cout << "ok" << endl;

#endif

#if 0
    vector<string> v_fn = { "d:/jd/t/img_rgb_cmp/_007/_007_ok/_007_rgb.jpg", "d:/jd/t/img_rgb_cmp/_007/_007_ok/_007_w.jpg" };

    ci_0.read_img(v_fn[0]); 
    ci_1.read_img(v_fn[1]);


    ci_0.s_i();
    ci_1.s_i();


    cv::Mat d_ci = ci_0.img - ci_1.img;

    d_ci = cv::abs(d_ci); 

    ci.s_i(d_ci); 


    

    cv::imwrite("d:/jd/t/img_rgb_cmp/_007/_007_ok/d_rgb_w.jpg",d_ci);

#endif

#if 0
    string fn = "D:/jd/t/w_2464_h_2056_chn_11_si_9_fx_0_fy_0_tm_20240507_190652.dat";

    ci.read_cmyimage_11chn_add_dna(fn);

#endif

#if 0

    auto fn_img = "D:/jd_d/jd/t/vs_p/t0/_3rd/opencv3.4.16/sources/doc/js_tutorials/js_assets/lena.jpg";

    ci.read_img(fn_img);
    

    int rows = ci.img.rows;
    int cols = ci.img.cols;
    int chn = ci.img.channels();
    assert(chn == 3); 

    int width_step = ci.img.step;

    for (auto r = 0; r < rows; r++)
    {
        auto* tr = ci.img.row(r).data;

        
        for (auto c = 0; c < cols; c++)
        {

            auto * ep = ci.img.row(r).col(c).data;
            ep = tr + c * chn; 

            auto& cb = ep[0]; 
            auto& cg = ep[1];
            auto& cr = ep[2];

  
            if (22<r && r <rows-22 && 22<c&&c<cols-22)
            {
                
            }
            else
            {
                //ci.img.at<Vec3b>(r, c) = Vec3b(0, 0, 0); // ????为????色
                //ep[0] = 0;
                //ep[1] = 0;
                //ep[2] = 0;

                cb = 0;
                cg = 0;
                cr = 0;
            }



        }
    }

    ci.s_i();


    cv::imwrite("1.jpg", ci.img);




    //cv::flip(ci.img,ci.img, 0);
    //ci.s_i();

#endif

#if 0

    string fn_img = "D:/jd/t/smb_share/t/img/_309/sz_ok/_309_y.bmp";
    ci.read_img(fn_img); 
    ci.s_i();

    cout << ci.img.channels() << endl;

#endif

#if 0
    ci.read_img("D:/jd/t/000000000724.jpg");
    int rows = ci.img.rows;
    int cols = ci.img.cols;
    cv::Mat img_small;
    cv::resize(ci.img, img_small, cv::Size(cols * 2.7, rows / 2.3));
    ci.img = img_small;
    ci.s_i();

#endif

#if 0
    BYTE img_b[1024];
    int cnt = 0;
    for (auto& e : img_b)
    {
        e = cnt % 255;
    }
    int rows = 5;
    int cols = 7;
    int chn = 3;
    write_byte_2_bin("bin_", img_b, rows, cols, chn);

#endif

#if 0


    vector<vector<string>> arr_fn =
    {

    {
        "D:/jd/t/smb_share/1200vs500/1B 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/1G 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/1R 1200.jpg"
    },

    {
        "D:/jd/t/smb_share/1200vs500/2B 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/2G 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/2R 1200.jpg"
    },

    {
        "D:/jd/t/smb_share/1200vs500/3B 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/3G 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/3R 1200.jpg"
    },

    {
        "D:/jd/t/smb_share/1200vs500/20x/AA B 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/20x/AA G 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/20x/AA R 1200.jpg",
    },

    {
        "D:/jd/t/smb_share/1200vs500/20x/BB B 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/20x/BB G 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/20x/BB R 1200.jpg",
    }


    };

    vector<string> arr_fn_save =
    {
        "D:/jd/t/smb_share/1200vs500/1_bgr_1200_merge.jpg",
        "D:/jd/t/smb_share/1200vs500/2_bgr_1200_merge.jpg",
        "D:/jd/t/smb_share/1200vs500/3_bgr_1200_merge.jpg",

        "D:/jd/t/smb_share/1200vs500/20x/aa_bgr_1200_merge.jpg",
        "D:/jd/t/smb_share/1200vs500/20x/bb_bgr_1200_merge.jpg",
    };

    for (int i = 3; i < arr_fn_save.size(); i++)
    {
        merge_bgr_to_color_batch(arr_fn[i], arr_fn_save[i]);
    }

#endif
#if 0

    vector<string> arr_fn =
    {
        "D:/jd/t/smb_share/1200vs500/1B 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/1G 1200.jpg",
        "D:/jd/t/smb_share/1200vs500/1R 1200.jpg"
    };

    //ci.s_i(ci.img);

    int cnt = 0;
    ci.read_img(arr_fn[cnt]);
    auto color_img = cv::Mat(ci.img.rows, ci.img.cols, CV_8UC(3));

    vector<cv::Mat> v_gray_img;

    cv::split(color_img, v_gray_img);

    cnt = 0;
    for (auto& e_img : v_gray_img)
    {
        ci.read_img(arr_fn[cnt]);
        e_img = ci.img.clone();
        cnt++;
    }




    cv::merge(v_gray_img, color_img);


    //cv::cvtColor(color_img, color_img, COLOR_BGR2RGB);
    ci.s_i(color_img);

    cv::imwrite("D:/jd/t/smb_share/1200vs500/1_bgr_1200_merge.jpg", color_img);

#endif

#if 0

    cout << s_("{}", 4) << endl;

    //ShellExecuteA(NULL, "open","notepad.exe", "NULL", NULL, SW_SHOW);

#endif

#if 0
#include <unordered_map>



    vector<int> nums = { 3,3 };
    int target = 6;

    vector<int> res{};
    unordered_map<int, int> um;
    int cnt = 0;
    for (auto e : nums)
    {
        um[e] = cnt;
        cnt++;
    }

    cout << s_("{}", um) << endl;

    cnt = 0;
    for (auto e : nums)
    {

        auto e_ = target - e;

        if (um.find(e_) != um.end() && cnt < um[e_])
        {
            // exists
            res.push_back(cnt);
            res.push_back(um[e_]);

        }
        cnt++;

    }

    cout << s_("{}", res) << endl;

#endif

#if 0

    std::function<int(string)> id_fun0;

    id_fun0 = [](string id_str)->int
        {
            cout << id_str.size() << endl;
            return 88;
        };

    unordered_map<string, std::function<int(string)>> u_o_m
    {
        {"i0", id_fun0},


        {
            "i1", [](string id_str) -> int
            {
                cout << id_str << id_str << endl;
                return 8;
            }
         },
        {"i2", id_fun1},

    };



    u_o_m["i0"]("i0");

    u_o_m["i1"]("i1");

    u_o_m["i2"]("i2");

#endif

#if 0

    vector<cv::Point2i> vpi = { {0,0}, {2,0}, {0,4},{-1,0} };
    cout << ci.perimeter_contour(vpi) << endl;;

    auto s = s_("{:#b}", 1024);

    cout << s << endl;

    cout << s, cout << s, cout << s << endl;

#endif
#if 0

    vector<pair<int, int>> vpi{};


    vpi = { {0,0}, {2,0}, {0,4},{-1,0} };

    if (vpi[0] != vpi[vpi.size() - 1])
    {
        vpi.push_back(vpi[0]);
    }



    auto start_p = vpi[0];
    auto p = start_p;



    unordered_map<string, int> d_cls{
        {"1,0", 0},
        {"1,1", 1},
        {"0,1", 2},
        {"-1,1", 3},
        {"-1,0", 4},
        {"-1,-1", 5},
        {"0,-1", 6},
        {"1,-1", 7},
        {"0,0", 8},
    };


    vector<int> vi{};
    for (int i = 1; i < vpi.size(); i++)
    {

        auto n = vpi[i];
        auto xd = n.first - p.first;
        auto yd = n.second - p.second;


        auto cnt_x = abs(xd) / 1;
        auto cnt_y = abs(yd) / 1;

        auto cnt_first = min(cnt_x, cnt_y);

        for (int j = 0; j < cnt_first; j++)
        {
            auto s = s_("{:d},{:d}", xd / cnt_x, yd / cnt_y);
            vi.push_back(d_cls[s]);
        }

        auto cnt_second = max(cnt_x, cnt_y);
        auto xd_off = 0;
        auto yd_off = 0;
        if (cnt_second == cnt_x)
        {
            yd_off = 0;
            xd_off = xd / cnt_x;
        }
        else
        {
            xd_off = 0;
            yd_off = yd / cnt_y;
        }

        for (int j = cnt_first; j < cnt_second; j++)
        {
            auto s = s_("{:d},{:d}", xd_off, yd_off);
            vi.push_back(d_cls[s]);
        }


        p = n;

    }

    cout << s_("{}", vi) << endl;

#endif
#if 0
    vector<tuple<int, float, string>> vif{};
    const int MAX_INT = 102;
    vif.resize(MAX_INT);

    int cnt = 0;
    for (auto& e : vif)
    {
        e = make_tuple(cnt, cnt * 0.1f, s_("cnt is {}", cnt));
        cnt++;
    }





    for (auto e : vif)
    {
        //cout << get<0>(e) << "," << get<1>(e) << "," << get<2>(e) << endl;
    }


    char* buf = (char*)(vif.data());


    string sbuf = string(buf, buf + vif.size() * (sizeof(int) + sizeof(float) + 128));




    ci.str_to_bin_file("1.bin", sbuf);

    auto sbuf2 = ci.read_bin_to_string("1.bin");

    vector<tuple<int, float, string>> vif2{};

    vif2.resize(MAX_INT);

    char* buf2 = (char*)vif2.data();




    //std::copy_n(sbuf2.data(), vif2.size() * (sizeof(int) + sizeof(float)+128), buf2);

    cnt = 0;
    for (auto e2 : vif2)
    {
        auto& e = vif[cnt];
        // cout << get<0>(e) << "," << get<1>(e) << "," << get<2>(e) << endl;
         //cout << get<0>(e2) << "," << get<1>(e2) << "," << get<2>(e2) << endl;

         //assert(e2 == e); 

        cnt++;
    }



    tuple<float, long int, string> t0 = make_tuple(1.23f, 34L, "abc");;

    tuple<float, long int, string> t1 = make_tuple(1.23f, 34L, "abc");;


    assert(t0 == t1);

#endif

#if 0
    string frombin = ci.read_bin_to_string("1.bin");

    vector<pair<int, int>> vi;
    vi.resize(3600);

    //std::copy_n((char*)frombin.c_str(), 3600 * sizeof(int) * 2, vi.data());

    cout << frombin.size() << endl;



    char* c = (char*)frombin.data();
    int* pint = (int*)c;

    int cnt = 0;
    while (cnt < 3600 / 2)
    {
        cout << pint[cnt] << "," << pint[cnt + 1] << "\|";
        //pint[cnt];
        cnt += 2;

    }

#endif

#if 0

    ci.read_img("d:\\jd\\t\\test_cell.png");
    ci.cvtcolor();
    cout << ci.info() << endl;

    int rows = ci.img.rows;
    int cols = ci.img.cols;


    int cen_r = 177;
    int cen_c = 196;


    for (int i = cen_r; i < cen_r + 4; i++)
    {
        auto* tr = ci.img.row(i).data;
        for (int j = cen_c; j < cen_c + 4; j++)
        {
            //tr[j] = 255;
        }
    }




    //ci.s_i();

    uchar cen_e = ci.img.row(cen_r).col(cen_c).data[0];

    uchar  cen_e_r0 = cen_e;

    vector<pair<int, int>> vi{};
    vector<pair<int, int>> vo{};

    for (int i = 0; i < 3600; i += 1)
    {
        double thet = 1.0 * i * 3.1415926 / 180.0;

        int r = 0;
        int cnt = 0;
        for (r; r < 300; r++)
        {
            auto circle_c_ = cen_c + r * std::cos(thet);
            auto circle_r_ = cen_r + r * std::sin(thet);
            uchar& circle_e_ = ci.img.row(circle_r_).col(circle_c_).data[0];

            if (abs(cen_e - circle_e_) > 20)
            {
                if (cnt == 0)
                {
                    vi.push_back({ circle_c_ , circle_r_ });
                    circle_e_ = 255;
                }

                cnt++;

                //circle_e_ = 255 / 2;
                r += 3;

                int circle_c_ = cen_c + r * std::cos(thet);
                int circle_r_ = cen_r + r * std::sin(thet);
                uchar& circle_e_ = ci.img.row(circle_r_).col(circle_c_).data[0];
                cen_e = circle_e_;
            }

            if (cnt == 2)
            {

                vo.push_back({ circle_c_ , circle_r_ });
                circle_e_ = 0;
                cen_e = cen_e_r0;
                break;
            }

        }

        if (i % 50 == 0)
        {
            // ci.s_i();
        }
    }


    //ci.s_i();

    cv::imwrite("d:/jd/t/test_cell_ok.png", ci.img);




    auto img = ci.img.clone();
    img.setTo(0);


    for (auto e : vi)
    {
        auto r_ = e.second;
        auto c_ = e.first;
        auto& el = img.row(r_).col(c_).data[0];
        el = 255;

    }

    cout << s_("size is {}", vi.size()) << endl;
    cout << s_("size is {}", vo.size()) << endl;


    for (auto e : vo)
    {
        auto r_ = e.second;
        auto c_ = e.first;
        auto& el = img.row(r_).col(c_).data[0];
        el = 111;
    }

    ci.s_i(img); // show img 




    cout << s_("{}", vi) << endl;

#if 0
    auto vibuff = vi.data();

    ofstream of_("1.bin");

    of_.write((char*)vibuff, vi.size() * sizeof(int) * 2);

    of_.close();
#endif

#endif

#if 0
    vector<int> vi{ 1,2,3,4 };
    cout << s_("{}", vi) << endl;

#endif

#if 0
    {
        cout << 1111 << endl;

        vector<int> vi{ 1,2,3,4,5,6,7,8 };

        fmt::print("{}\n", vi);
        fmt::print("{0},{2},{1}\n", "hello", string("world"), 1024);
        string s = fmt::format("{0},{2:08d},{1}\n", "hello", string("world"), 1024);
        cout << s << endl;

        string myname = "JD";
        using namespace fmt::literals;
        fmt::print("Hello, {name}! The answer is {number}. Goodbye, {name}.\n", "name"_a = myname, "number"_a = 42);

        char buf[1024] = { 0 };
        auto tt = fmt::format_to(buf, "{} {}\n", "hello world", 1024);
        cout << string(buf) << endl;

        fmt::text_style ts = fg(fmt::rgb(0, 200, 30));
        std::string out;
        fmt::format_to(std::back_inserter(out), ts, FMT_STRING("rgb(255,20,30){}{}{}"), 1, 2, 3);  // works on git bash

        //cout << out << endl; 


        std::vector<char> hello = { 'h', 'e', 'l', 'l', 'o' };
        s = fmt::format(FMT_STRING("{}"), hello); // 'h',...

        s = fmt::to_string(1.24f);

        s = fmt::format("{0:<10}___", "012");  // left align
        s = fmt::format("{0:^10}___", "012");  // center align
        s = fmt::format("{0:+}___", 12);  // show + or - , always 
        s = fmt::format("{0: }___", -12);  // show " " or - , always 
        s = fmt::format("{0:#b}___", -12);  // binary string "0b0101xxx", b,x,o,e
        s = fmt::format("{0:b}", 13);  // 
        s = fmt::format("{}", true);
        s = fmt::format("{{neverchange}}");

        uint64_t addr = 0x123456;
        s = fmt::format("{}", fmt::ptr((uint64_t*)addr));  //   addr to 0x 


        auto end = fmt::format_to(buf, "{}", "012");
        assert(end - buf == 3);


        std::string sb;
        fmt::format_to(std::back_inserter(sb), "part{}+", 1);
        fmt::format_to(std::back_inserter(sb), "part{}", 2);
        cout << sb << endl;

        s = fmt::to_string(fmt::join(vi.begin() + 2, vi.begin() + 6, "+"));
        s = fmt::format("{}", fmt::join(vi.begin(), vi.begin() + 7, "+"));

        fmt::ostream of_ = fmt::output_file("test-file");
        vector<string> vcontent = { "hello", "world","me" };
        for (auto e : vcontent)
        {
            of_.print("{}", e);
        }
        of_.flush();


        s = fmt::format("{}", (fmt::join(vi, "+")));

        int arr[2][3] = { {1,2,3},{4,5,6} };
        s = fmt::format("{}", arr);

        vector<vector<string>> vstr = { {"ab","cd","ef"},{"gh","gj"} };
        s = fmt::format("{:n:n}", vstr);  // "ab",..."gj"


        std::map<string, int> m_s_i{ {"k1",1}, {"k2",2} };
        s = fmt::format("{}", m_s_i);

        std::unordered_map<string, int> um_s_i{ {"k1_",1}, {"k2_",2} };
        s = fmt::format("{}", um_s_i);
        auto tp = std::tuple<int, float, std::string, char>(42, 1.5f, "this is tuple", 'i');
        s = fmt::format("{}", tp);

        auto bval = bitset<32>("0111111111111");
        bitset<32> bv("0111111111111");
        s = fmt::format("{}--{}", bv, bv[1]);

        cout << s << endl;

    }

#endif
#if 0
    bitset<8> a(42);
    cout << a << endl;
    bitset<8> b("01001");
    auto bs = b.to_string();
    auto blong = b.to_ullong();
    cout << blong << endl;
    cout << b[0] << endl;
#endif

#if 0
    vector<cv::Point2i> vp{ {0,0},{10,0},{0,5}, {0,2} ,{-2,0} };
    ci.area_contour(vp);
#endif
#if 0
    vector<vector<int>> vi{ {0,0},{10,0},{0,5}, {0,2} ,{-2,0} };
    int cnt = 0;
    float sum = 0;

    for (int i = 0; i < vi.size() - 1; i++)
    {
        auto x = vi[i][0];
        auto y = vi[i][1];
        auto xn = vi[i + 1][0];
        auto yn = vi[i + 1][1];
        auto e_val = (x * yn - xn * y);
        sum += e_val;
    }

    cout << sum * 1 / 2 << endl;
#endif

#if 0
    auto* s = "abcdefg";

    string ss(s, s + 3);
    cout << ss << endl;

    string sb(1024, 0);

    snprintf((char*)sb.data(), sb.size(), "%d,%d", 1222, 2344);

    cout << sb << endl;

#endif

#if 0
    vector<string> v_fn = { "./t0/overlay.bmp", "./t1/overlay.bmp" };
    vector<cv::Mat> vm;
    vm.resize(2);
    auto vm_(vm);

    int cnt = 0;
    int obj_plane = 1;
    int x, y, xs, ys, xe, ye, grey, k;
    k = 1;
    xs = 0;
    ys = 0;
    x = xs;
    y = ys;



    for (auto fn : v_fn)
    {
        ci.read_img(fn);
        vm[cnt] = ci.img.clone();
        xe = vm[cnt].cols;
        ye = vm[cnt].rows;





        threshold(vm[cnt], vm_[cnt], 8, 255, 0);


        cv::imwrite("./t" + to_string(cnt) + ".bmp", vm_[cnt]);

        cnt++;
    }





    assert(0 == 1);

#endif
#if 0
    //string fn = "H:\\tlq\\tmp_\\t3\\w_2464_h_2056_chn_11_si_8_fx_3_fy_3_tm_20231017_153232.dat";
    string fn = "H:\\tlq\\tmp_\\t3\\w_2464_h_2056_chn_11_si_8_fx_3_fy_3_tm_20231017_153233.dat";
    ci.read_cmyimage(fn);

#endif

#if 0

    //string fn =  "D:\\jd\\t\\platform_test_data\\t0_140\\w_2464_h_2056_chn_11_si_56_fx_7_fy_7_tm_20230808_113525.dat";
    string fn = "D:\\jd\\t\\w_2464_h_2056_chn_11_si_96_fx_4_fy_10_tm_20230912_141546.dat";
    //ci.read_cmyimage(fn);
    auto s = ci.get_timestamp();
    cout << s << endl;

#endif
#if 0

    ci.read_img(".\\rgbdata.jpg");


    ci.img;



    ci.s_i();


    cout << ci.img.channels() << endl;



    vector<cv::Mat> vm_;

    cv::split(ci.img, vm_);

    int cnt = 0;

    vector<string> bgr_fn{ "b.jpg", "g.jpg", "r.jpg" };
    for (auto e_fn : bgr_fn)
    {
        imwrite(e_fn, vm_[cnt]); cnt++;
    }








    assert(0 == 1);

#endif

#if 0
    string fn = "rgbdata.jpg";
    ci.read_img(fn);

    auto img_ = ci.img.clone();

    //ci.cvtcolor("RGB");

    vector<cv::Mat> vm_;
    cv::split(ci.img, vm_);


    auto start = 555;
    auto end = start + 140;
    cv::imwrite("b_old.jpg", vm_[0]({ start,end }, { start,end }));
    cv::imwrite("g_old.jpg", vm_[1]({ start,end }, { start,end }));
    cv::imwrite("r_old.jpg", vm_[2]({ start,end }, { start,end }));
    cv::imwrite("rgb_old.jpg", img_({ start,end }, { start,end }));






    assert(0 == 1);

#endif

#if 0



    string fn = dirname + "t0_133\\w_2464_h_2056_chn_11_si_8_fx_3_fy_4_tm_20230808_124322.dat";
    ci.read_cmyimage(fn);

#if 0
    int w = 2464;
    int h = 2056;
    int chn = 1;
    int& cols = w;
    int& rows = h;

    string bin_content = ci.read_bin_to_string(fn);
    uchar* imgdata = (uchar*)bin_content.data();
    auto img = cv::Mat(rows, cols, CV_8UC(3));


    std::copy_n(imgdata, rows * cols * 3, img.data);

    vector<cv::Mat> vm;
    vm.resize(5 + 3 - 3);

    int start_chn = 3;
    for (auto& em : vm)
    {
        em = cv::Mat(rows, cols, CV_8UC(1));
        std::copy_n(imgdata + rows * cols * start_chn, rows * cols * 1, em.data);
        start_chn++;
        //ci.s_i(em);
    }
    int idx = 0;
    auto& i3_ = vm[idx++];  // i3_ == dna_ image 
    auto& i4_ = vm[idx++];   // DAB image 

    auto& dna_ = vm[idx++];
    auto& overlay_ = vm[idx++];
    auto& contour_ = vm[idx++];


    for (auto em : vm)
    {
        ci.s_i(em);
    }
#endif

#endif

#if 0
    string fn = "D:\\jd\\t\\yh_rgb.jpg";
    ci.read_img(fn);
    ci.img;

    vector<cv::Mat> vm;

    cv::split(ci.img, vm);
    auto r = vm[0]({ 200,320 }, { 200,320 });
    auto g = vm[1]({ 200,320 }, { 200,320 });
    auto b = vm[2]({ 200,320 }, { 200,320 });

    cv::imwrite("r.jpg", r);
    cv::imwrite("g.jpg", g);

    cv::imwrite("b.jpg", b);

    cout << "- ok" << endl;
    assert(0 == 1);

#endif

#if 0

    string fn = "d:/jd/t/hand_bin.jpg";
    ci.read_img(fn);
    ci.cvtcolor("GRAY");

    //ci.s_i();


    cv::threshold(ci.img, ci.img, 100, 255, THRESH_BINARY);



    cv::Canny(ci.img, ci.img, 22, 55, 3);
    //ci.s_i();
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;

    //findContours(ci.img, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    cv::findContours(ci.img, contours0, RETR_EXTERNAL, CHAIN_APPROX_NONE);
#endif
#if 0
    for (auto& ep : contours0)
    {
        cout << ep << endl;
        cout << endl;
    }
#endif

#if 0
    auto img_cp = ci.img.clone();
    img_cp = img_cp / 255 * 33;

    //ci.s_i(img_cp);

    cout << contours0.size() << endl;
    int cnt = 1;
    for (auto& ep : contours0)
    {

        for (auto e : ep)
        {
            auto er = e.y;
            auto ec = e.x;

            img_cp.row(er).col(ec).data[0] = 66 + 80 * cnt;
        }
        cnt++;

    }

    //ci.s_i(img_cp);



    vector<Point> vp = {};

    for (auto& ep : contours0)
    {

        for (auto e : ep)
        {
            //auto er = e.y;
            //auto ec = e.x;

            //img_cp.row(er).col(ec).data[0] = 66 + 80 * cnt;
            vp.push_back(e);
        }


    }

    vector<int> hull;
    convexHull(vp, hull, true);

    for (auto e : hull)
    {
        cout << e << endl;
    }



    img_cp.setTo(0);

    for (auto ei : hull)
    {
        auto c = vp[ei].x;
        auto r = vp[ei].y;
        img_cp.row(r).col(c).data[0] = 255;
        //cout << ei << endl;
    }

    Mat img_ = img_cp * 0.7 + ci.img * 0.3;
    ci.s_i(img_);

#endif

#if 0
    ns0::id_mat;

#endif

#if 0
    string fn = "D:\\jd\\t\\big_merge.jpg";
    ci.read_img(fn);

    cout << ci.img.rows << endl;







    cv::resize(ci.img, ci.img, cvSize(ci.img.rows / 10, ci.img.cols / 10));


    cv::imwrite("D:\\jd\\t\\big_merge_another.png", ci.img);
    //ci.s_i();

    ci.img;
    assert(0 == 1);

#endif

#if 0

    string fn = "orig.jpg";

    string dirname_ = "D:\\jd\\t\\t0\\t1\\";
    fn = dirname_ + fn;

    ci.read_img(fn);

    ci.img;






    int rows = ci.img.rows;
    int cols = ci.img.cols;
    int div_n = 8;
    auto dr = rows / div_n;
    auto dc = cols / div_n;

    int cnt = 0;
    dirname_ = dirname_ + "f_1_0\\";

    for (auto r = 0; r < div_n; r++)
    {
        for (auto c = 0; c < div_n; c++)
        {
            Mat eimg_block = ci.img({ dr * r,dr * (r + 1) }, { dc * c,dc * (c + 1) });
            char buf[1024] = { 0 };
            snprintf(buf, 1024, "%d_%d.jpg", r, c);

            fn = dirname_ + buf;

            cv::imwrite(fn, eimg_block);



            cout << "r:" << r << ",c:" << c << ",cnt:" << cnt << endl;
            cnt++;
        }
    }






    assert(0 == 1);

#endif

#if 0
    ci.td_sleep(0.01);
#endif

#if 0
    float a = 9;
    float b = 3;
    float c = a / (0.1, (a == 9) && (b = 6), a);

    cout << b << endl;
#endif

#if 0
    auto img = cv::Mat::eye(4, 4, 0);
    auto m = cv::Mat(img);

    cv::flip(m, m, 0); // vert
    cout << m << endl;

#endif
#if 0
    auto rows = 2056;
    auto cols = 2464;


    string a = "aaaaaaaaaaaaaaaaaaaaaaa";
    string b(7, 'b');
    b[0] = 'A';

    uchar buf_array[1024] = "01234567890";

    std::copy_n(a.cbegin(), 7, b.begin());
    cout << b << endl;

    std::copy_n(buf_array, 2, b.begin());
    cout << b << endl;

    std::copy_n(b.cbegin() + b.size() - 4, 3, buf_array);
    cout << buf_array << endl;





    assert(0 == 1);







    auto v_c = ci.read_bin_to_string("d:/jd/t/1.bin");

    auto id_img = cv::Mat(rows, cols, CV_8UC(3));

    memcpy(id_img.data, v_c.data(), rows * cols * 3);

    //auto img_ = id_img({ 100,200 }, { 100,200 });

    cv::imwrite("d:/1.jpg", id_img);




    assert(0 == 1);

#endif

#if 0
    char buf[1024] = { 0 };

    _getcwd(buf, 1024);

    cout << buf << endl;

    auto perl_p = ci.get_env("perl_p_");

    if ("NULL" != perl_p)
    {
        cout << perl_p << endl;
    }
    else
    {
        cout << perl_p << endl;
    }

#endif

#if 0

    for (int i = 0; i < 6400; i++)
    {

        uchar* a = new uchar[2000 * 20000];
        a[2000 * 20000 - 1] = 'A' + i;
        delete a;


    }



    while (1)
    {

        cout << "- end" << endl;
    }

#endif
#if 0



    std::thread id_td0 = std::thread(

        [](cimg& ci) {

            ci.td_sleep(1.5);
            cout << 0 << endl;
        },

        std::ref(ci)
    );

    std::thread id_td1 = std::thread(

        [](cimg& ci) {

            ci.td_sleep(1.5);
            cout << 1 << endl;
        },

        std::ref(ci)
    );


    id_td0.join();
    id_td1.join();

#endif
#if 0
    cout << ci.get_env("VSAPPIDNAME") << endl;;



    int w = 40;
    auto ww = (w * 3 + 3) / 4 * 4;
    cout << ww << endl;

#endif
#if 0
    A* pa = A::getA();
    pa->func();
#endif

#if 0

    const int SZ_BUF = 1024;
    char buf[SZ_BUF];

    auto len_ = snprintf(buf, SZ_BUF, "%03d", 255);  // only 18
    assert(len_ < SZ_BUF); //len_ == 3
    cout << len_ << endl;
    cout << string(buf).size() << endl;
    cout << string(buf) << endl;

#endif

#if 0

    auto color_rgb = cv::imread("./rgb_range.jpg", IMREAD_COLOR);
    vector<cv::Mat> vi;
    cv::split(color_rgb, vi);


    bin_img(vi[0], 90);

    vi[0] = ~vi[0];

    vi[0];

    auto vi_0_copy = vi[0].clone();


    for (auto i = 1; i < vi[0].rows - 1; i++)
    {

        uchar* tr_b = vi[0].row(i - 1).data;
        uchar* tr_z = vi[0].row(i + 0).data;
        uchar* tr_p = vi[0].row(i + 1).data;

        for (auto j = 1; j < vi[0].cols - 1; j++)
        {
            tr_b[j - 1]; tr_b[j + 0]; tr_b[j + 1];
            tr_z[j - 1]; tr_z[j + 0]; tr_z[j + 1];
            tr_p[j - 1]; tr_p[j + 0]; tr_p[j + 1];

            int32_t sum_ = tr_b[j - 1] + tr_b[j + 0] + tr_b[j + 1] +
                tr_z[j - 1] + tr_z[j + 0] + tr_z[j + 1] +
                tr_p[j - 1] + tr_p[j + 0] + tr_p[j + 1];

            if (sum_ == 255 * 9)
            {
                vi_0_copy.row(i).col(j).data[0] = 0;
            }


        }
    }


    vi_0_copy;


    cv::imwrite("contour_white.jpg", vi_0_copy);


    assert(0 == 1);

#endif

#if 0
    ci.read_img("./bin_r.jpg");



    ci.img = ~ci.img;


    auto color_rgb = cv::imread("./rgb_range.jpg", IMREAD_COLOR);

    ci.s_i(color_rgb);

    //ci.s_i();



    auto img = ci.img.clone();


    img.setTo(0);



    cv::Canny(ci.img, ci.img, 127, 127 * 2, 3);


    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;
    cv::findContours(ci.img, contours0, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int which_draw = -1;
    //assert(which_draw < contours0.size()); 

    drawContours(color_rgb, contours0, which_draw, Scalar(255, 0, 0),
        1, LINE_8/*hierarchy, 1*/);

    ci.img;

    color_rgb;

    cv::imwrite("contour_range.jpg", color_rgb);


    assert(0 == 1);

    //ci.s_i();
#endif

#if 0
    RNG id_rng(time(NULL));
    auto img = cv::Mat(400, 400, 0);

    id_rng.fill(img, RNG::UNIFORM, 0, 255);

    //cout << CV_8UC3 << endl;
    //cout << img << endl;

    //ci.s_i(img);
    cv::Mat img_b;
    cv::copyMakeBorder(img, img_b, 10, 10, 10, 10, BORDER_CONSTANT, 0);  // size+=20 







    //ci.s_i(img_b);

#endif

#if 0

    ci.read_img("rgbdata.jpg");
    //ci.cvtcolor("RGB");


    auto rs = 666;
    auto cs = 236;
    auto sz = 140;
    auto img_ = ci.img({ rs,rs + sz }, { cs,cs + sz });



    //ci.s_i(img_);

    vector<cv::Mat> v_c;


    cv::split(img_, v_c);

    auto img_save_r = img_.clone();
    vector<cv::Mat> v_c_save_r;
    cv::split(img_save_r, v_c_save_r);

    v_c_save_r[1].setTo(0);

    v_c_save_r[2].setTo(0);

    cv::merge(v_c_save_r, img_save_r);

    cv::cvtColor(img_save_r, img_save_r, COLOR_BGR2RGB);







    cv::imwrite("save_r_only_range.jpg", img_save_r);




    cv::imwrite("r_range.jpg", v_c[0]);
    cv::imwrite("rgb_range.jpg", img_);


    bin_img(v_c[0], 110);


    cv::imwrite("bin_r.jpg", v_c[0]);



    assert(0 == 1);

#endif

#if 0
    RNG id_rng(time(NULL));

    vector<int> v_i(90, 0);
    for (auto& e : v_i)
    {
        e = id_rng.operator()(3); //0-1-2
        cout << e << endl;
    }




    //min(8, 9);
    //auto s = ci.get_env("VSAPPIDNAME");
    //cout << s << endl;

#endif
#if 0

    cv::Mat img(20, 20, CV_8UC(1), cv::Scalar(255));

    auto img_2 = img({ 11,13 }, { 12,20 });


    img_2.setTo(cv::Scalar(0));

    cout << img << endl;

    cout << img_2 << endl;

#endif
#if 0

    string ts = get_current_time();
    cout << ts.size();
    assert(0 == 1);
    return 0;

#endif

#if 0

    dirname = "D:\\jd\\t\\platform_test_data\\";

    string fn = dirname + "t0_133\\w_2464_h_2056_chn_11_si_8_fx_3_fy_4_tm_20230808_124322.dat";
    int w = 2464;
    int h = 2056;
    int chn = 1;
    int& cols = w;
    int& rows = h;

    int sz_img = w * h * chn;


    cout << fn << endl;
    string bin_content = ci.read_bin_to_string(fn);

    uchar* buf_img = (uchar*)bin_content.data();
    uchar* buf_img_r0 = buf_img;


    auto rgb = cv::Mat(rows, cols, CV_8UC(3));




    uchar* bgrdata = rgb.data;

    memcpy(bgrdata, buf_img_r0, sz_img * 3);


    ci.img = rgb;

    ci.cvtcolor("RGB");

    rgb = ci.img;




    vector<cv::Mat> v_rgb{};

    cv::split(rgb, v_rgb);

    auto& r_maybe = v_rgb[0];
    auto& g_maybe = v_rgb[1];
    auto& b_maybe = v_rgb[2];


    vector<cv::Mat> v_mat;
    v_mat.resize(5 + 3);

    for (auto& e_mat : v_mat)
    {
        e_mat = cv::Mat(rows, cols, CV_8UC(chn));
    }

    for (int split_n = 0; split_n < v_mat.size(); split_n++)
    {

        buf_img = buf_img_r0 + split_n * sz_img;

        for (int r = 0; r < rows; r++)
        {
            uchar* tr = v_mat[split_n].row(r).data;
            memcpy((char*)tr, buf_img, cols);
            buf_img += cols;
        }

    }

    int idx = 0;
    auto& b_ = v_mat[idx++]; // forsake 
    auto& g_ = v_mat[idx++]; // forsake 
    auto& r_ = v_mat[idx++]; // forsake 

    auto& i3_ = v_mat[idx++];  // i3_ == dna_ image 
    auto& i4_ = v_mat[idx++];   // DAB image 

    auto& dna_ = v_mat[idx++];
    auto& overlay_ = v_mat[idx++];
    auto& contour_ = v_mat[idx++];

#if 0
    cv::imwrite("rgbdata.jpg", rgb);

    cv::imwrite("b_maybe.jpg", b_maybe);
    cv::imwrite("g_maybe.jpg", g_maybe);
    cv::imwrite("r_maybe.jpg", r_maybe);

    cv::imwrite("i3_.jpg", i3_);
    cv::imwrite("i4_.jpg", i4_);
    cv::imwrite("dna_.jpg", dna_);
    cv::imwrite("overlay_.jpg", overlay_);
    cv::imwrite("contour_.jpg", contour_);
#endif 

    //assert("dna_" == "i3_");
    for (auto i : { 3,4,55,777 })
    {
        assert(dna_.row(i).col(i).data[0] == r_maybe.row(i).col(i).data[0]);
    }

    bin_img(r_maybe, 140);


    cout << "- end " << endl;

#endif

#if 0
    string fn = "rgbdata.jpg";

    ci.read_img(fn);

    ci.img;


    vector<cv::Mat> v_mat;
    cv::split(ci.img, v_mat);

    auto& r_ = v_mat[0];
    auto& b_ = v_mat[2];

    auto d_ = r_ - b_;

    d_, r_, b_;

    cv::Mat id_d = cv::Mat(d_);



    assert(0 == 1);

#endif

#if 0
    cimg ci;
    string id_s = "3:55:abc:def";

    auto vs = ci.split_str_2_vec(id_s, ':');
    for (auto e : vs)
    {
        cout << e << endl;

    }

#endif

#if 0
    char buf_res[1024] = { 0 };


    run_cmd((char*)"dir . /s /b", buf_res);

    cout << string(buf_res) << endl;
    assert(0 == 1);
    //string fn = "D:\\jd\\t\\lena128.bmp";

#endif

#if 0
    string fn = "D:\\ff.jpg";

    cimg ci;

    ci.read_img(fn);

    ci.img;

    vector<cv::Mat> v_mat;
    cv::split(ci.img, v_mat);


    int ys, ye, xs, xe;

    ys = 50;
    ye = 90;
    xs = 55;
    xe = 100;

    auto& img_r = v_mat[0];
    auto& img_g = v_mat[1];
    auto& img_b = v_mat[2];

    auto start_rows = ys;
    auto end_rows = ye;

    auto start_cols = xs;
    auto end_cols = xe;

    for (auto& e_chn : v_mat)
    {
        e_chn = ci.get_rectangle_mat(e_chn, start_rows, end_rows, start_cols, end_cols);
    }
    //auto img_r_part = ci.get_rectangle_mat(img_g, start_rows, end_rows, start_cols, end_cols);

    v_mat;

#endif

#if 0
    for (auto r = ys; r < ye; r++)
    {
        uchar* tr = v_mat[0].row(r).data;

    }

#endif

    // ci.cvtcolor("GRAY");

#if 0
    //cv::adaptiveThreshold(ci.img, ci.img, );

    auto img_cp = ci.img.clone();

    adaptiveThreshold(ci.img, img_cp, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, -2);


    ci.img = img_cp;

    ci.s_i();

#endif

#if 0

    string fn = "D:\\ff.jpg";

    cimg ci;

    ci.read_txt_to_img("d:\\1.txt");
    ci.img;


    ci.write_mat_to_txt("d:\\2.txt");

    ci.s_i();


    //ci.read_img(fn); 


    //ci.read_bin_to_mat("d:\\2.dat", 120, 120, 1);

    //ci.write_mat_to_txt("d:\\1.txt");

#endif

    // vector<Mat> _3chn_img;

    // cv::split(ci.img, _3chn_img);

    // cv::cvtColor(ci.img, ci.img, cv::COLOR_RGB2GRAY);

    // ci.img;
    // ci.s_i();

    // ci.si();

#if 0

    ci.read_bin_to_mat("d:\\2.dat", 120, 120, 1);
    ci.img;

    ci.write_mat_to_txt("d:\\1.txt");
#endif
    // ci.s_i();

#if 0
    char* buf = new char[1024];
    char* buf_r0 = buf;

    char bv[] = { 22,11,4,44, 1,2,3,4 };
    char* bv_ = bv;


    for (auto i = 0; i < 2; i++)
    {
        for (auto j = 0; j < 4 - 1; j++)
        {
            auto len_ = sprintf_s(buf, 1024, "%u,", *bv_);
            buf += len_;
            bv_++;
        }
        auto len_ = sprintf_s(buf, 1024, "%u\n", *bv_);
        buf += len_;


    }

    cout << buf_r0 << endl;

#endif

    // std::cout << "Hello World!\n";

    // system("pause");
}

#if 1

// scanapp_ cpp start
scanapp::scanapp()
{
}
// scanapp_ cpp end

// scandlg_ cpp start
scandlg::scandlg()
{
}
// scandlg_ cpp end

// stage_ cpp start
stage::stage()
{
}
stage *stage::get_stage()
{
    if (id_stage == nullptr)
    {
        init();
    }
    return id_stage;
}
void stage::init()
{
}
// stage_ cpp end

// measure_ cpp start
measure::measure()
{
}
// measure_ cpp end

// dbmgr_ cpp start
dbmgr::dbmgr()
{
}
// dbmgr_ cpp end
#endif
