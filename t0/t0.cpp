// #define _CRT_SECURE_NO_WARNINGS 1
#pragma once
#pragma execution_character_set("utf-8")
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
#include <nlohmann_json.h>

#include <map>
#include <unordered_map>
#include <bitset>
#include <functional>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath> 
#include <regex>
#include <rex.h>  // jd define regex
#include <random>
#include <unordered_set>
 
// #define PCRE2_CODE_UNIT_WIDTH 8
// #include <pcre2.h>   // .\vcpkg install pcre2:x64-windows && .\vcpkg integrate install 




#define byte uchar
#define WIDTH_ALIGN(_w, _align) \
    ((_w % _align != 0) ? (_w / _align + 1) * _align : _w)
#define WIDTH_4(_w) WIDTH_ALIGN(_w, 4)
#define DBG(x) cout << __FILE__ << ":" << __LINE__ << " " << #x << " = " << (x) << endl

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

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
using c_mss = unordered_map<string, string>;
using c_msi = unordered_map<string, int>;
using c_msf = unordered_map<string, float>;

constexpr const char* LR = "\n";
 
struct d_ts
{
    uint64_t t0;
    uint64_t t1; 
    int hour;
    int minute;
    int second;
    int ms;
    int d;
    string s_d;
    string ts0;
    string ts1;
};


class cimg
{

public:
    cv::Mat img;
    cv::Mat img_copy;

    std::vector<std::vector<cv::Point>> v_contours;
    cv::Mat img_contour_mask;
    cv::Mat img_contour_line;

    void norm_img();

    void read_img(string fn);
    void r_i(string fn);
	void w_i(const string& fn, const cv::Mat &img_);
    void w_i(const string& fn);
    void read_img_gif(const std::string& filePath, int frameIndex = 0);
    // void cvtcolor();
    void cvtcolor(string colormode = "GRAY");
    void s_i(cv::Mat &id_img);
    void s_i(); // show img on GUI
    string read_to_str(const string & fn);
    void read_bin_to_mat(string fn, int rows, int cols, int channels);
    void write_mat_to_bin(string fn);
    void write_mat_to_txt(string filename, int flag);
    void fcout(string fn, const string & id_s, string flag_w = "ios::out");   // str2txt str2bin can use
    void fcoutln(string fn, const string& id_s, string flag_w = "ios::out"); // str2txt str2bin can use
    void write_mat_to_csv(string fn);
    void str_to_bin_file(string fn, const string &str_to_serial);
    double round(double val, int precision=3);
	float round(float val, int precision=3);

    std::vector<std::string> vec_grep_v(std::vector<std::string>& vstr, const std::string& pattern);
    std::vector<std::string> vec_grep(std::vector<std::string>& vstr, const std::string& pattern);
    std::vector<std::string> vec_grep_v_empty(std::vector<std::string>& vstr);

    template<typename T> vector<vector<T>> vec_group(const vector<T> &v, size_t e_group_size);
    template<typename T> vector<T> vec_flatten(const vector<vector<T>> &vv);
    // declarations in class
    template<typename T>
    vector<vector<T>> vec_reshape(const vector<T>& v, int rows, int cols);
    cv::Mat mat_reshape(const cv::Mat& mat, int rows, int cols);

    template<typename T> vector<T> vec_rmdup(const vector<T> &v);


    // 1D vector ×ª Mat - ´øÀàÐÍ×ª»»
    template<typename TargetT, typename SourceT>
    cv::Mat vec_to_mat(const vector<SourceT>& v);

    // 1D vector ×ª Mat - ÎÞÀàÐÍ×ª»»£¨±ã½Ý°æ±¾£©
    template<typename T>
    cv::Mat vec_to_mat(const vector<T>& v);

    // 2D vector ×ª Mat - ´øÀàÐÍ×ª»»
    template<typename TargetT, typename SourceT>
    cv::Mat vec_to_mat(const vector<vector<SourceT>>& vva);

    // 2D vector ×ª Mat - ÎÞÀàÐÍ×ª»»£¨±ã½Ý°æ±¾£©
    template<typename T>
    cv::Mat vec_to_mat(const vector<vector<T>>& vva);

    template<typename T>
    vector<T> mat_to_vec_flatten(const cv::Mat& mat); 

    template<typename TargetT, typename SourceT>
        vector<TargetT> mat_to_vec_flatten(const cv::Mat& mat);

    // ¸¨Öúº¯Êý£º»ñÈ¡ OpenCV ÀàÐÍ
    template<typename T>
    int get_cv_type();




    // 2D vector - with type conversion
    template<typename TargetT, typename SourceT>
    vector<vector<TargetT>> mat_to_vec(const cv::Mat& mat);

    // 2D vector - convenient version (no type conversion)
    template<typename T>
    vector<vector<T>> mat_to_vec(const cv::Mat& mat);


    // find min/max value and index
    template<typename T> pair<T, size_t> vec_min(const vector<T> &v);
    template<typename T> pair<T, size_t> vec_max(const vector<T> &v);

    // reverse vector
    template<typename T> vector<T> vec_reverse(const vector<T> &v);

    // unique elements
    template<typename T> vector<T> vec_unique(const vector<T> &v);

    // slice vector (like python v[start:end])
    template<typename T> vector<T> vec_range(const vector<T> &v, int start, int end);
    template<typename T>
    vector<T> vec_range(const vector<T>& v, int start);

    // sort and return sorted copy
    template<typename T> vector<T> vec_sort(const vector<T> &v, bool descending = false);

    // filter by lambda
    template<typename T, typename Func> 
        vector<T> vec_grep(const vector<T> &v, Func pred);

    template<typename T, typename Func>
        void vec_map(vector<T> &v, Func fn);


    //template <typename T> std::vector<T> vec_combine(const std::vector<T> &v0, const std::vector<T> &v1);
    // declaration in class
template<typename T, typename...  Args>
vector<T> vec_combine(const vector<T> &v0, const Args&... args);
    template <typename T> T vec_sum(const vector<T> &v);
    template <typename T> float vec_mean(const vector<T> &v);
    template <typename T> float vec_stddev(const vector<T> &v);
    template <typename T> std::vector<T> vec_smooth(const vector<T>& v, int win_sz);
    template<typename T> std::vector<T> vec_norm_by_minmax(const vector<T>& v);
    template <typename T> vector<T> vec_diff(const vector<T> &v);

    void serial_to_mat(const string &fn);
    void deserial_from_mat(const string &fn);
    string info();

    vector<string> split(string& str, const string& delimiter);
    string join(const vector<string> &vstr, const string & delimiter); 

    // The Base64 character set
    const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string b64(const std::string& in);
    std::string b64_d(const std::string& in);
    bool is_b64(unsigned char c);

    void read_txt_to_img(string fn, int flag_has_header = 1);
    vector<string> read_to_vec(string fn);
    vector<string> read_all(const string & fn);
    void chomp(string& str);
    void chomp(vector<string>& vec);

    // trim whitespace from both ends
    string trim(const string &str);

    // check if string starts with prefix
    bool startwith(const string &src, const string &prefix);
    bool contains(const string &str, const string &sub);
    template<typename T> bool vec_match(const vector<T> &v, const T &val);
    bool vec_match(const vector<string> &v, const rex_match  &e_match_);
    template<typename T>
    bool vec_match(const vector<T>& v0, const vector<T>& v1);
    // repeat string n times
    string str_repeat(const string &str, int n);
template<typename T> vector<T> vec_repeat(const vector<T> &v, int n);
    

    // pad string to width (left/right)
    string pad_left(const string &str, size_t width, char ch = ' ');
    string pad_right(const string &str, size_t width, char ch = ' ');

    void read_cmyimage_11chn_add_dna(string fn);
    cv::Mat get_rectangle_mat(cv::Mat &id_m32, int start_rows, int end_rows, int start_cols, int end_cols);
    void read_cmyimage(string fn);
    string get_timestamp(const time_t & now_);
    string get_timestamp();

    // format current time
    string now(const string &fmt = "%Y%m%d_%H%M%S");

    // parse time string to time_t
    time_t str2time(const string &str, const string &fmt = "%Y%m%d_%H%M%S");

    string run_cmd_simple(string cmd_);
    std::unordered_map<std::string, std::string> run_cmd(const std::string & cmd);   // ans has key: code, stdout, stderr, err
    
    string get_env(string env_name);
    void td_sleep(double seconds);
    double area_contour(vector<cv::Point2i> &vp);
    cv::Mat hconcat(const vector<cv::Mat> &vimg, int intv = 0);
    vector<cv::Mat> unify_vimg(const vector<cv::Mat> & vimg);
    cv::Mat vconcat(const vector<cv::Mat> &vimg, int intv = 0);
    double perimeter_contour(vector<cv::Point2i> &vp);
    cv::Mat hist_img(const cv::Mat & img_);
    void threshold(int thres, int dst_val = 255);
    std::vector<cv::Point> smooth_contour(const std::vector<cv::Point>& inputContour, int intensity = 3, bool closed = true);
    cv::Mat find_contours(int dst_val = 255, int dst_sz = 300);


    template <typename T>
    cv::Mat P(const vector<T> &data, int flag_show = 1);

    template<typename T>
    cv::Mat plt_scatter(const vector<T>& y_data, int flag_show = 1,
        int marker_size = 4, cv::Scalar color = cv::Scalar(0, 0, 255),
       const string& title = "",
        const string& xlabel = "x", const string& ylabel = "y");

    template<typename T>
    cv::Mat plt_scatter(const vector<T>& x_data, const vector<T>& y_data, int flag_show = 1,
        int marker_size = 4, cv::Scalar color = cv::Scalar(0, 0, 255),
         const string& title = "",
        const string& xlabel = "x", const string& ylabel = "y");

    void resize(float scale_f);
    void puttext(const string &text, cv::Scalar color = cv::Scalar(0, 0, 0));
    void puttext(cv::Mat &img, const string &text, cv::Scalar color = cv::Scalar(0, 0, 0));
    cv::Mat create_img_rc_chn(int rows, int cols, int chn);
    std::vector<double> linspace(double start, double stop, int num);

    // clamp value
    template<typename T> T clamp(T val, T min_val, T max_val);


    // map value from one range to another
    float map_range(float val, float in_min, float in_max, float out_min, float out_max);

    std::string replace_str(const std::string& str, const std::string& from_str, const std::string& to_str, const std::string& opt="");
    
    std::vector<std::string> get_folder_content(const string& dir_path, const string& pattern=".*", const string& type = "all", bool recursive = false);
    
    vector<int> R(int starti, int endi, int step = 1);
    vector<int> R(int endi);

    string norm_path(const std::string& path);
    string abs_path(const std::string & path);
    bool endwith(const std::string & src, const std::string & suffix); 
    string toupper(const string & str);
    string tolower(const string & str);
    cv::Mat pad2square(const cv::Mat & img_, const cv::Scalar& pad_val = cv::Scalar(217,217,217));
    cv::Mat reverse_pad2square(const cv::Mat & img_square, const cv::Size & original_size);

    bool is_contour_same(const vector<cv::Point>& contour1, const vector<cv::Point>& contour2);
    cv::Rect expand_rect(const cv::Rect& rect, float expand_ratio, const cv::Mat& img);
    cv::Mat img2hsv(const cv::Mat &inputImage);
    void byte2cvmat(byte *src, int rows, int cols, int chn, cv::Mat &id_mat);
    d_ts ts0();
    d_ts ts1(d_ts & d_ts_t0);

	int difftime(const time_t& t1, const time_t& t0);
    int difftime(const time_t& t1, const time_t& t0, string& timestring);

    bool fs_e(const std::string & fn);
    bool fs_f(const std:: string & fn);
    bool fs_d(const std::string & fn);
    bool fs_rm(const std::string & fn);
    bool fs_touch(const std::string & fn);
    std::unordered_map<std::string, std::string> fs_stat(const std::string & fn);
    uintmax_t fs_du(const std::string & fn);
    bool fs_mv(const std::string & from, const std:: string & to);
    
    std::string ts2str(const std::filesystem::file_time_type & ftime);  // time_t to string
    bool fs_cp(const std:: string & from, const std::string & to);

    // get filename without path
    string fs_basename(const string &path);

    // get directory name
    string fs_dirname(const string &path);

    // get file extension
    string fs_ext(const string &path);

    // change extension
    string fs_chext(const string &path, const string &new_ext);

    // create directory recursively
    bool fs_mkdir(const string &path);  // recursive mkdir

    // read/write json (if you use nlohmann/json)
    // void fs_write_json(const string &fn, const json &j);
    // json fs_read_json(const string &fn);
    nlohmann::json fs_read_json(const string &fn);
    void fs_write_json(const string &fn, const nlohmann::json &j, int indent = 4);

    std::vector<std::string> fs_ls(const std::string & dir_path, const std::string & pattern = "");
    std::vector<std::string> fs_ls_r(const std::string & dir_path, const std::string & pattern = "");

    std::unordered_map<std::string, std::string> parse_args(int argc, char** argv, const unordered_map<string, string>& arg_map_default = {}, const string& help_msg_ = "");

    template <typename T> string serial_struct_2_str(T &pt);
    template<typename T> T str_to_struct(const std::string & bin_str);
    template <typename T> string serial_p_2_str(T *cstr, size_t sz);


    string vec_to_table(const vector<vector<string>> &cols);
    template<typename T> vector<string> vec_estr(const vector<T> &v, int precision = -1);
    template<typename... Args> string vec_to_table_templ(const Args&... args);
    template<typename T> vector<T> vec_rotate(const vector<T> &v, int n);  // left shift n element to tail
    template<typename T1, typename T0>
    auto vec_to_type(const vector<T0>& v) -> vector<T1>;

    // random int in range [min, max]
    int rand_int(int min_val, int max_val);

    // random float in range [0, 1]
    float rand_float();

    // random choice from vector
    template<typename T>
        T rand_pick(const vector<T> &v);

        template<typename T>
        std::vector<std::vector<T>> vec_t(const std::vector<std::vector<T>>& mat);

        // shuffle vector
    template<typename T>
        vector<T> vec_shuffle(const vector<T> &v);

    // random string
    string rand_str(size_t len, const string &charset = "abcdefghijklmnopqrstuvwxyz0123456789");

    string str_reverse(const string &str);
    // declaration
    float lerp(float a, float b, float lambda_);


    template<typename K, typename V> vector<K> map_keys(const unordered_map<K, V> &m);
    template<typename K, typename V> vector<V> map_values(const unordered_map<K, V> &m);
    template<typename K, typename V> bool map_has(const unordered_map<K, V> &m, const K &key);
    template<typename K, typename V> V map_get(const unordered_map<K, V> &m, const K &key, const V &default_val);

};

// implementations

// template_ start

template<typename T>
vector<T> cimg::vec_repeat(const vector<T> &v, int n)
{
    if (n <= 0 || v.empty())
    {
        return {};
    }
    
    vector<T> ret;
    ret.reserve(v.size() * n);
    
    for (int i = 0; i < n; ++i)
    {
        ret.insert(ret.end(), v.begin(), v.end());
    }
    
    return ret;
}

template<typename T>
vector<T> cimg::vec_rmdup(const vector<T> &v)
{
    vector<T> ret;
    unordered_set<T> seen;
    for (const auto &e : v)
    {
        if (seen.find(e) == seen.end())
        {
            seen.insert(e);
            ret.push_back(e);
        }
    }
    return ret;
}

// Helper function to get cv_type from template type
template<typename T>
int get_cv_type() {
    if (std::is_same<T, uchar>::value) return CV_8UC1;
    else if (std::is_same<T, char>::value) return CV_8SC1;
    else if (std::is_same<T, ushort>::value) return CV_16UC1;
    else if (std::is_same<T, short>::value) return CV_16SC1;
    else if (std::is_same<T, int>::value) return CV_32SC1;
    else if (std::is_same<T, float>::value) return CV_32FC1;
    else if (std::is_same<T, double>::value) return CV_64FC1;
    else throw std::runtime_error("Unsupported type");
}

// Helper function to get cv_type from template type
template<typename T>
int cimg::get_cv_type() {
    if (std::is_same<T, uchar>::value) return CV_8UC1;
    else if (std::is_same<T, char>::value) return CV_8SC1;
    else if (std::is_same<T, ushort>::value) return CV_16UC1;
    else if (std::is_same<T, short>::value) return CV_16SC1;
    else if (std::is_same<T, int>::value) return CV_32SC1;
    else if (std::is_same<T, float>::value) return CV_32FC1;
    else if (std::is_same<T, double>::value) return CV_64FC1;
    else throw std::runtime_error("Unsupported type");
}

// 1D vector - Ê¹ÓÃ OpenCV ÄÚÖÃº¯Êý + convertTo
template<typename TargetT, typename SourceT>
cv::Mat cimg::vec_to_mat(const vector<SourceT>& v) {
    if (v.empty()) {
        return cv::Mat();
    }

    // ´´½¨Ô´ÀàÐÍµÄ Mat (ÐÐÏòÁ¿)
    cv::Mat src(1, v.size(), get_cv_type<SourceT>(), (void*)v.data());

    // Èç¹ûÔ´ÀàÐÍºÍÄ¿±êÀàÐÍÏàÍ¬£¬Ö±½Ó¿ËÂ¡·µ»Ø
    if (std::is_same<TargetT, SourceT>::value) {
        return src.clone();
    }

    // Ê¹ÓÃ OpenCV µÄ convertTo ½øÐÐÀàÐÍ×ª»»
    cv::Mat dst;
    src.convertTo(dst, get_cv_type<TargetT>());

    return dst;
}

// 1D vector - ±ã½Ý°æ±¾£¨Ô´ºÍÄ¿±êÀàÐÍÏàÍ¬£©
template<typename T>
cv::Mat cimg::vec_to_mat(const vector<T>& v) {
    if (v.empty()) {
        return cv::Mat();
    }
    // Ê¹ÓÃ OpenCV Mat ¹¹Ôìº¯Êý£¬È»ºó clone ÒÔÓµÓÐÊý¾Ý
    cv::Mat mat(1, v.size(), get_cv_type<T>(), (void*)v.data());
    return mat.clone();
}

// 2D vector - Ê¹ÓÃ OpenCV ÄÚÖÃº¯Êý + convertTo
template<typename TargetT, typename SourceT>
cv::Mat cimg::vec_to_mat(const vector<vector<SourceT>>& vva) {
    if (vva.empty() || vva[0].empty()) {
        return cv::Mat();
    }

    int rows = vva.size();
    int cols = vva[0].size();

    // ´´½¨Ô´ÀàÐÍµÄ Mat
    cv::Mat src(rows, cols, get_cv_type<SourceT>());

    // ¸´ÖÆÊý¾Ýµ½ Mat
    for (int i = 0; i < rows; i++) {
        if (vva[i].size() != cols) {
            throw std::runtime_error("All rows must have the same size");
        }
        memcpy(src.ptr<SourceT>(i), vva[i].data(), cols * sizeof(SourceT));
    }

    // Èç¹ûÔ´ÀàÐÍºÍÄ¿±êÀàÐÍÏàÍ¬£¬Ö±½Ó·µ»Ø
    if (std::is_same<TargetT, SourceT>::value) {
        return src;
    }

    // Ê¹ÓÃ OpenCV µÄ convertTo ½øÐÐÀàÐÍ×ª»»
    cv::Mat dst;
    src.convertTo(dst, get_cv_type<TargetT>());

    return dst;
}

// 2D vector - ±ã½Ý°æ±¾£¨Ô´ºÍÄ¿±êÀàÐÍÏàÍ¬£©
template<typename T>
cv::Mat cimg::vec_to_mat(const vector<vector<T>>& vva) {
    if (vva.empty() || vva[0].empty()) {
        return cv::Mat();
    }

    int rows = vva.size();
    int cols = vva[0].size();

    // ´´½¨ Mat
    cv::Mat mat(rows, cols, get_cv_type<T>());

    // ¸´ÖÆÊý¾Ý
    for (int i = 0; i < rows; i++) {
        if (vva[i].size() != cols) {
            throw std::runtime_error("All rows must have the same size");
        }
        memcpy(mat.ptr<T>(i), vva[i].data(), cols * sizeof(T));
    }

    return mat;
}


cv::Mat cimg::mat_reshape(const cv::Mat& mat, int rows, int cols)
{
    // È·±£ Mat ÊÇÁ¬ÐøµÄ
    if (!mat.isContinuous())
    {
        throw runtime_error("mat_reshape: Mat must be continuous");
    }

    int total = mat.total() * mat.channels();

    // handle -1 dimension
    if (rows == -1 && cols == -1)
    {
        throw runtime_error("mat_reshape: only one dimension can be -1");
    }
    else if (rows == -1)
    {
        if (total % cols != 0)
        {
            throw runtime_error("mat_reshape: cannot reshape, total not divisible by cols");
        }
        rows = total / cols;
    }
    else if (cols == -1)
    {
        if (total % rows != 0)
        {
            throw runtime_error("mat_reshape: cannot reshape, total not divisible by rows");
        }
        cols = total / rows;
    }

    // check size match
    if (rows * cols != total)
    {
        throw runtime_error("mat_reshape: size mismatch, " +
            to_string(total) + " != " +
            to_string(rows) + " * " + to_string(cols));
    }

    // Ê¹ÓÃ OpenCV µÄ reshape£¬Í¨µÀÊýÉèÎª1£¬È»ºóÖ¸¶¨ÐÐÊý
    return mat.reshape(1, rows);
}


// 2D vector - with type conversion
template<typename TargetT, typename SourceT>
vector<vector<TargetT>> cimg::mat_to_vec(const cv::Mat& mat) 
{
    if (mat.empty()) {
        return vector<vector<TargetT>>();
    }

    // Ensure it's single channel
    if (mat.channels() != 1) {
        throw std::runtime_error("mat_to_vec2d only supports single-channel Mat");
    }

    int rows = mat.rows;
    int cols = mat.cols;

    // If types are the same, direct copy
    if (std::is_same<TargetT, SourceT>::value && mat.type() == get_cv_type<SourceT>()) {
        vector<vector<TargetT>> result(rows, vector<TargetT>(cols));
        for (int i = 0; i < rows; i++) {
            memcpy(result[i].data(), mat.ptr<SourceT>(i), cols * sizeof(SourceT));
        }
        return result;
    }

    // Type conversion needed
    cv::Mat converted;
    mat.convertTo(converted, get_cv_type<TargetT>());

    vector<vector<TargetT>> result(rows, vector<TargetT>(cols));
    for (int i = 0; i < rows; i++) {
        memcpy(result[i].data(), converted.ptr<TargetT>(i), cols * sizeof(TargetT));
    }
    return result;
}

// 1D vector - with type conversion
template<typename TargetT, typename SourceT>
vector<TargetT> cimg::mat_to_vec_flatten(const cv::Mat& mat) 
{
    if (mat.empty()) {
        return vector<TargetT>();
    }

    // Ensure it's single channel
    if (mat.channels() != 1) {
        throw std::runtime_error("mat_to_vec only supports single-channel Mat");
    }

    // Convert to 1D if needed (flatten)
    cv::Mat flat = mat.reshape(1, 1);  // Reshape to 1 row

    // If types are the same, direct copy
    if (std::is_same<TargetT, SourceT>::value && flat.type() == get_cv_type<SourceT>()) {
        vector<TargetT> result(flat.cols);
        memcpy(result.data(), flat.ptr<SourceT>(0), flat.cols * sizeof(SourceT));
        return result;
    }

    // Type conversion needed
    cv::Mat converted;
    flat.convertTo(converted, get_cv_type<TargetT>());

    vector<TargetT> result(converted.cols);
    memcpy(result.data(), converted.ptr<TargetT>(0), converted.cols * sizeof(TargetT));
    return result;
}

// 1D vector - convenient version (no type conversion)
template<typename T>
vector<T> cimg::mat_to_vec_flatten(const cv::Mat& mat) {
    if (mat.empty()) {
        return vector<T>();
    }

    // Ensure it's single channel
    if (mat.channels() != 1) {
        throw std::runtime_error("mat_to_vec only supports single-channel Mat");
    }

    // Convert to 1D if needed (flatten)
    cv::Mat flat = mat.reshape(1, 1);  // Reshape to 1 row

    // Verify the type matches
    if (flat.type() != get_cv_type<T>()) {
        throw std::runtime_error("Mat type does not match template type T");
    }

    vector<T> result(flat.cols);
    memcpy(result.data(), flat.ptr<T>(0), flat.cols * sizeof(T));
    return result;
}


// 2D vector - convenient version (no type conversion)
template<typename T>
vector<vector<T>> cimg::mat_to_vec(const cv::Mat& mat) {
    if (mat.empty()) {
        return vector<vector<T>>();
    }

    // Ensure it's single channel
    if (mat.channels() != 1) {
        throw std::runtime_error("mat_to_vec2d only supports single-channel Mat");
    }

    // Verify the type matches
    if (mat.type() != get_cv_type<T>()) {
        throw std::runtime_error("Mat type does not match template type T");
    }

    int rows = mat.rows;
    int cols = mat.cols;

    vector<vector<T>> result(rows, vector<T>(cols));
    for (int i = 0; i < rows; i++) {
        memcpy(result[i].data(), mat.ptr<T>(i), cols * sizeof(T));
    }
    return result;
}



template<typename T>
vector<vector<T>> cimg::vec_reshape(const vector<T>& v, int rows, int cols)
{
    int size = (int)v.size();

    // handle -1 dimension
    if (rows == -1 && cols == -1)
    {
        throw runtime_error("reshape:  only one dimension can be -1");
    }
    else if (rows == -1)
    {
        if (size % cols != 0)
        {
            throw runtime_error("reshape:  cannot reshape, size not divisible by cols");
        }
        rows = size / cols;
    }
    else if (cols == -1)
    {
        if (size % rows != 0)
        {
            throw runtime_error("reshape: cannot reshape, size not divisible by rows");
        }
        cols = size / rows;
    }

    // check size match
    if (rows * cols != size)
    {
        throw runtime_error("reshape: size mismatch, " + to_string(size) + " != " + to_string(rows) + " * " + to_string(cols));
    }

    vector<vector<T>> result(rows, vector<T>(cols));

    for (int i = 0; i < size; ++i)
    {
        result[i / cols][i % cols] = v[i];
    }

    return result;
}

template<typename T>
vector<T> cimg::vec_flatten(const vector<vector<T>> &vv)
{
    vector<T> ret;
    size_t total = 0;
    for (const auto &v : vv)
    {
        total += v.size();
    }
    ret.reserve(total);
    for (const auto &v : vv)
    {
        ret.insert(ret.end(), v.begin(), v.end());
    }
    return ret;
}

template<typename T>
vector<vector<T>> cimg::vec_group(const vector<T> &v, size_t e_group_size)
{
    vector<vector<T>> ret;
    if (e_group_size == 0)
    {
        return ret;
    }
    for (size_t i = 0; i < v.size(); i += e_group_size)
    {
        size_t end = min(i + e_group_size, v.size());
        ret.push_back(vector<T>(v.begin() + i, v.begin() + end));
    }
    return ret;
}

template<typename T, typename... Args>
vector<T> cimg::vec_combine(const vector<T> &v0, const Args&... args)
{
    vector<T> combined;
    
    // calculate total size
    size_t total_size = v0.size();
    ((total_size += args. size()), ...);
    
    combined.reserve(total_size);
    
    // insert first vector
    combined.insert(combined.end(), v0.begin(), v0.end());
    
    // insert remaining vectors using fold expression
    ((combined.insert(combined.end(), args.begin(), args.end())), ...);
    
    return combined;
}

template <typename T>
T cimg::vec_sum(const vector<T>& v)
{

    T sum = 0;
    for (auto& e : v)
    {
        sum += e;
    }

    return sum;
}

template <typename T>
vector<T> cimg::vec_diff(const vector<T>& v)
{
    assert(v.size() >= 2);
    std::vector<T> diff_p(v.size());
    std::adjacent_difference(v.begin(), v.end(), diff_p.begin());

    return vector<T>(diff_p.begin() + 1, diff_p.end());
}

template <typename T>
float cimg::vec_stddev(const vector<T>& v)
{
    float sumOfSquares = 0.0f;
    auto mean = vec_mean(v);

    for (float num : v)
    {
        sumOfSquares += (num - mean) * (num - mean);
    }

    return std::sqrt(sumOfSquares / v.size());
}

template <typename T>
float cimg::vec_mean(const vector<T>& v)
{

    float sum = 0.0f;

    sum += vec_sum(v);

    return sum * 1.0f / v.size();
}

template<typename T>
T cimg::clamp(T val, T min_val, T max_val)
{
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

template<typename T>
T cimg::str_to_struct(const std::string& bin_str)
{
    T result;
    if (bin_str.size() >= sizeof(T))
    {
        std::memcpy(&result, bin_str.data(), sizeof(T));
    }
    else
    {
        std::memset(&result, 0, sizeof(T));
    }
    return result;
}

template <typename T>
string cimg::serial_struct_2_str(T& pt)
{
    string id_s = "";
    char* c_pt = (char*)&pt;
    id_s += string(c_pt, c_pt + sizeof(T));
    return id_s;
}

template <typename T>
string cimg::serial_p_2_str(T* cstr, size_t sz)
{
    char* cstr_ = (char*)cstr;
    return string(cstr_, cstr_ + sz);
}

template<typename T>
vector<string> cimg::vec_estr(const vector<T>& v, int precision)
{
    vector<string> ret;
    ret.reserve(v.size());
    for (const auto& e : v)
    {
        if constexpr (is_same_v<T, string>)
        {
            ret.push_back(e);
        }
        else if constexpr (is_floating_point_v<T>)
        {
            ostringstream oss;
            if (precision >= 0)
            {
                oss << fixed << setprecision(precision);
            }
            oss << e;
            ret.push_back(oss.str());
        }
        else if constexpr (is_integral_v<T>)
        {
            ret.push_back(to_string(e));
        }
    }
    return ret;
}


template<typename...  Args>
string cimg::vec_to_table_templ(const Args&... args)
{
    vector<vector<string>> cols = { vec_estr(args)... };
    //vector<vector<string>> cols = {vec_estr(args, 3)...};
    return vec_to_table(cols);
}

template<typename T>
pair<T, size_t> cimg::vec_min(const vector<T>& v)
{
    // auto [val, idx] = ci.vec_min(vi);
    assert(!v.empty());
    auto it = min_element(v.begin(), v.end());
    return { *it, distance(v.begin(), it) };

}

template<typename T>
pair<T, size_t> cimg::vec_max(const vector<T>& v)
{
    assert(!v.empty());
    auto it = max_element(v.begin(), v.end());
    return { *it, distance(v.begin(), it) };
}

template<typename T>
vector<T> cimg::vec_reverse(const vector<T>& v)
{
    return vector<T>(v.rbegin(), v.rend());
}

// implementation
template<typename T, typename Func>
vector<T> cimg::vec_grep(const vector<T> &v, Func pred)
{
    // ci.vec_grep(nums, [](int x) { return x % 2 == 0; });
    vector<T> ret;
    for (const auto &e : v)
    {
        if (pred(e))
        {
            ret.push_back(e);
        }
    }
    return ret;
}

template<typename T, typename Func>
void cimg::vec_map(vector<T> &v, Func fn)
{
    for (auto& e : v)
    {
        fn(e);
    }
}

template<typename T>
vector<T> cimg::vec_unique(const vector<T>& v)
{
    vector<T> ret = v;
    sort(ret.begin(), ret.end());
    ret.erase(unique(ret.begin(), ret.end()), ret.end());
    return ret;
}

template<typename T>
vector<T> cimg::vec_range(const vector<T>& v, int start, int end)
{
    if (start < 0) start = v.size() + start;
    if (end < 0) end = v.size() + end;
    if (start < 0) start = 0;
    if (end > (int)v.size()) end = v.size();
    if (start >= end) return {};
    return vector<T>(v.begin() + start, v.begin() + end);
}


template<typename T>
vector<T> cimg::vec_sort(const vector<T> &v, bool ascending)
{
    vector<T> ret = v;
    if (ascending)
    {
        sort(ret.begin(), ret.end());
    }
    else
    {
        sort(ret.begin(), ret.end(), greater<T>());
    }
    return ret;
}

template<typename T>
vector<T> cimg::vec_range(const vector<T>& v, int start)
{
    return vec_range(v, start, (int)v.size());
}

template<typename K, typename V>
vector<K> cimg::map_keys(const unordered_map<K, V> &m)
{
    vector<K> ret;
    ret.reserve(m.size());
    for (const auto &p :  m)
    {
        ret.push_back(p.first);
    }
    return ret;
}

template<typename K, typename V>
vector<V> cimg::map_values(const unordered_map<K, V> &m)
{
    vector<V> ret;
    ret.reserve(m. size());
    for (const auto &p : m)
    {
        ret.push_back(p.second);
    }
    return ret;
}

template<typename K, typename V>
bool cimg::map_has(const unordered_map<K, V> &m, const K &key)
{
    // auto b0 = ci.map_has(mm, string("a"));
    return m.find(key) != m.end();
}
template<typename T1, typename T0>
auto cimg::vec_to_type(const vector<T0>& v) -> vector<T1>
{
    vector<T1> ret;
    ret.reserve(v.size());
    for (const auto& e : v)
    {
        ret.push_back(static_cast<T1>(e));
    }
    return ret;
}

template<typename T>
vector<T> cimg::vec_rotate(const vector<T> &v, int n)
{
    if (v.empty())
    {
        return v;
    }
    vector<T> ret = v;
    int sz = static_cast<int>(ret.size());
    n = ((n % sz) + sz) % sz;
    rotate(ret.begin(), ret.begin() + n, ret.end());
    return ret;
}

template<typename K, typename V>
V cimg::map_get(const unordered_map<K, V> &m, const K &key, const V &default_val)
{
    auto it = m.find(key);
    if (it == m.end())
    {
        return default_val;
    }
    return it->second;
}


int cimg::rand_int(int min_val, int max_val)
{
    // right max_val cannot be selected [min_val~max_val)
    
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(min_val, max_val - 1);
    return dist(gen);
}

float cimg::rand_float()
{
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

template<typename T>
T cimg::rand_pick(const vector<T> &v)
{
    assert(!v.empty());
    return v[rand_int(0, static_cast<int>(v.size()) - 1)];
}


template<typename T>
std::vector<std::vector<T>> cimg::vec_t(const std::vector<std::vector<T>>& mat) {
    if (mat.empty() || mat.front().empty()) {
        return {};
    }

    std::size_t rows = mat.size();
    std::size_t cols = mat[0].size();

    std::vector<std::vector<T>> result(cols, std::vector<T>(rows));

    for (std::size_t j = 0; j < cols; ++j) {
        auto& dest_col = result[j];
        for (std::size_t i = 0; i < rows; ++i) {
            dest_col[i] = mat[i][j];
        }
    }

    return result;
}
template<typename T>
vector<T> cimg::vec_shuffle(const vector<T> &v)
{
    vector<T> ret = v;
    static std::mt19937 gen(std::random_device{}());
    shuffle(ret.begin(), ret.end(), gen);
    return ret;
}

// template_ end


// implementation
float cimg::lerp(float a, float b, float lambda_)
{
    return a + lambda_ * (b - a);
}

string cimg::str_reverse(const string &str)
{
    return string(str.rbegin(), str.rend());
}

string cimg::rand_str(size_t len, const string &charset)
{
    string ret;
    ret.reserve(len);
    for (size_t i = 0; i < len; ++i)
    {
        ret += charset[rand_int(0, static_cast<int>(charset.size()) - 1)];
    }
    return ret;
}

string cimg::str_repeat(const string &str, int n)
{
    string ret;
    ret.reserve(str.size() * n);
    for (int i = 0; i < n; ++i)
    {
        ret += str;
    }
    return ret;
}

string cimg::pad_left(const string &str, size_t width, char ch)
{
    if (str.size() >= width)
    {
        return str;
    }
    return string(width - str.size(), ch) + str;
}

string cimg::pad_right(const string &str, size_t width, char ch)
{
    if (str.size() >= width)
    {
        return str;
    }
    return str + string(width - str.size(), ch);
}


string cimg::fs_basename(const string &path)
{
    return fs:: path(path).filename().string();
}

string cimg::fs_dirname(const string &path)
{
    return fs::path(path).parent_path().string();
}

string cimg::fs_ext(const string &path)
{
    return fs::path(path).extension().string();
}

string cimg::fs_chext(const string &path, const string &new_ext)
{
    fs::path p(path);
    return (p.parent_path() / (p.stem().string() + new_ext)).string();
}

bool cimg::fs_mkdir(const string &path)
{
    try
    {
        return fs::create_directories(path);
    }
    catch (...)
    {
        return false;
    }
}

string cimg::now(const string &fmt)
{
    time_t t = time(nullptr);
    tm local_tm;
    localtime_s(&local_tm, &t);
    char buf[128] = {0};
    strftime(buf, sizeof(buf), fmt.c_str(), &local_tm);
    return string(buf);
}



string cimg::vec_to_table(const vector<vector<string>> &cols)
{

    // for(auto i:ci.R(vstr.size()))
    // {
    //     cout_("{:<18}{:<18}{:<18}\n", vstr[i], vstr_1[i], vstr_2[i]);
    // }

    if (cols.empty())
    {
        return string();
    }

    size_t max_rows = 0;
    for (const auto &col : cols)
    {
        if (col. size() > max_rows)
        {
            max_rows = col.size();
        }
    }

    vector<size_t> col_widths;
    for (const auto &col :  cols)
    {
        size_t max_w = 0;
        for (const auto &s : col)
        {
            if (s.length() > max_w)
            {
                max_w = s.length();
            }
        }
        col_widths.push_back(max_w + 2);
    }

    string fmt;
    for (size_t c = 0; c < cols.size(); ++c)
    {
        fmt += "{:<" + to_string(col_widths[c]) + "}";
    }

    vector<string> vstr_ans;
    for (size_t row = 0; row < max_rows; ++row)
    {
        vector<string> row_vals;
        for (size_t c = 0; c < cols.size(); ++c)
        {
            string val = (row < cols[c]. size()) ? cols[c][row] : "";
            row_vals.push_back(val);
        }

        string es;
        switch (cols.size())
        {
            case 1:
                es = s_(fmt, row_vals[0]);
                break;
            case 2:
                es = s_(fmt, row_vals[0], row_vals[1]);
                break;
            case 3:
                es = s_(fmt, row_vals[0], row_vals[1], row_vals[2]);
                break;
            case 4:
                es = s_(fmt, row_vals[0], row_vals[1], row_vals[2], row_vals[3]);
                break;
            case 5:
                es = s_(fmt, row_vals[0], row_vals[1], row_vals[2], row_vals[3], row_vals[4]);
                break;
            case 6:
                es = s_(fmt, row_vals[0], row_vals[1], row_vals[2], row_vals[3], row_vals[4], row_vals[5]);
                break;
        }
        vstr_ans.push_back(es);
    }

    return join(vstr_ans, "\n");
}



// cimg_ hpp end


// cimg_ cpp start
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

// implementations

string cimg::trim(const string &str)
{
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == string::npos)
    {
        return "";
    }
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}


bool cimg::vec_match(const vector<string> &v, const rex_match  &e_match_)
{
    for(auto & e: v)
    {
        if (rex(e) % e_match_)
        {
            return true;
        }
    }

    return false;
}


template<typename T>
bool cimg::vec_match(const vector<T> &v, const T &val)
{
    return find(v.begin(), v.end(), val) != v.end();
}

// check if two vectors are equal
template<typename T>
bool cimg::vec_match(const vector<T>& v0, const vector<T>& v1)
{
    return v0 == v1;
}


bool cimg::contains(const string &str, const string &sub)
{
    return str.find(sub) != string::npos;
}

bool cimg::startwith(const string &src, const string &prefix)
{
    if (src.length() < prefix.length())
    {
        return false;
    }
    return src.compare(0, prefix.length(), prefix) == 0;
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
float cimg::map_range(float val, float in_min, float in_max, float out_min, float out_max)
{
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

std::string cimg::replace_str(const std::string& str, const std::string& from_str, const std::string& to_str, const std::string& opt)
{
    if (from_str.empty())
    {
        return str;
    }

    bool case_insensitive = (opt.find('i') != std::string::npos);
    bool global_replace = (opt.find('g') != std::string::npos);

    std::string result = str;
    std::string search_str = result;
    std::string from_lower = from_str;

    // prepare lowercase versions for case-insensitive search
    if (case_insensitive)
    {
        std::transform(search_str.begin(), search_str.end(), search_str.begin(), ::tolower);
        std::transform(from_lower.begin(), from_lower.end(), from_lower.begin(), ::tolower);
    }

    size_t pos = 0;
    while ((pos = search_str.find(from_lower, pos)) != std::string::npos)
    {
        result.replace(pos, from_str.size(), to_str);

        if (!global_replace)
        {
            break;
        }

        // update search_str after replacement
        if (case_insensitive)
        {
            search_str = result;
            std::transform(search_str.begin(), search_str.end(), search_str.begin(), ::tolower);
        }
        else
        {
            search_str = result;
        }

        pos += to_str.size();
    }

    return result;
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
    img_.setTo(0);
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


template <typename T>
cv::Mat cimg::plt_scatter(const vector<T>& x_data, const vector<T>& y_data, int flag_show,
    int marker_size, cv::Scalar color,
     const string& title,
    const string& xlabel , const string& ylabel) {
    if (x_data.empty() || y_data.empty())
    {
        cout << "Data vectors are empty!" << endl;
        return cv::Mat();
    }

    if (x_data.size() != y_data.size())
    {
        cout << "X and Y data vectors must have the same size!" << endl;
        return cv::Mat();
    }

    // Calculate data ranges
    auto x_max = *max_element(x_data.begin(), x_data.end());
    auto x_min = *min_element(x_data.begin(), x_data.end());
    auto y_max = *max_element(y_data.begin(), y_data.end());
    auto y_min = *min_element(y_data.begin(), y_data.end());

    double x_range = x_max - x_min;
    double y_range = y_max - y_min;

    if (x_range == 0) x_range = 1;
    if (y_range == 0) y_range = 1;

    // Canvas settings
    int width = 900;
    int height = 600;
    int margin_left = 80;
    int margin_right = 50;
    int margin_top = title.empty() ? 50 : 80;
    int margin_bottom = xlabel.empty() ? 60 : 80;

    int plot_width = width - margin_left - margin_right;
    int plot_height = height - margin_top - margin_bottom;

    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255)); // White background

    // Helper lambda for coordinate transformation
    auto transform_x = [&](T val) -> int {
        return margin_left + static_cast<int>(((val - x_min) / x_range) * plot_width);
    };

    auto transform_y = [&](T val) -> int {
        return height - margin_bottom - static_cast<int>(((val - y_min) / y_range) * plot_height);
    };

    // Draw axes
    line(image, Point(margin_left, height - margin_bottom),
        Point(margin_left, margin_top), Scalar(0, 0, 0), 2); // y axis
    line(image, Point(margin_left, height - margin_bottom),
        Point(width - margin_right, height - margin_bottom), Scalar(0, 0, 0), 2); // x axis

    // Helper for formatting tick labels
    auto format_label = [](double val) -> string {
        char buf[64] = { 0 };
        if (abs(val) < 0.01 || abs(val) > 10000) {
            sprintf_s(buf, sizeof(buf), "%.1e", val);
        }
        else if (abs(val - std::round(val)) < 0.01) {
            sprintf_s(buf, sizeof(buf), "%.0f", val);
        }
        else {
            sprintf_s(buf, sizeof(buf), "%.2f", val);
        }
        return string(buf);
    };

    // Draw y-axis ticks and grid
    int num_y_ticks = 8;
    for (int i = 0; i <= num_y_ticks; i++)
    {
        double val = y_min + i * (y_range / num_y_ticks);
        int y = transform_y(val);

        // Tick mark
        line(image, Point(margin_left - 5, y), Point(margin_left + 5, y), Scalar(0, 0, 0), 1);

        // Grid line
        line(image, Point(margin_left, y), Point(width - margin_right, y),
            Scalar(220, 220, 220), 1, cv::LINE_AA);

        // Label
        string label = format_label(val);
        putText(image, label, Point(margin_left - 70, y + 5),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    // Draw x-axis ticks and grid (adaptive)
    int dataSize = static_cast<int>(x_data.size());
    int num_x_ticks = 8;

    // Calculate adaptive tick interval for x-axis
    double x_tick_interval = x_range / num_x_ticks;

    for (int i = 0; i <= num_x_ticks; i++)
    {
        double val = x_min + i * x_tick_interval;
        int x = transform_x(val);

        // Tick mark
        line(image, Point(x, height - margin_bottom - 5),
            Point(x, height - margin_bottom + 5), Scalar(0, 0, 0), 1);

        // Grid line
        line(image, Point(x, margin_top), Point(x, height - margin_bottom),
            Scalar(220, 220, 220), 1, cv::LINE_AA);

        // Label
        string label = format_label(val);
        int label_width = label.length() * 6;
        putText(image, label, Point(x - label_width / 2, height - margin_bottom + 25),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 0), 1);
    }

    // Draw scatter points
    for (size_t i = 0; i < x_data.size(); i++)
    {
        int x = transform_x(x_data[i]);
        int y = transform_y(y_data[i]);

        // Draw filled circle
        circle(image, Point(x, y), marker_size, color, -1, cv::LINE_AA);

        // Optional: add edge for better visibility
        circle(image, Point(x, y), marker_size, Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // Add title
    if (!title.empty())
    {
        putText(image, title, Point(width / 2 - title.length() * 5, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
    }

    // Add x-axis label
    if (!xlabel.empty())
    {
        putText(image, xlabel, Point(width / 2 - xlabel.length() * 4, height - 10),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    }

    // Add y-axis label (rotated text simulation with vertical text)
    if (!ylabel.empty())
    {
        putText(image, ylabel, Point(10, height / 2 + ylabel.length() * 4),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    }

    // Show image
    if (flag_show)
    {
        s_i(image);
    }

    return image;
}


// Overload for single vector (y-data only, x = indices)
template <typename T>
cv::Mat cimg::plt_scatter(const vector<T>& y_data, int flag_show,
    int marker_size, cv::Scalar color,
     const string& title,
    const string& xlabel, const string& ylabel)
{
    vector<T> x_data(y_data.size());
    for (size_t i = 0; i < y_data.size(); i++)
    {
        x_data[i] = static_cast<T>(i);
    }

    return plt_scatter(x_data, y_data, marker_size, color, flag_show, title, xlabel, ylabel);
}



template <typename T>
cv::Mat cimg::P(const vector<T>& data, int flag_show)
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

    // Draw y-axis ticks (fixed at 10 intervals - this is usually fine)
    for (int i = 0; i <= 10; i++)
    {
        int y = height - 50 - (i * (height - 100) / 10);
        line(image, Point(45, y), Point(55, y), Scalar(0, 0, 0), 1);
        putText(image, to_string(static_cast<int>(minValue + i * (range / 10))),
            Point(10, y + 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Adaptive x-axis ticks
    int dataSize = static_cast<int>(data.size());

    // Calculate adaptive tick interval
    int numDesiredTicks = 10; // Target number of ticks
    int tickInterval;

    if (dataSize <= 20) {
        tickInterval = 1; // Show every point for small datasets
    }
    else if (dataSize <= 100) {
        tickInterval = (dataSize + numDesiredTicks - 1) / numDesiredTicks; // Round up division
    }
    else if (dataSize <= 1000) {
        // Use round numbers (10, 20, 50, 100, etc.)
        int roughInterval = dataSize / numDesiredTicks;
        if (roughInterval < 10) tickInterval = 10;
        else if (roughInterval < 20) tickInterval = 20;
        else if (roughInterval < 50) tickInterval = 50;
        else if (roughInterval < 100) tickInterval = 100;
        else tickInterval = ((roughInterval + 99) / 100) * 100; // Round to nearest 100
    }
    else {
        // For very large datasets, use powers of 10
        int roughInterval = dataSize / numDesiredTicks;
        int magnitude = static_cast<int>(pow(10, floor(log10(roughInterval))));
        tickInterval = ((roughInterval + magnitude - 1) / magnitude) * magnitude;
    }

    // Draw x-axis ticks with adaptive interval
    for (int i = 0; i < dataSize; i += tickInterval)
    {
        int x = 50 + (i * (width - 100) / (dataSize - 1));
        line(image, Point(x, height - 45), Point(x, height - 55), Scalar(0, 0, 0), 1);
        putText(image, to_string(i), Point(x - 10, height - 30),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Draw data points and lines
    double prevX = 50;
    double prevY = height - 50 - ((static_cast<double>(data[0]) - minValue) / range * (height - 100));

    // Adaptive point drawing - skip points if too many
    int pointInterval = max(1, dataSize / 500); // Draw at most 500 points

    for (int i = 0; i < dataSize; i++)
    {
        double normalizedValue = (static_cast<double>(data[i]) - minValue) / range;
        double x = 50 + (i * (width - 100) / (dataSize - 1));
        double y = height - 50 - (normalizedValue * (height - 100));

        // Draw point only at intervals for large datasets
        if (i % pointInterval == 0 || i == dataSize - 1)
        {
            circle(image, Point(x, y), 5, Scalar(0, 0, 255), -1);
        }

        // Always draw line segments for continuity
        if (i > 0)
        {
            line(image, Point(prevX, prevY), Point(x, y), Scalar(255, 0, 0), 2);
        }

        prevX = x;
        prevY = y;
    }

    auto tostring_02f = [](float f)
    {
        char buf[1024] = { 0 };
        sprintf_s(buf, "%0.2f", std::round(f * 100) / 100.0f);
        string sb(buf);
        cimg ci;
        sb = ci.replace_str(sb, ".00", "");
        return sb;
    };

    // Label max value
    int maxIndex = distance(data.begin(), max_element(data.begin(), data.end()));
    int maxX = 50 + (maxIndex * (width - 100) / (dataSize - 1));
    int maxY = height - 50 - ((maxValue - minValue) / range * (height - 100));
    putText(image, "v:" + tostring_02f(maxValue) + ",i:" + to_string(maxIndex),
        Point(maxX + 10, maxY - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    // Label min value
    int minIndex = distance(data.begin(), min_element(data.begin(), data.end()));
    int minX = 50 + (minIndex * (width - 100) / (dataSize - 1));
    int minY = height - 50;
    putText(image, "v:" + tostring_02f(minValue) + ",i:" + to_string(minIndex),
        Point(minX + 10, minY - 15), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    // Show image
    if (flag_show)
    {
        s_i(image);
    }

    return image;
}


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

std::vector<cv::Point> cimg::smooth_contour(const std::vector<cv::Point>& inputContour, int intensity, bool closed)
{
    int n = inputContour.size();
    if (n < 3 || intensity <= 0) return inputContour;

    std::vector<cv::Point> smoothedContour(n);

    // ´°¿Ú´óÐ¡ÓÉÁ¦¶È¾ö¶¨£¬°ë¾¶ = intensity
    int radius = intensity;

    for (int i = 0; i < n; i++) {
        double sumX = 0;
        double sumY = 0;
        double totalWeight = 0;

        for (int j = -radius; j <= radius; j++) {
            int idx;
            if (closed) {
                // ±ÕºÏÂÖÀª£ºË÷ÒýÑ­»·»·ÈÆ
                idx = (i + j + n) % n;
            }
            else {
                // ·Ç±ÕºÏÂÖÀª£º±ß½ç½Ø¶Ï
                idx = i + j;
                if (idx < 0) idx = 0;
                if (idx >= n) idx = n - 1;
            }

            // ¸ßË¹È¨ÖØ£º¾àÀëµ±Ç°µãÔ½½ü£¬È¨ÖØÔ½¸ß
            // ÕâÀïµÄÈ¨ÖØ¼ò»¯Îª£º1 / (1 + distance^2) 
            // »òÕß¼òµ¥Ê¹ÓÃ¾ùÖµÆ½»¬£¨È¨ÖØÈ«Îª1£©
            double weight = 1.0;

            sumX += inputContour[idx].x * weight;
            sumY += inputContour[idx].y * weight;
            totalWeight += weight;
        }

        smoothedContour[i].x = cvRound(sumX / totalWeight);
        smoothedContour[i].y = cvRound(sumY / totalWeight);
    }

    return smoothedContour;
}

cv::Mat cimg::hist_img(const cv::Mat & img_)
{
    std::vector<cv:: Mat> channels;
    cv:: split(img_, channels);

    std::vector<cv::Mat> hist(3);
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    
    for (int i = 0; i < 3; i++)
    {
        cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist[i], 1, &histSize, &histRange);
    }

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
    
    cv::normalize(hist[0], hist[0], 0, histImage.rows, cv:: NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist[1], hist[1], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist[2], hist[2], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[0].at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(hist[0].at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[1].at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(hist[1].at<float>(i))),
                 cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist[2].at<float>(i - 1))),
                 cv::Point(bin_w * i, hist_h - cvRound(hist[2].at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::line(histImage, cv::Point(0, hist_h), cv::Point(hist_w, hist_h), cv::Scalar(0, 0, 0), 1, 8, 0);
    cv::line(histImage, cv::Point(0, hist_h), cv::Point(0, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
    
    for (int i = 0; i < histSize; i += 32)
    {
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, hist_h - 5), cv::Scalar(0, 0, 0), 1, 8, 0);
        cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, 0), cv::Scalar(200, 200, 200), 1, 8, 0);
        
        std::stringstream ss;
        ss << i;
        cv::putText(histImage, ss.str(), cv::Point(bin_w * i, hist_h - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, cv:: Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    return histImage;
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

std::unordered_map<std::string, std::string> cimg::run_cmd(const std::string & cmd)
{

    std::unordered_map<std::string, std::string> result;
    // result has key: stdout, stderr, code, err
    try
    {
#ifdef _WIN32
        // create temporary files for stdout and stderr
        char temp_path[MAX_PATH];
        GetTempPathA(MAX_PATH, temp_path);
        
        std::string stdout_file = std::string(temp_path) + "stdout_" + std::to_string(GetCurrentProcessId()) + ".txt";
        std::string stderr_file = std::string(temp_path) + "stderr_" + std:: to_string(GetCurrentProcessId()) + ".txt";
        
        // build command with redirection
        std::string full_cmd = cmd + " > \"" + stdout_file + "\" 2> \"" + stderr_file + "\"";
        
        // execute command
        int exit_code = system(full_cmd.c_str());
        result["code"] = std::to_string(exit_code);
        
        // read stdout
        std::ifstream out_stream(stdout_file, std::ios::binary);
        if (out_stream)
        {
            std::stringstream buffer;
            buffer << out_stream.rdbuf();
            result["stdout"] = buffer.str();
            out_stream.close();
        }
        
        // read stderr
        std::ifstream err_stream(stderr_file, std::ios::binary);
        if (err_stream)
        {
            std::stringstream buffer;
            buffer << err_stream.rdbuf();
            result["stderr"] = buffer.str();
            err_stream.close();
        }
        
        // cleanup temp files
        DeleteFileA(stdout_file.c_str());
        DeleteFileA(stderr_file.c_str());
        
#else
        // Unix/Linux implementation
        std::string stdout_file = "/tmp/stdout_" + std::to_string(getpid()) + ".txt";
        std::string stderr_file = "/tmp/stderr_" + std::to_string(getpid()) + ".txt";
        
        std::string full_cmd = cmd + " > " + stdout_file + " 2> " + stderr_file;
        
        int exit_code = system(full_cmd.c_str());
        result["code"] = std::to_string(WEXITSTATUS(exit_code));
        
        // read stdout
        std::ifstream out_stream(stdout_file, std::ios::binary);
        if (out_stream)
        {
            std::stringstream buffer;
            buffer << out_stream.rdbuf();
            result["stdout"] = buffer.str();
            out_stream.close();
        }
        
        // read stderr
        std::ifstream err_stream(stderr_file, std::ios::binary);
        if (err_stream)
        {
            std::stringstream buffer;
            buffer << err_stream.rdbuf();
            if (buffer.str().size() > 0)
            {
                result["stderr"] = buffer.str();
            }
            err_stream.close();
        }
        
        // cleanup temp files
        unlink(stdout_file.c_str());
        unlink(stderr_file.c_str());
#endif
    }
    catch (const std::exception & e)
    {
        result["err"] = e.what();
    }
    
    return result;
}


string cimg::run_cmd_simple(string cmd_)
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

    string bin_content = read_to_str(fn);
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

    string bin_content = read_to_str(fn);
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

string cimg::join(const vector<string> &vstr, const string & delimiter)
{
    return fmt::format("{}",fmt::join(vstr.begin(), vstr.end(), delimiter));
}

vector<string> cimg::split(string& str, const string& delimiter)
{
    string eline = str;
    size_t next = 0;
    size_t prev = 0;
    size_t delim_len = delimiter.length();

    vector<string> vec_ret{};
    vector<pair<size_t, size_t>> sub_loc;

    while ((next = eline.find(delimiter, prev)) != string::npos)
    {
        sub_loc.push_back({ prev, next - prev });
        prev = next + delim_len;
    }

    sub_loc.push_back({ prev, eline.length() - prev });

    vec_ret.reserve(sub_loc.size());

    for (auto ep : sub_loc)
    {
        auto sub_str = eline.substr(ep.first, ep.second);
        vec_ret.push_back(sub_str);
    }

    return vec_ret;
}


void cimg::chomp(vector<string>& vec)
{
    for (auto& s : vec)
    {
        chomp(s);
    }
}

void cimg::chomp(string& str)
{
    while (!str.empty() && (str.back() == '\n' || str.back() == '\r'))
    {
        str.pop_back();
    }
}

vector<string> cimg::read_all(const string &fn)
{
    // with LR
    auto vfc = read_to_vec(fn);

    for(auto & eline: vfc)
    {
        chomp(eline);
        eline += "\n";
    }

    return vfc; 
}

vector<string> cimg::read_to_vec(string fn)
{
    // no LR
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
        auto v_s = split(e_line, ";");
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
        auto v_s = split(e_line, ",");
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

void cimg::fcoutln(string fn,const string& id_s, string flag_w)
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
        cout << "- make sure the file path is accessible!" << "\t" << fn << endl;
    }
    assert(if_.is_open());

    if_ << id_s << std::endl;
    if_.close();
}

void cimg::fcout(string fn,const string& id_s, string flag_w)
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
        cout << "- make sure the file path is accessible!" << "\t" << fn << endl;
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
double cimg::round(double val, int precision)
{
    std::stringstream s;
    s << std::setprecision(precision) << std::setiosflags(std::ios_base::fixed) << val;
    s >> val;
    return val;
}
float cimg::round(float val, int precision)
{
    std::stringstream s;
    s << std::setprecision(precision) << std::setiosflags(std::ios_base::fixed) << val;
    s >> val;
    return val;
}
void cimg::str_to_bin_file(string fn, const string &str_to_serial)
{
    ofstream of_(fn.c_str(), ios::binary);
    of_ << str_to_serial; // never have "<< endl";
    of_.close();
}

std::vector<std::string> cimg::vec_grep_v_empty(std::vector<std::string>& vstr)
{
    // vstr = ci.filter_out_vstr(vstr, R"(^#.*)");
	string pattern = R"(^\s*$)";
    vstr = vec_grep_v(vstr, pattern);
    return vstr; 
  
    return vstr;
}

std::vector<std::string> cimg::vec_grep_v(std::vector<std::string>& vstr, const std::string& pattern) 
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


std::string cimg::b64(const std::string& in)
{
    std::string out;
    int val = 0, valb = -6;
    for (unsigned char c : in)
    {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0)
        {
            out.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6)
    {
        out.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (out.size() % 4)
    {
        out.push_back('=');
    }
    return out;
}

inline bool cimg::is_b64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

// Function to decode a Base64 string
std::string cimg::b64_d(const std::string& in)
{
    std::string out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
    {
        T[base64_chars[i]] = i;
    }

    int val = 0, valb = -8;
    for (unsigned char c : in)
    {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0)
        {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::unordered_map<std::string, std::string> cimg::parse_args(int argc, char** argv, const unordered_map<string, string>& arg_map_default, const string& help_msg_)
{
    // help_msg must use R"()" to define raw string literal to avoid escape char issues
    std::unordered_map<std::string, std::string> arg_map = arg_map_default;

    auto help_msg = help_msg_;
    if (argc == 1 or argv[1] == string("--help"))
    {

        help_msg = replace_str(help_msg, "\\t", "\t", "g");
        help_msg = replace_str(help_msg, "\\n", "\n", "g");

        cout << "help msg:" << endl;
        cout << help_msg << endl;
        exit(0);
    }


    // first need verify argc is even number
    if (argc % 2 == 0)
    {
        throw std::invalid_argument("Invalid number of arguments. Expected key-value pairs.");
    }



    // assertx("input folder must match raw_data !", rex(arg_map["--folder"]) % m(R"(raw_data)"));


    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        // check if it's a key (starts with --)
        if (arg.rfind("--", 0) == 0)
        {
            std::string key = arg;
            std::string value;

            // check for = in the argument (--key=value format)
            size_t eq_pos = arg. find('=');
            if (eq_pos != std::string:: npos)
            {
                key = arg.substr(0, eq_pos);
                value = arg.substr(eq_pos + 1);
            }
            // check if next argument is the value (--key value format)
            else if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                value = argv[++i];
            }
            else
            {
                value = "true";  // flag without value
            }

            arg_map[key] = value;
        }
    }

    return arg_map;
}


string cimg::norm_path(const std::string& path)
{
    fs::path p(path);
    fs::path normalized = p.lexically_normal();
    return normalized.string();
}


namespace fs = std::filesystem;

bool cimg::fs_e(const std::string & fn)
{
    return fs::exists(fn);
}

bool cimg::fs_f(const std::string & fn)
{
    return fs::exists(fn) && fs::is_regular_file(fn);
}

bool cimg::fs_d(const std::string & fn)
{
    return fs::exists(fn) && fs::is_directory(fn);
}

bool cimg::fs_rm(const std::string & fn)
{
    try
    {
        if (fs::exists(fn))
        {
            if (fs::is_directory(fn))
            {
                fs:: remove_all(fn);
            }
            else
            {
                fs::remove(fn);
            }
            return true;
        }
        return false;
    }
    catch (...)
    {
        return false;
    }
}

bool cimg::fs_touch(const std::string & fn)
{
    try
    {
        fs::path p(fn);
        
        // create parent directories if not exist
        if (p.has_parent_path())
        {
            fs::path parent = p.parent_path();
            if (! fs::exists(parent))
            {
                fs::create_directories(parent);
            }
        }
        
        if (fs::exists(fn))
        {
            fs::last_write_time(fn, fs:: file_time_type:: clock::now());
            return true;
        }
        else
        {
            std::ofstream ofs(fn);
            if (ofs)
            {
                ofs.close();
                return true;
            }
            return false;
        }
    }
    catch (...)
    {
        return false;
    }
}

std::string cimg::ts2str(const fs::file_time_type & ftime)
{
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
    );
    std::time_t tt = std:: chrono::system_clock::to_time_t(sctp);
    std::tm tm;
    
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

std::unordered_map<std::string, std::string> cimg::fs_stat(const std::string & fn)
{
    std::unordered_map<std:: string, std::string> result;
    
    try
    {
        if (! fs::exists(fn))
        {
            result["err"] = "not_exist";
            return result;
        }
        
        result["path"] = fn;
        result["type"] = fs::is_directory(fn) ? "dir" : "file";
        result["size"] = std::to_string(fs::file_size(fn));
        result["mtime"] = ts2str(fs::last_write_time(fn));
        
        auto perms = fs::status(fn).permissions();
        std::stringstream ss;
        ss << std::oct << static_cast<int>(perms);
        result["perm"] = ss.str();
        
        // get atime and ctime using platform-specific API
#ifdef _WIN32
        struct _stat64 st;
        if (_stat64(fn.c_str(), &st) == 0)
        {
            char buf[64];
            std:: tm tm;
            
            // access time
            localtime_s(&tm, &st.st_atime);
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
            result["atime"] = std::string(buf);
            
            // create time (birth time on Windows)
            localtime_s(&tm, &st.st_ctime);
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
            result["ctime"] = std::string(buf);
        }
#else
        struct stat st;
        if (stat(fn.c_str(), &st) == 0)
        {
            char buf[64];
            std::tm tm;
            
            // access time
            localtime_r(&st.st_atime, &tm);
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
            result["atime"] = std:: string(buf);
            
            // change time (or birth time if available)
#ifdef __APPLE__
            localtime_r(&st.st_birthtime, &tm);
#else
            localtime_r(&st.st_ctime, &tm);
#endif
            std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
            result["ctime"] = std::string(buf);
        }
#endif
    }
    catch (const std::exception & e)
    {
        result["err"] = e.what();
    }
    
    return result;
}

bool cimg::fs_mv(const std::string & from, const std:: string & to)
{
    try
    {
        if (! fs::exists(from))
        {
            return false;
        }
        
        fs:: path from_path(from);
        fs::path to_path(to);
        
        // create parent directories of destination if not exist
        if (to_path. has_parent_path())
        {
            fs::path parent = to_path.parent_path();
            if (! fs::exists(parent))
            {
                fs::create_directories(parent);
            }
        }
        
        // if destination is an existing directory, move into it
        if (fs::exists(to) && fs::is_directory(to))
        {
            to_path = to_path / from_path.filename();
        }
        
        // if destination exists, remove it first
        if (fs::exists(to_path))
        {
            if (fs::is_directory(to_path))
            {
                fs::remove_all(to_path);
            }
            else
            {
                fs::remove(to_path);
            }
        }
        
        // rename/move the file or directory
        fs::rename(from, to_path);
        
        return true;
    }
    catch (...)
    {
        return false;
    }
}

nlohmann::json cimg::fs_read_json(const string &fn)
{
    nlohmann::json j = nlohmann::json::parse( read_to_str(fn) );
    return j;
}

void cimg::fs_write_json(const string &fn, const nlohmann::json &j, int indent)
{
    fs_touch(fn); 
    fcoutln(fn, j.dump(indent), "ios::trunc");
}

std::vector<std::string> cimg::fs_ls(const std::string & dir_path, const std::string & pattern)
{
    return get_folder_content(dir_path, pattern, "all", false);
}

std::vector<std::string> cimg:: fs_ls_r(const std::string & dir_path, const std::string & pattern)
{
    // cout_("dir_path:{}, pattern:{}\n", dir_path, pattern);
    return get_folder_content(dir_path, pattern, "all", true);
}

bool cimg::fs_cp(const std::string & from, const std::string & to)
{
    try
    {
        if (! fs::exists(from))
        {
            return false;
        }
        
        fs::path from_path(from);
        fs::path to_path(to);
        
        // create parent directories of destination if not exist
        if (to_path. has_parent_path())
        {
            fs::path parent = to_path.parent_path();
            if (! fs::exists(parent))
            {
                fs::create_directories(parent);
            }
        }
        
        if (fs::is_directory(from))
        {
            // if destination exists and is not a directory, remove it
            if (fs::exists(to) && ! fs::is_directory(to))
            {
                fs::remove(to);
            }
            
            // copy directory recursively
            if (!  fs::exists(to))
            {
                fs::create_directories(to);
            }
            
            for (const auto & entry : fs::recursive_directory_iterator(from))
            {
                fs::path rel_path = fs::relative(entry.path(), from);
                fs::path dest_path = to_path / rel_path;
                
                if (fs::is_directory(entry))
                {
                    fs:: create_directories(dest_path);
                }
                else
                {
                    if (dest_path.has_parent_path())
                    {
                        fs::create_directories(dest_path.parent_path());
                    }
                    fs::copy_file(entry. path(), dest_path, fs::copy_options::overwrite_existing);
                }
            }
        }
        else
        {
            // if destination is a directory, copy file into it
            if (fs::exists(to) && fs::is_directory(to))
            {
                to_path = to_path / from_path.filename();
            }
            
            // copy single file
            fs::copy_file(from, to_path, fs::copy_options::overwrite_existing);
        }
        
        return true;
    }
    catch (...)
    {
        return false;
    }
}

uintmax_t cimg::fs_du(const std::string & fn)
{

    auto calc_dir_size = [](const std:: string & fn)
    {
        uintmax_t total = 0;

        try
        {
            for (const auto & entry : fs:: recursive_directory_iterator(fn))
            {
                if (fs::is_regular_file(entry))
                {
                    total += fs::file_size(entry);
                }
            }
        }
        catch (...)
        {
        }

        return total;
    };

    
    std::unordered_map<std::string, std:: string> result;
    
    try
    {
        if (! fs::exists(fn))
        {
            result["err"] = "not_exist";
        }
        
        uintmax_t total_size = 0;
        
        if (fs::is_directory(fn))
        {
            total_size = calc_dir_size(fn);
        }
        else
        {
            total_size = fs::file_size(fn);
        }
        
        result["path"] = fn;
        result["bytes"] = std::to_string(total_size);
        result["kb"] = std::to_string(total_size / 1024);
        result["mb"] = std::to_string(total_size / (1024 * 1024));
    }
    catch (const std:: exception & e)
    {
        result["err"] = e. what();
    }

    if (result.find("err") != result.end())
    {
        // std::cerr << "- fs_du error: " << result["err"] << "\t" << fn << std::endl;
        throw std::runtime_error(result["err"]);
        return -1;
    }

    
    return result["bytes"].empty() ? -1 : std::stoi(result["bytes"]);
}

d_ts cimg::ts0()
{
    auto tick_v = GetTickCount64(); 
    d_ts e_d_ts; 
    e_d_ts.t0 = tick_v;
    e_d_ts.ts0 = get_timestamp();
    return e_d_ts;
}

d_ts cimg::ts1(d_ts & stru_t0)
{  
    

    d_ts & e_d_ts = stru_t0;
    e_d_ts.ts1 = get_timestamp();

    e_d_ts.t1 = GetTickCount64();
	stru_t0.d = e_d_ts.t1 - e_d_ts.t0;
 
    auto d_ts_t = stru_t0.d;

    e_d_ts.ms = d_ts_t % 1000;
    d_ts_t /= 1000;

    e_d_ts.second = d_ts_t % 60;
    d_ts_t /= 60;

    e_d_ts.minute = d_ts_t % 60;
    d_ts_t /= 60;

    e_d_ts.hour = d_ts_t;

    e_d_ts.s_d = s_("{:2}h{:2}m{:2}s{:4}ms", e_d_ts.hour, e_d_ts.minute, e_d_ts.second, e_d_ts.ms);
	cout << "time cost: " << e_d_ts.s_d << endl;

    return e_d_ts;
}


int cimg::difftime(const time_t& t1, const time_t& t0)
{
    return std::difftime(t1,t0);
}

int difftime(const time_t& t1, const time_t& t0, string& timestring)
{
    int diff_seconds = std::difftime(t1, t0);
    int hours = diff_seconds / 3600;
    int minutes = (diff_seconds % 3600) / 60;
    int seconds = diff_seconds % 60;
    timestring = s_("{:02}h {:02}m {:02}s", hours, minutes, seconds);
    return diff_seconds;
}

void cimg::byte2cvmat(byte *src, int rows, int cols, int chn, cv::Mat &id_mat)
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


cv::Mat cimg::img2hsv(const cv::Mat &inputImage)
{
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    auto value = hsvChannels[2].clone(); // V channel

    return value;
}

cv::Rect cimg::expand_rect(const cv::Rect& rect, float expand_ratio, const cv::Mat& img)
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

bool cimg::is_contour_same(const vector<cv::Point>& contour1, const vector<cv::Point>& contour2)
{
    bool ret_same = false;

    cv::Moments m1 = cv::moments(contour1);
    cv::Moments m2 = cv::moments(contour2);

    const float lowestF = 1e-6;
    if (abs(m1.m00) < lowestF) { m1.m00 = lowestF; }
    if (abs(m2.m00) < lowestF) { m2.m00 = lowestF; }

    cv::Point2f center1(m1.m10 / m1.m00, m1.m01 / m1.m00);
    cv::Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);

    double centerDistance = cv::norm(center1 - center2);

    double maxWidth = max(cv::boundingRect(contour1).width, cv::boundingRect(contour2).width);
    double centerDeviation = centerDistance / maxWidth;

    if (centerDeviation > 0.2) { return ret_same; }

    double area1 = cv::contourArea(contour1);
    double area2 = cv::contourArea(contour2);

    double areaDeviation = abs(area1 - area2) / max(area1, area2);

    if (areaDeviation <= 0.38) { ret_same = true; }

    return ret_same;
}

cv::Mat cimg::reverse_pad2square(const cv::Mat& padded_img, const cv::Size& original_size)
{
    int padded_height = padded_img.rows;
    int padded_width = padded_img.cols;
    int original_height = original_size.height;
    int original_width = original_size.width;

    int top = (padded_height - original_height) / 2;
    int bottom = top + original_height;
    int left = (padded_width - original_width) / 2;
    int right = left + original_width;

    cv::Rect roi(left, top, original_width, original_height);
    cv::Mat cropped_img = padded_img(roi).clone(); // Use clone to create a new Mat

    return cropped_img;
}

cv::Mat cimg::pad2square(const cv::Mat& img, const cv::Scalar& pad_value)
{
    // auto orig_size = img.size();
    int height = img.rows;
    int width = img.cols;
    int size = (std::max)(height, width);
    int top = (size - height) / 2;
    int bottom = size - height - top;
    int left = (size - width) / 2;
    int right = size - width - left;

    cv::Mat padded_img;
    cv::copyMakeBorder(img, padded_img, top, bottom, left, right, cv::BORDER_CONSTANT, pad_value);
    return padded_img;
}

string cimg::tolower(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}


string cimg::toupper(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

bool cimg::endwith(const std::string& str, const std::string& suffix)
{
    if (str.length() >= suffix.length())
    {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    else
    {
        return false;
    }
}

std::string cimg::abs_path(const std::string& path)
{
    fs::path p(path);
    fs::path abs_path = fs::absolute(p);
    fs::path normalized = abs_path.lexically_normal();
    return normalized.string();
}


std::vector<std::string> cimg::get_folder_content(const string& dir_path, const string& pattern, const string& type, bool recursive)
{
    std::vector<std::string> result;

    cout_("dir_path:{}, pattern:{}, type:{}, recursive:{}\n", dir_path, pattern, type, recursive);

    if (! fs::exists(dir_path) || !fs::is_directory(dir_path))
    {
        return result;
    }

    auto match_entry = [&](const fs::directory_entry& entry) -> bool
    {
        if ((type == "file" || type == "files") && ! entry.is_regular_file())
        {
            return false;
        }
        if ((type == "folder" || type == "folders") && !entry.is_directory())
        {
            return false;
        }
        if (type == "all" || type == "file|folder" || type == "files|folders")
        {
            if (! entry.is_regular_file() && !entry.is_directory())
            {
                return false;
            }
        }
        return true;
    };

    auto process_entry = [&](const fs::directory_entry& entry)
    {
        if (match_entry(entry))
        {
            string abs_path = fs::absolute(entry.path()).string();
            abs_path = norm_path(abs_path);
            if (pattern.empty() || rex(abs_path) % m(pattern))
            {
                result.push_back(abs_path);
            }
        }
    };

    try
    {
        if (recursive)
        {
            for (const auto& entry : fs::recursive_directory_iterator(dir_path))
            {
                process_entry(entry);
            }
        }
        else
        {
            for (const auto& entry : fs::directory_iterator(dir_path))
            {
                process_entry(entry);
            }
        }
    }
    catch (const fs::filesystem_error& e)
    {
        cerr << "Error accessing directory: " << e.what() << endl;
    }

    return result;
}



std::vector<std::string> cimg::vec_grep(std::vector<std::string>& vstr, const std::string& pattern) 
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
std::vector<T> cimg::vec_norm_by_minmax(const vector<T>& v)
{

    auto v_min = *std::min_element(v.begin(), v.end());
    auto v_max = *std::max_element(v.begin(), v.end());
    vector<T> v2 = v;
    for (auto& e : v2)
    {
        e = (e - v_min) / (v_max - v_min) * 1.0f;
    }
    return v2;
}

template <typename T>
std::vector<T> cimg::vec_smooth(const std::vector<T> &v, int window_size)
{
    assert(window_size > 2); // Ensure window size is positive and odd
    int half_window = window_size / 2;
    std::vector<T> smoothed(v.size());

    for (size_t i = 0; i < v.size(); ++i)
    {
        T sum = 0;
        int count = 0;

        // Calculate the sum within the window
        for (int j = -half_window; j <= half_window; ++j)
        {
            int idx = static_cast<int>(i) + j;
            if (idx >= 0 && idx < static_cast<int>(v.size()))
            {
                sum += v[idx];
                count++;
            }
        }

        smoothed[i] = sum / count; // Average value
    }

    return smoothed;
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

string cimg::read_to_str(const string & fn)
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
void cimg::norm_img()
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

void cimg::w_i(const string& fn, const cv::Mat& img_)
{
	cv::imwrite(fn, img_);
}

void cimg::w_i(const string& fn)
{
    cv::imwrite(fn, img);
}

void cimg::r_i(string fn)
{
	read_img(fn);
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


time_t cimg::str2time(const string& str, const string& fmt)
{
    struct tm tm_info = {};

#ifdef _WIN32
    // Windows: use sscanf based parsing
    if (fmt == "%Y-%m-%d %H:%M:%S")
    {
 
        sscanf_s(str.c_str(), "%d-%d-%d %d:%d:%d",
            &tm_info.tm_year, &tm_info.tm_mon, &tm_info.tm_mday,
            &tm_info.tm_hour, &tm_info.tm_min, &tm_info.tm_sec);
        // use s_

    

        tm_info.tm_year -= 1900;
        tm_info.tm_mon -= 1;
    }
    else if (fmt == "%Y-%m-%d")
    {
        sscanf_s(str.c_str(), "%d-%d-%d",
            &tm_info.tm_year, &tm_info.tm_mon, &tm_info.tm_mday);
        tm_info.tm_year -= 1900;
        tm_info.tm_mon -= 1;
    }
    else if (fmt == "%Y/%m/%d %H:%M:%S")
    {
        sscanf_s(str.c_str(), "%d/%d/%d %d:%d:%d",
            &tm_info.tm_year, &tm_info.tm_mon, &tm_info.tm_mday,
            &tm_info.tm_hour, &tm_info.tm_min, &tm_info.tm_sec);
        tm_info.tm_year -= 1900;
        tm_info.tm_mon -= 1;
    }
    else if (fmt == "%Y/%m/%d")
    {
        sscanf_s(str.c_str(), "%d/%d/%d",
            &tm_info.tm_year, &tm_info.tm_mon, &tm_info.tm_mday);
        tm_info.tm_year -= 1900;
        tm_info.tm_mon -= 1;
    }
    else if (fmt == "%H:%M:%S")
    {
        sscanf_s(str.c_str(), "%d:%d:%d",
            &tm_info.tm_hour, &tm_info.tm_min, &tm_info.tm_sec);
    }
    else
    {
        // fallback:  try istream parsing
        istringstream ss(str);
        ss >> get_time(&tm_info, fmt.c_str());
    }
#else
    // Linux/Mac: use strptime
    strptime(str.c_str(), fmt.c_str(), &tm_info);
#endif

    return mktime(&tm_info);
}

string cimg::get_timestamp()
{
	time_t now_ = time(nullptr);
	return get_timestamp(now_);
}

string cimg::get_timestamp(const time_t & now_)
{
    const int MAX = 1024;
    char buf[MAX] = {0};
    time_t now = now_;

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


// csv_data struct for cal_range_by_ref function
struct csv_data
{
    vector<float> l;
    vector<float> l_r0;
    vector<float> r;
    vector<float> r_r0;
    vector<string> efc;
    bool need_2nd_process;
    int peak_valley_idx;
    int peak_valley_idx_left;
    int peak_valley_idx_right;
};

// Global function converted from lambda
void cal_range_by_ref( unordered_map<string, csv_data>& csvdata, int peak_valley_idx_param, float normal_thres, float markpoint_add_val, int flag_show, std::function<int(string, vector<float>&, int, string, int, int)> seed_point_wide_range)
{
    cimg ci;
    const float special_factor = 0.7;

    for (auto & epair : csvdata)
    {
        auto fn = epair.first;
        auto& e_csv_data = epair.second;

        e_csv_data.need_2nd_process = false;
        e_csv_data.peak_valley_idx = peak_valley_idx_param;

        auto e_csv_data_r0 = e_csv_data.r;

        e_csv_data.r = ci.vec_smooth(e_csv_data.r, 20);
        e_csv_data.r = ci.vec_smooth(e_csv_data.r, 7);

        e_csv_data.peak_valley_idx_left = seed_point_wide_range(fn, e_csv_data.r, e_csv_data.peak_valley_idx, "left", 2, 30);
        e_csv_data.peak_valley_idx_right = seed_point_wide_range(fn, e_csv_data.r, e_csv_data.peak_valley_idx, "right", 2, 30);

        // get max value and index between peak_valley_idx_left ~ peak_valley_idx_right

        auto max_t = 0.0f;
        for(auto i = e_csv_data.peak_valley_idx_left; i<= e_csv_data.peak_valley_idx_right; i++) 
        {
            if (e_csv_data.r[i] > max_t) 
            {
                max_t = e_csv_data.r[i];
            }
        }

        if (max_t < normal_thres * special_factor)
        {
            e_csv_data.need_2nd_process = true; 

            cout << "special case found in file:\t" << fn << endl;

            for(int i = 0; i<e_csv_data.r.size(); i++)
            {
                if (e_csv_data.r[i] > normal_thres * 0.9)
                {
                    e_csv_data.peak_valley_idx = i;
                    break;
                }
            }

            e_csv_data.peak_valley_idx_left = seed_point_wide_range(fn, e_csv_data.r, e_csv_data.peak_valley_idx, "left", 2, 30);
            e_csv_data.peak_valley_idx_right = seed_point_wide_range(fn, e_csv_data.r, e_csv_data.peak_valley_idx, "right", 2, 30);
        }

        int intv_left = 80;
        int intv_right = 160;

        if (rex(fn) % m(R"(type_6)"))
        {
            intv_left = e_csv_data.peak_valley_idx - 10;
            intv_right = e_csv_data.r.size() - e_csv_data.peak_valley_idx - 10;
        }
        if (rex(fn) % m(R"(type_8)"))
        {
            intv_left = e_csv_data.peak_valley_idx - 400;
            intv_right = e_csv_data.r.size() - e_csv_data.peak_valley_idx - 800;;
        }

        vector<float> v_tmp;

        e_csv_data.r[e_csv_data.peak_valley_idx] += markpoint_add_val;
        e_csv_data.r[e_csv_data.peak_valley_idx_left] += markpoint_add_val;
        e_csv_data.r[e_csv_data.peak_valley_idx_right] += markpoint_add_val;

        auto l_v = e_csv_data.peak_valley_idx_left - 100;
        ;
        if (l_v < 0)
        {
            l_v = 0;
        }

        auto r_v = e_csv_data.peak_valley_idx_right + 450;
        if (r_v >= e_csv_data.r.size())
        {
            r_v = e_csv_data.r.size() - 1;
        }

        for (int i = l_v; i < r_v; i++)
        {
            v_tmp.push_back(e_csv_data.r[i]);
        }

        if (flag_show) ci.P(v_tmp, true);

        e_csv_data.peak_valley_idx_left = l_v;
        e_csv_data.peak_valley_idx_right = r_v;
    }
}


// global_ start
static unordered_map<string, string> g_arg_map;
// global_ end



#if 1  // test
namespace ns {
	using json = nlohmann::json;

    // a simple struct to model a person
    struct person {
        std::string name;
        std::string address;
        int age;
    };

	void to_json(json& j, const person& p) {
		j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
	}

	void from_json(const json& j, person& p) {
		j.at("name").get_to(p.name);
		j.at("address").get_to(p.address);
		j.at("age").get_to(p.age);
	}
};


std::vector<cv::Point> smoothContour(const std::vector<cv::Point>& inputContour, int intensity, bool closed = true) 
{
    int n = inputContour.size();
    if (n < 3 || intensity <= 0) return inputContour;

    std::vector<cv::Point> smoothedContour(n);
    
    // ´°¿Ú´óÐ¡ÓÉÁ¦¶È¾ö¶¨£¬°ë¾¶ = intensity
    int radius = intensity;

    for (int i = 0; i < n; i++) {
        double sumX = 0;
        double sumY = 0;
        double totalWeight = 0;

        for (int j = -radius; j <= radius; j++) {
            int idx;
            if (closed) {
                // ±ÕºÏÂÖÀª£ºË÷ÒýÑ­»·»·ÈÆ
                idx = (i + j + n) % n;
            } else {
                // ·Ç±ÕºÏÂÖÀª£º±ß½ç½Ø¶Ï
                idx = i + j;
                if (idx < 0) idx = 0;
                if (idx >= n) idx = n - 1;
            }

            // ¸ßË¹È¨ÖØ£º¾àÀëµ±Ç°µãÔ½½ü£¬È¨ÖØÔ½¸ß
            // ÕâÀïµÄÈ¨ÖØ¼ò»¯Îª£º1 / (1 + distance^2) 
            // »òÕß¼òµ¥Ê¹ÓÃ¾ùÖµÆ½»¬£¨È¨ÖØÈ«Îª1£©
            double weight = 1.0; 
            
            sumX += inputContour[idx].x * weight;
            sumY += inputContour[idx].y * weight;
            totalWeight += weight;
        }

        smoothedContour[i].x = cvRound(sumX / totalWeight);
        smoothedContour[i].y = cvRound(sumY / totalWeight);
    }

    return smoothedContour;
}


namespace An {
    enum Enums : int {
        var0, var1 = 99
    };
}

namespace Bn{
    enum Enums_0 : int {
    var0, var1 = 999
};
}




#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Checks if the cell core (nucleus) is surrounded by cytoplasm.
 *
 * @param img           Grayscale image (CV_8U)
 * @param coreContour   Contour of the cell core (nucleus)
 * @param maxRadius     Maximum search radius (in pixels) around the core
 * @param lowPixelVal   Lower threshold for cytoplasm (exclusive > low)
 * @param highPixelVal  Upper threshold for cytoplasm (exclusive < high)
 * @param minCytoRatio  Minimum ratio of cytoplasm pixels in the search ring (default 0.25)
 * @return true if the core is sufficiently surrounded by cytoplasm
 */
bool isCoreSurroundedByCyto(const cv::Mat& img,
    const std::vector<cv::Point>& coreContour,
    int maxRadius,
    int lowPixelVal,
    int highPixelVal,
    double minCytoRatio = 0.25)
{
    if (img.empty() || coreContour.empty() || maxRadius <= 0) {
        return false;
    }

    // Ensure image is grayscale
    cv::Mat gray;
    if (img.channels() > 1) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = img;
    }

    // Create nucleus mask
    cv::Mat nucleusMask = cv::Mat::zeros(gray.size(), CV_8U);
    cv::drawContours(nucleusMask, std::vector<std::vector<cv::Point>>{coreContour}, 0, cv::Scalar(255), -1);

    // Create outer mask (dilated)
    cv::Mat outerMask = nucleusMask.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * maxRadius + 1, 2 * maxRadius + 1));
    cv::dilate(outerMask, outerMask, kernel);

    // Search ring = outer - nucleus
    cv::Mat searchRing;
    cv::subtract(outerMask, nucleusMask, searchRing);

    int totalRingPixels = cv::countNonZero(searchRing);
    if (totalRingPixels == 0) return false;

    // Create cytoplasm mask: lowPixelVal < pixel < highPixelVal
    cv::Mat cytoMask;
    cv::inRange(gray, lowPixelVal + 1, highPixelVal - 1, cytoMask);

    // Count cytoplasm pixels inside the search ring
    cv::Mat cytoInRing;
    cv::bitwise_and(cytoMask, searchRing, cytoInRing);

    int cytoPixels = cv::countNonZero(cytoInRing);

    double cytoRatio = static_cast<double>(cytoPixels) / totalRingPixels;

    return cytoRatio >= minCytoRatio;
}




void copy_ecellcore(cv::Mat & gray_r0, const std::vector<std::vector<cv::Point>> & v_contours, int contour_idx, cv::Point2i  v)
{


	cv::Mat contour_mask = cv::Mat::zeros(gray_r0.size(), CV_8U);
	cv::drawContours(contour_mask, v_contours, contour_idx, cv::Scalar(255), cv::FILLED);

    for (int r = 0; r < gray_r0.rows; r++)
    {
        for(int c=0; c < gray_r0.cols; c++)
        {
            auto pixel = contour_mask.at<uchar>(r, c);
            if (pixel == 255)
            {
                gray_r0.at<uchar>(r+v.y, c+ v.x) = gray_r0.at<uchar>(r, c);
            }
		}
    }

}


/*
double calcOuterRingMean(cv::Mat& gray,
    const std::vector<cv::Point>& econtour,
    double percentage)
{
    if (gray.empty() || econtour.size() < 3 || percentage <= 0.0)
        return -1.0;

    // ©¤©¤ 1. ¼ÆËãÂÖÀªÖÊÐÄ ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    cv::Moments m = cv::moments(econtour);
    if (std::abs(m.m00) < 1e-6)
        return -1.0;

    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;

    // ©¤©¤ 2. ¹¹½¨ÂÖÀªÑÚÂë£¬ÓÃÓÚÅÐ¶ÏµãÊÇ·ñÔÚÂÖÀªÄÚ ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    cv::Rect bbox = cv::boundingRect(econtour);
    // À©Õ¹ bbox£¬ÎªÍâÑÓÇøÓòÁô³ö¿Õ¼ä
    int margin = static_cast<int>(
        max(bbox.width, bbox.height) * percentage + 2);
    cv::Rect safeBbox(
        max(0, bbox.x - margin),
        max(0, bbox.y - margin),
        min(gray.cols - max(0, bbox.x - margin), bbox.width + 2 * margin),
        min(gray.rows - max(0, bbox.y - margin), bbox.height + 2 * margin)
    );

    // ©¤©¤ 3. Ã¿¸ô 25¡ã ·¢ÉäÒ»ÌõÉäÏß ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    std::vector<double> pixelValues;
	std::vector<cv::Point2i> sampledPoints; // ¼ÇÂ¼²ÉÑùµã×ø±ê
    pixelValues.reserve(360 / 25 * 10); // Ô¤¹ÀÈÝÁ¿

    const int angleStep = 25; // Ã¿ 25¡ã Ò»ÌõÉäÏß

    for (int angleDeg = 0; angleDeg < 360; angleDeg += angleStep)
    {
        double rad = angleDeg * CV_PI / 180.0;
        double dx = std::cos(rad);
        double dy = std::sin(rad);

        // ©¤©¤ 3a. ÑØÉäÏß·½ÏòÕÒµ½ÓëÂÖÀªµÄ½»µã£¨ÂÖÀª±ß½çµã£© ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
        // ´ÓÖÊÐÄ³ö·¢£¬Öð²½²½½øÖ±µ½Àë¿ªÂÖÀª
        double stepSize = 0.5; // ÑÇÏñËØ¾«¶È
        double rx = cx, ry = cy;
        double contourRadius = 0.0;

        // ÏòÍâ²½½ø£¬ÕÒµ½ÂÖÀª±ß½ç
        while (true)
        {
            rx += dx * stepSize;
            ry += dy * stepSize;

            // ³¬³öÍ¼Ïñ·¶Î§ÔòÍ£Ö¹
            if (rx < 0 || rx >= gray.cols || ry < 0 || ry >= gray.rows)
                break;

            // ¼ì²âÊÇ·ñÈÔÔÚÂÖÀªÄÚ
            double inside = cv::pointPolygonTest(
                econtour, cv::Point2f((float)rx, (float)ry), false);

            if (inside < 0) // ÒÑÀë¿ªÂÖÀª
            {
                // ´ËÊ± (rx, ry) ÊÇ¸ÕÀë¿ªÂÖÀªµÄµã£¬ÏòÄÚ»ØÍËÒ»²½¼´Îª±ß½ç
                rx -= dx * stepSize;
                ry -= dy * stepSize;
                contourRadius = std::sqrt(
                    (rx - cx) * (rx - cx) + (ry - cy) * (ry - cy));
                break;
            }
        }

        if (contourRadius < 1e-6)
            continue; // ¸Ã·½ÏòÎ´ÕÒµ½ÓÐÐ§±ß½ç£¬Ìø¹ý

        // ©¤©¤ 3b. ¼ÆËãÍâÑÓ³¤¶È ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
        double extendLen = contourRadius * percentage;

        // ©¤©¤ 3c. ²É¼¯ÍâÑÓÇøÓòµÄÏñËØÖµ£¨´ÓÂÖÀª±ß½çµ½ÍâÑÓÖÕµã£©©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
        double startR = contourRadius;
        double endR = contourRadius + extendLen;

        // ²½³¤ 1 ÏñËØ£¬ÑØÉäÏß²ÉÑù
        for (double r = startR; r <= endR; r += 1.0)
        {
            int px = static_cast<int>(std::round(cx + dx * r));
            int py = static_cast<int>(std::round(cy + dy * r));

            if (px < 0 || px >= gray.cols || py < 0 || py >= gray.rows)
                break; // ³¬³öÍ¼Ïñ·¶Î§

            pixelValues.push_back(
                static_cast<double>(gray.at<uchar>(py, px)));
			sampledPoints.push_back(cv::Point2i(px, py)); // ¼ÇÂ¼²ÉÑùµã×ø±ê
        }
    }

    if (pixelValues.empty())
        return -1.0;

    // ©¤©¤ 4. ¼ÆËã×ÜÌåÆ½¾ùÖµ ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    double sum = std::accumulate(pixelValues.begin(), pixelValues.end(), 0.0);



    for(auto e: sampledPoints)
    {
		gray.at<uchar>(e.y, e.x) = 255; // ±ê¼Ç²ÉÑùµã
	}


    

    return sum / static_cast<double>(pixelValues.size());
}
*/


double calcOuterRingMean(cv::Mat& gray,
    const std::vector<cv::Point>& econtour,
    double p_s = 0.45,   // ¿ªÊ¼²ÉÑù£ºÂÖÀª°ë¾¶ * p_s Ö®Íâ
    double p_e = 1.0)   // ½áÊø²ÉÑù£ºÂÖÀª°ë¾¶ * p_e Ö®Íâ
{
    if (gray.empty() || econtour.size() < 3 || p_s < 0.0 || p_e <= p_s)
        return -1.0;

    // ©¤©¤ 1. ¼ÆËãÂÖÀªÖÊÐÄ ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    cv::Moments m = cv::moments(econtour);
    if (std::abs(m.m00) < 1e-6)
        return -1.0;

    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;

    // ©¤©¤ 2. Ã¿¸ô 25¡ã ·¢ÉäÒ»ÌõÉäÏß ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    std::vector<double> pixelValues;
	std::vector<cv::Point2i> sampledPoints; // ¼ÇÂ¼²ÉÑùµã×ø±ê
    pixelValues.reserve(360 / 25 * 20);

    const int angleStep = 35;

    for (int angleDeg = 0; angleDeg < 360; angleDeg += angleStep)
    {
        double rad = angleDeg * CV_PI / 180.0;
        double dx = std::cos(rad);
        double dy = std::sin(rad);

        // ©¤©¤ 2a. ÑØÉäÏßÕÒµ½ÂÖÀª±ß½çµã ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
        double stepSize = 0.5;
        double rx = cx, ry = cy;
        double contourRadius = 0.0;

        while (true)
        {
            rx += dx * stepSize;
            ry += dy * stepSize;

            if (rx < 0 || rx >= gray.cols || ry < 0 || ry >= gray.rows)
                break;

            double inside = cv::pointPolygonTest(
                econtour, cv::Point2f((float)rx, (float)ry), false);

            if (inside < 0)
            {
                rx -= dx * stepSize;
                ry -= dy * stepSize;
                contourRadius = std::sqrt(
                    (rx - cx) * (rx - cx) + (ry - cy) * (ry - cy));
                break;
            }
        }

        if (contourRadius < 1e-6)
            continue;

        // ©¤©¤ 2b. ²ÉÑùÇø¼ä£º[contourRadius * (1+p_s), contourRadius * (1+p_e)] ©¤©¤
        double sampleStart = contourRadius * (1.0 + p_s);
        double sampleEnd = contourRadius * (1.0 + p_e);

        for (double r = sampleStart; r <= sampleEnd; r += 1.0)
        {
            int px = static_cast<int>(std::round(cx + dx * r));
            int py = static_cast<int>(std::round(cy + dy * r));

            if (px < 0 || px >= gray.cols || py < 0 || py >= gray.rows)
                break;

            pixelValues.push_back(
                static_cast<double>(gray.at<uchar>(py, px)));

            // sampledPoints.push_back(cv::Point2i(px, py)); // ¼ÇÂ¼²ÉÑùµã×ø±ê
        }
    }

    if (pixelValues.empty())
        return -1.0;

    // ©¤©¤ 3. ¼ÆËã×ÜÌåÆ½¾ùÖµ ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
    double sum = std::accumulate(pixelValues.begin(), pixelValues.end(), 0.0);


    //for (auto e : sampledPoints)
    //{
    //    gray.at<uchar>(e.y, e.x) = 255; // ±ê¼Ç²ÉÑùµã
    //}
    return sum / static_cast<double>(pixelValues.size());
}



#endif 



// main_
int main(int argc, char** argv)
{

    //  global var  // 
    com& ec = s_com;
    ec.glo_init(argc, argv);
    // ecl(string(argv[0]) + " start__");
    // string dirname = "D:\\jd\\t\\platform_test_data\\";
    // dirname = string("D:/jd/t/img_rgb_cmp/_309/sz_big/");
    // string fn_r0 = "d:/jd/t/rgb_10_12.jpg";
    cimg ci; cimg ci_0; cimg ci_1; cimg ci_2;
    // -------- //



#if 1
    string fn = "d:/jd/t/t0/b_rgb.jpg"; 
    ci.read_img(fn); 
    // ci.resize(0.5f); 

    float scale_f = 0.5f; 
    cv::resize(ci.img, ci.img, cv::Size((int)(ci.img.cols * scale_f), int(ci.img.rows * scale_f)));

    ci.w_i("d:/jd/t/t0/b_half_rgb.jpg");





#endif 
#if 0

    string fn = "d:/jd/t/gray_r0.png"; 
    ci.read_img(fn); 
	auto gray_r0 = ci.img.clone();


    ci.threshold(66); 
    ci.find_contours();

    ci.v_contours; 


    int cnt = 0;

    vector<int> roi_contour_idx;
    auto gray = gray_r0.clone();
    for (auto e : ci.v_contours)
    {
        auto bb = cv::boundingRect(e);
        auto area = cv::contourArea(e);

        // bb centerx, centery
        auto centerx = bb.x + bb.width / 2;
        auto centery = bb.y + bb.height / 2;

        // ci.puttext cnt to center

        if (area > 50)
        {
            cv::putText(gray, std::to_string(cnt), cv::Point(centerx, centery), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1);
            cout << "cnt:" << cnt << endl;
            roi_contour_idx.push_back(cnt);
        }

        cnt++;
    }



    ci.s_i(gray); 
	ci.w_i("d:/jd/t/1.png", gray);


    vector<int> roiidx = {
    45,42,7,
    64,43,33, 
    44,34,51,55
    };


    for (auto eidx : roi_contour_idx)
    {
        auto ec = ci.v_contours[eidx];

		// get center of ec
		auto bb = cv::boundingRect(ec);
		auto centerx = bb.x + bb.width / 2;
		auto centery = bb.y + bb.height / 2;

		


        auto mean_val = calcOuterRingMean(gray_r0, ec, 0.45, 1);

        bool flag_nuk = false;
        if (mean_val > 188)
        {
            flag_nuk = true; 
        }


        if(flag_nuk)
        {
            cv::putText(gray_r0, std::to_string(eidx) + ":" + std::to_string(flag_nuk) + ":" + to_string((int)mean_val), cv::Point(centerx, centery), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1);
        }
      

		// ci.s_i(gray_r0);

        if (flag_nuk)
        {
            cout << "contour idx:" << eidx << ", overall outer mean: " << std::fixed << std::setprecision(1) << mean_val << endl;
        }
		
    }


    // ci.s_i(gray_r0); 



	ci.w_i("d:/jd/t/gray_r0_outer_mean.png", gray_r0);




#endif 

#if 0
    string fn = "d:/jd/t/git/analysis-ui/out/HeerPathData/result/__view_cache__/DT4PH052701/rgb_8_8.jpg";

	ci.r_i(fn);
    ci.cvtcolor("GRAY");

    



    
    // ci.s_i(); 


    ci.img = ci.img({ 100,700 }, { 100,700 });

    auto gray_r0 = ci.img.clone();

	
    

    

    cout << 1 << endl;
	// ci.s_i();


    ci.threshold(66);


    
    ci.find_contours(); 

    ci.v_contours;

    // cv::Point2i v = {50, 176}; 
    unordered_map<int, cv::Point2i> ctov = {
    {58, {50, 176}},
    {38, {-176, 1}},
    {32, {97, 147}}
    };


    for (auto& e : ctov)
    {
        copy_ecellcore(gray_r0, ci.v_contours, e.first, e.second);

    }

    

    


	ci.s_i(gray_r0);


	ci.w_i("d:/jd/t/gray_r0.png", gray_r0);
	


    

    
    
    int cnt = 0; 

	vector<int> roi_contour_idx;
    auto gray = gray_r0.clone();
    for(auto e: ci.v_contours)
    {
		auto bb = cv::boundingRect(e);
		auto area = cv::contourArea(e);

        // bb centerx, centery
		auto centerx = bb.x + bb.width / 2;
		auto centery = bb.y + bb.height / 2;

        // ci.puttext cnt to center
    
        if (area > 50)
        {
            cv::putText(gray, std::to_string(cnt), cv::Point(centerx, centery), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1);
            cout << "cnt:" << cnt << endl;
			roi_contour_idx.push_back(cnt);
        }
		
		cnt++;
	}


//    ci.s_i(gray); 

    ci.w_i("d:/jd/t/1.png", gray); 




    gray = gray_r0.clone(); 
    for(auto eroi: roi_contour_idx)
    {
        auto e_contour = ci.v_contours[eroi];
        bool have_cyto = isCoreSurroundedByCyto(gray_r0, e_contour, 6, 77, 111, 0.15); 

       
		auto bb = cv::boundingRect(e_contour);
		auto area = cv::contourArea(e_contour);
		// bb centerx, centery
		auto centerx = bb.x + bb.width / 2;
		auto centery = bb.y + bb.height / 2;


        cv::putText(gray, std::to_string(eroi)+":"+std::to_string(have_cyto), cv::Point(centerx, centery), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1);

	}


	ci.s_i(gray);

    


    cout << 1 << endl; 

#endif


#if 0


    ci.r_i("d:/jd/t/mask_non_square.png"); 
    ci.cvtcolor(); 

    ci.threshold(122);

    ci.img;

    auto img_contour = ci.find_contours(255, 1111); 


    cout << ci.v_contours.size() << endl; 

    ci.w_i("d:/jd/t/c.png", img_contour);


#endif 
#if 0
    ci.r_i("d:/jd/t/input_image_big.png"); 
	// gausesmooth ci.img with kernel size 5
    cv::GaussianBlur(ci.img, ci.img, cv::Size(0,0), 2.4);

	ci.w_i("d:/jd/t/input_image_big_smooth.png", ci.img);



#endif 
#if 0
    ci.r_i("d:/jd/t/f32_b_s.png"); 







    ci.img; 
    ci_0.r_i("d:/jd/t/f32_s_s.png");

    ci_0.img; 



    ci.cvtcolor("RGB");

    // for ci.img if pixel value > 122, set to red color
    for (int r = 0; r < ci.img.rows; r++)
    {
        for (int c = 0; c < ci.img.cols; c++)
        {
            auto pixel = ci.img.at<cv::Vec3b>(r, c);
            if (pixel[0] > 122 || pixel[1] > 122 || pixel[2] > 122)
            {
                ci.img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 255); // set to red
            }
        }
	}





    ci_0.cvtcolor("RGB"); 




	cv::Mat img_ = ci_0.img * 0.5 + ci.img * 0.5;

    // ci.s_i(img_); 



    ci.w_i("d:/jd/t/contour_cmp.png", img_); 



    ci_1.r_i("d:/jd/t/input_image_square_resized.png");
    cv::Mat img_m = img_ * 0.7 + ci_1.img * 0.3; 


    cout << ci.info()<< ci_0.info() << endl; 

    ci.w_i("d:/jd/t/contour_cmp_merge.png", img_m);

#endif 
#if 0

    string fn = "d:/jd/t/t0/20260305001_08_07.jpg"; 
    ci.r_i(fn); 
    ci.resize(0.5); 
    cout << ci.info() << endl; 
    ci.w_i("d:/jd/t/t0/20260305001_08_07_small.png"); 




#endif 
#if 0
    vector<int> v = { 1 };

    for (auto& e : ci.R(10))
    {
        cout_("{}{}", v, LR);
        vector<int> v_s = v;
        vector<int> v_p = v;
        v_s.push_back(0);

        auto v_p_t = v_p;
        v_p = { 0 };
        for (auto e : v_p_t)
        {
            v_p.push_back(e);
        }


        vector<int> vr = {};
        for (int i = 0; i < v_s.size(); i++)
        {
            vr.push_back(v_s[i] + v_p[i]);
        }


        v = vr;

    }


#endif 

#if 0


    vector<int> v = { 1}; 

    auto yh_trangle = [](vector<int> v)
    {
        vector<int>  v2 = { 1 };
        for (int i = 0; i < v.size(); i++)
        {
            auto bi = v.size() - 1;
            // cout << i << ", " << bi << endl;


            if (i + 1 <= v.size() - 1)
            {
                // cout << v[i] + v[i + 1] << endl;
                v2.push_back(v[i] + v[i + 1]);
            }
            else
            {
                break;
            }
        }

        v2.push_back(1);

        

        return v2; 
    };
   
    for (auto e : ci.R(28))
    {

        v = yh_trangle(v);

        cout_("{}{}", v, LR);

    }




#endif 
#if 0
    string fn = "d:/jd/t/t0/tilejpg/1.txt";

    auto v_str = ci.read_all(fn); 
    for (auto & e : v_str)
    {
        //cout << e.substr(100) << endl;
        ci.chomp(e); 
        auto v_part = ci.split(e, ";"); 
     
        auto img_nm = v_part[0];
		auto start_end = v_part[1];  //[1,3]

		int start = 0, end = 0;
        if (sscanf_s(start_end.c_str(), "[%d,%d]", &start, &end) != 2)
        {
            cerr << "Error parsing start and end indices from: " << start_end << endl;
            continue; // Skip this line if parsing fails
		}

		cout << start << "\t" << end << endl;

       
        
		auto type_points = v_part[2]; // {1, [2,3],[3,4],[6,7]},{1, [22,33],[43,44],[56,67]}
        cout << type_points << endl; 
        //string type_points = "{1,[2,3],[3,4],[6,7]},{3,[22,33],[43,44],[56,67]}"; 

        vector<int> v_type;
		vector<vector<cv::Point>> v_contours;
		auto v_type_points = ci.split(type_points, "},"); // split by "},{"

        for (auto & e : v_type_points)
        {
            e = ci.replace_str(e, "}", "");
            e = ci.replace_str(e, "{", "");

            int e_type = e[0] - '0'; 
            e = e.substr(2); // [2,3],[3,4],[6,7]
            vector<cv::Point> e_contour = {}; 
            // e_contour parse e to get value
			e = ci.replace_str(e, "],[", ",", "g");
			e = e.substr(1, e.size() - 2); // remove leading [ and trailing ]

			auto v_pt = ci.split(e, ","); // split by ","
            for (size_t i = 0; i < v_pt.size(); i += 2)
            {
                int x = stoi(v_pt[i]);
                int y = stoi(v_pt[i + 1]);
                e_contour.push_back(cv::Point(x, y));
			}

            for(auto & pt : e_contour)
            {
                cout << "(" << pt.x << "," << pt.y << "):";
			}

            cout << endl<<  endl;
        }

        cout << endl; 

        




        break;
    }

#endif 

#if 0
    string fn_b = "d:/jd/t/t0/DTLISL007_07_07";
	string fn_json = fn_b + ".json";
	string fn_jpg = fn_b + ".jpg";
	string fn_s_json = fn_b + ".smooth.json";

	auto j = ci.fs_read_json(fn_json);
    auto j_s = ci.fs_read_json(fn_s_json);
   
    cout << j["contours"][0] << endl; 
	cout << j_s["contours"][0] << endl;


    std::vector<std::vector<cv::Point>> v_contours{}; 

    int cnt = 0; 
    for(auto & e_contour : j["contours"])
    {
    
        
        std::vector<cv::Point> contour_points;
        
        for (auto& point : e_contour)
        {
            int x = point["x"];
            int y = point["y"];
			contour_points.push_back(cv::Point(x, y));
         
        }
  
        

		v_contours.push_back(contour_points);
        cnt++;
 

	}


    cout << cnt << endl; 

	

    ci.r_i(fn_jpg); 

	
	auto img_cp = ci.img.clone();
    cv::drawContours(img_cp, v_contours, -1, cv::Scalar(0, 222, 255), 1);


	ci.w_i("1.jpg", img_cp(cv::Rect(33, 55, 444, 666)));

    //===
    cnt = 0;
    v_contours = {};

    for (auto& e_contour : j_s["contours"])
    {


        std::vector<cv::Point> contour_points;

        for (auto& point : e_contour)
        {
            int x = point["x"];
            int y = point["y"];
            contour_points.push_back(cv::Point(x, y));

        }



        v_contours.push_back(contour_points);
        cnt++;

    }

    cout << cnt << endl;
    ci.r_i(fn_jpg);
    img_cp = ci.img.clone();
    cv::drawContours(img_cp, v_contours, -1, cv::Scalar(0, 222, 255), 1);


    ci.w_i("2.jpg", img_cp(cv::Rect(33, 55, 444, 666)));




    ci.r_i("1.jpg"); 
	auto img_1 = ci.img.clone();
	ci.r_i("2.jpg");
	auto img_2 = ci.img.clone();

	auto img_all = ci.hconcat({ img_1, img_2 },4);
    ci.s_i(img_all);




    // draw bb on img_crop 












#endif 


#if 0

// Contours in OpenCV are represented as a vector of vectors of points
int smoothingFactor = 3; 
std::vector<std::vector<cv::Point>> v_contours = {{
    cv::Point(1220, 1086), // 0
    cv::Point(1220, 1087), // 1
    cv::Point(1220, 1087), // 2
    cv::Point(1219, 1087), // 3
    cv::Point(1218, 1087), // 4
    cv::Point(1217, 1087), // 5
    cv::Point(1215, 1088), // 8
    cv::Point(1214, 1089), // 9
    cv::Point(1214, 1090), // 10
    cv::Point(1214, 1090), // 11
    cv::Point(1213, 1090), // 12
    cv::Point(1213, 1091), // 13
    cv::Point(1212, 1092), // 14
    cv::Point(1212, 1093), // 15
    cv::Point(1212, 1093), // 16
    cv::Point(1212, 1094), // 17
    cv::Point(1212, 1095), // 18
    cv::Point(1211, 1096), // 19
    cv::Point(1212, 1096), // 20
    cv::Point(1212, 1097), // 21
    cv::Point(1212, 1098), // 22
    cv::Point(1212, 1099), // 23
    cv::Point(1213, 1099), // 24
    cv::Point(1213, 1100), // 25
    cv::Point(1214, 1101), // 26
    cv::Point(1214, 1101), // 27
    cv::Point(1215, 1102), // 28
    cv::Point(1216, 1102), // 29
    cv::Point(1217, 1103), // 30
    cv::Point(1217, 1103), // 31
    cv::Point(1217, 1104), // 32
    cv::Point(1218, 1104), // 33
    cv::Point(1219, 1104), // 34
    cv::Point(1220, 1104), // 35
    cv::Point(1220, 1104), // 36
    cv::Point(1221, 1105), // 37
    cv::Point(1222, 1105), // 38
    cv::Point(1222, 1106), // 39
    cv::Point(1222, 1105), // 40
    cv::Point(1223, 1105), // 41
    cv::Point(1224, 1105), // 42
    cv::Point(1225, 1105), // 43
    cv::Point(1225, 1105), // 44
    cv::Point(1226, 1105), // 45
    cv::Point(1227, 1105), // 46
    cv::Point(1228, 1105), // 47
    cv::Point(1228, 1105), // 48
    cv::Point(1229, 1104), // 49
    cv::Point(1230, 1104), // 50
    cv::Point(1231, 1104), // 51
    cv::Point(1231, 1104), // 52
    cv::Point(1231, 1104), // 53
    cv::Point(1231, 1103), // 54
    cv::Point(1232, 1102), // 55
    cv::Point(1233, 1101), // 56
    cv::Point(1233, 1101), // 57
    cv::Point(1233, 1101), // 58
    cv::Point(1233, 1100), // 59
    cv::Point(1233, 1099), // 60
    cv::Point(1234, 1099), // 61
    cv::Point(1234, 1098), // 62
    cv::Point(1234, 1097), // 63
    cv::Point(1234, 1096), // 64
    cv::Point(1234, 1096), // 65
    cv::Point(1233, 1095), // 66
    cv::Point(1233, 1094), // 67
    cv::Point(1233, 1093), // 68
    cv::Point(1233, 1093), // 69
    cv::Point(1232, 1092), // 70
    cv::Point(1232, 1091), // 71
    cv::Point(1231, 1090), // 72
    cv::Point(1231, 1090), // 73
    cv::Point(1230, 1089), // 74
    cv::Point(1229, 1089), // 75
    cv::Point(1228, 1088), // 76
    cv::Point(1228, 1088), // 77
    cv::Point(1227, 1088), // 78
    cv::Point(1226, 1088), // 79
    cv::Point(1225, 1088), // 80
    cv::Point(1225, 1088), // 81
    cv::Point(1224, 1088), // 82
    cv::Point(1223, 1088), // 83
    cv::Point(1222, 1087), // 84
    cv::Point(1222, 1087), // 85
    cv::Point(1222, 1086), // 86
    cv::Point(1221, 1086)  // 87
}};


for(auto &ec : v_contours)
{
    ec = smoothContour(ec, smoothingFactor, true);
}

// draw this contour in a picture
ci.img = ci.create_img_rc_chn(1500,1500,3); 
ci.img.setTo(cv::Scalar(255, 255, 200)); // set white background




cv::drawContours(ci.img, v_contours, -1, cv::Scalar(255, 0, 0), 1);


ci.img = ci.img(Rect(1100, 1000, 222, 200));
// ci.s_i();

cv::imwrite("2.jpg", ci.img);

auto img_2_jpg = ci.img.clone();


for(auto &ec: v_contours)
{
    for(auto & ep: ec)
    {
        ep.x -=1100;
        ep.y -=1000;
        ep.x *= 2; 
        ep.y *= 2;
        cout << ep.x << "," << ep.y << endl;
    }

}



ci.img = ci.create_img_rc_chn(360,360,3);
ci.img.setTo(cv::Scalar(255, 255, 200)); // set white background


cv::drawContours(ci.img, v_contours, -1, cv::Scalar(255, 0, 0), 1);


cv::imwrite("3.jpg", ci.img);
auto img_3_jpg = ci.img.clone();

for(auto &ec : v_contours)
{
    // ec = smoothContour(ec, smoothingFactor, true);
}

ci.img.setTo(cv::Scalar(255, 255, 200)); // set white background

cv::drawContours(ci.img, v_contours, -1, cv::Scalar(0, 255, 0), 1);

auto img_4_jpg = ci.img.clone();
cv::imwrite("4.jpg", ci.img);

ci.img = ci.vconcat({img_2_jpg, img_3_jpg, img_4_jpg}, 4);

cv::imwrite("5.jpg", ci.img);


#endif 

#if 0
    string file_prefix = "d:/jd/t/t0/cc1/DTH2507144_07_07"; 
    string fn_jpg = file_prefix + ".jpg"; 
    string fn_json = file_prefix + ".json"; 


    ci.r_i(fn_jpg); 

    auto corejson = ci.fs_read_json(fn_json);

    // cout << corejson["contours"][0].dump() << endl;

    vector< std::vector<Point>> v_contours {}; 

    for (auto ec : corejson["contours"])
    {

        std::vector<Point> v_ec; 
        for (auto ep : ec)
        {
            int x = ep["x"]; 
            int y = ep["y"]; 
            v_ec.push_back(Point(x, y));
            // cout << x << "," << y << endl;
        }

        v_contours.push_back(v_ec);
    }

    auto cimg_img  = ci.img.clone();

    // draw contours on ci.img

    cv::drawContours(ci.img, v_contours, -1, cv::Scalar(255,0, 0), 1);
    cv::drawContours(cimg_img, v_contours, -1, cv::Scalar(255, 0, 0), 2);

    ci.img = ci.img(Rect(111, 111, 811, 311));
    cimg_img = cimg_img(Rect(111, 111, 811, 311));

    ci.img = ci.vconcat({ci.img, cimg_img}, 4);


    cv::imwrite("1.jpg", ci.img);

#endif 

#if 0
    string fn = R"(d:\jd\t\t0\1.txt)"; 

	auto v_str = ci.read_to_vec(fn);
    
    ci.r_i(R"(D:\jd\t\git\analysis-ui\out\HeerPathData\result\DAEST124\middle\7_7_seed_test.jpg)"); 




    for (auto eline : v_str)
    {

        if ()
		json ejson = json::parse(eline);
		
		json jp = ejson["/imagePara/cellRect"_json_pointer];
        
		cv::Rect rect(
            jp["x"].get<int>(),
            jp["y"].get<int>(),
            jp["width"].get<int>(),
            jp["height"].get<int>()
		);

		// draw rect on ci.img
		cv::rectangle(ci.img, rect, cv::Scalar(0, 0, 255), 2);



      

    }

    ci.s_i(); 


	// cout_("v_str:{}{}", v_str, LR);


#endif 


#if 0

    auto vmat = ci.create_img_rc_chn(15, 15, 1); 

  //  for (auto r = 0; r < vmat.rows; r++)
  //  {
  //      for(auto c=0; c<vmat.cols; c++)
  //      {
  //          if (c=r+1)
  //          {
  //              // vmat.at<uchar>(r, c) = 1;
  //          }
		//}
  //  }

    int cnt = 0; 

    for(auto i=0; i< vmat.rows; i++)
    {
     
        for(auto j=0; j< vmat.cols; j++)
        {
            if (abs(i - j)<4)
            {
                vmat.at<uchar>(i, j) = 1;
                cnt++;
            }
		}
	}
    cout << vmat << endl; 
    cout << cnt << endl; 





#endif 

#if 0

    vector<vector<int>> vvi = {
        {1,2,3},
        {4,5,6},
        {7,8,9}
	};

    auto vvi_t = ci.vec_t(vvi);
	cout_("vvi:{}{}vvi_t:{}{}", vvi, LR, vvi_t, LR);




#endif 
#if 0

	auto vfn = ci.get_folder_content("d:/jd/t/git/analysis-ui/dna-analysis/models/custom_data/", R"(\.dat$)", "file", true);

    json j_all; 

    for (auto fn : vfn)
    {
		
        // string fn = "d:/jd/t/git/analysis-ui/dna-analysis/models/custom_data/sgp_SZK-S001.dat";
        
        // cout << fn << endl; 


        auto fc = ci.read_to_vec(fn);
        fc = ci.vec_range(fc, 1);

        vector<vector<float>>  fc_cda;  // celltype, dnaindex, area

        fc_cda.reserve(fc.size());


        int cnt = 0;
        for (auto& eline : fc)
        {
            // if (cnt > 3) break;
            eline = rex(eline) % s(R"(^.*\]\s)", "");
            auto vrow = ci.split(eline, ",");
            auto efc_cda = vector<float>({ std::stof(vrow[0]), std::stof(vrow[1]), std::stof(vrow[2]) });
            fc_cda.push_back(efc_cda);
            cnt++;
        }

        j_all[ci.fs_basename(fn)] = fc_cda;

    }



	ci.fcoutln("j_all.json", j_all.dump(1));


	auto j_all_copy = ci.fs_read_json("j_all.json");
	// cout << j_all_copy.dump(1) << endl;

    for(auto ejtem : j_all_copy.items())
    {
        auto key = ejtem.key();
        auto val = ejtem.value();

		auto vv = val.get<vector<vector<float>>>();
        auto vvmat = ci.vec_to_mat(vv); 

        auto vt = ci.mat_to_vec<float>(vvmat.t());
        

		ci.plt_scatter<float>(vt[1], vt[2], true);

        // cout_("key:{}\tvalue:{}{}{}", key, val.dump(2),vv,  LR);
	}


#endif 

#if 0
    auto vi = ci.R(300); 

	auto vi_shuffle = ci.vec_shuffle(vi);

	auto vi_s_shuffle = ci.vec_sort(vi_shuffle);

    ci.plt_scatter<int>(vi_shuffle, vi_s_shuffle, true);

	
    // ci.P(vi, true); 

#endif 
#if 0

    auto vi_d = ci.linspace(0, 10, 25); 



	auto vi = ci.vec_to_type<int>(vi_d);


    auto vm_int = ci.vec_to_mat(vi);
    auto vi_copy = ci.mat_to_vec<int>(vm_int);
	auto vi_copy_flat = ci.mat_to_vec_flatten<int>(vm_int);


	cout_("{}{}{}{}", ci.vec_flatten(vi_copy), LR, vi_copy_flat,  LR);


    auto vf = ci.vec_to_type<float>(vi);
    vf = ci.vec_shuffle(vf);

    auto vmat = ci.vec_to_mat(vf);

	vmat = ci.mat_reshape(vmat,5 , -1);
    cout << vmat << endl;
    vmat = vmat * vmat; 
    cout << vmat << endl;


	auto vvf = ci.mat_to_vec<float>(vmat);

    auto vvf_flat = ci.vec_flatten(vvf);

	cout_("{}{}", vvf_flat, LR);


#endif 


#if 0

    auto ts_str =  ci.get_timestamp();

	string results = rex(ts_str) % s(R"(_\d{6})", "");
    cout << results << endl; 

	time_t t_ = ci.str2time(results, "%Y%m%d");
    auto t = time(0); 

    ci.ts0();

    string timestring_tofill; 
    int  diff_sec = difftime(t , t_, timestring_tofill);

	cout << "diff_sec:" << timestring_tofill << "\t" << diff_sec << endl;
    
    auto ts_str_2 = ci.get_timestamp(t_);
	cout_("ts_str:{}\tts_str_2:{}{}", ts_str, ts_str_2, LR);

#endif 
#if 0

	vector<double> vi = ci.linspace(0, 100, 20);


    cv::Mat matdouble = ci.vec_to_mat(vi); 
	cout << matdouble << endl;


	cv::Mat matfloat = ci.vec_to_mat<float>(vi);
	cout << matfloat << endl;

	cv::Mat matint = ci.vec_to_mat<int>(vi);
	cout << matint << endl;

    auto vvi = ci.vec_reshape(vi, 5, -1); 
	cv::Mat matdoublereshape = ci.vec_to_mat(vvi);
	cout << matdoublereshape << endl;
	cv::Mat matfloatreshape = ci.vec_to_mat<float>(vvi);
	cout << matfloatreshape << endl;
	cv::Mat matintreshape = ci.vec_to_mat<int>(vvi);
	cout << matintreshape << endl;

    cv::Mat matintreshape_4_5 = ci.mat_reshape(matintreshape, 4, -1); 
	cout << matintreshape_4_5 << endl;



#endif 


#if 0

    auto vi = ci.linspace(0, 1,100);
	auto vmat = ci.vec_reshape(vi,20, -1);
    for (auto& er : vmat)
    {
		
        cout_("{}{}", er, LR);
		ci.vec_map(er, [&ci](double & val) { val = ci.round(val); });
        vector<int> er_int = ci.vec_to_type<int>(er);
        cout_("{}{}{}", er, er_int, LR);

    }


	// cout_("{}{}", vmat, LR);

#endif 
#if 0

    json j0; 
    j0["class0"]["id"]=10; 
    j0["class1"]["id"]=12;
    


    ci.fcoutln("1.json", j0.dump());

    auto j1 = json::parse(ci.read_to_str("1.json"));

    cout << j0 << endl; 
    cout << j1 << endl; 

    // cout << "___"<< j0[u8"ÖÐÎÄkey"] << "____" << endl;


    json j2;
    j2.push_back("abc");
    j2.push_back(30);
    j2.push_back(true); 
    j2.emplace_back(3.14);
    j2.emplace_back(ci.linspace(0, 1, 5));

    cout << j2 << endl;

    auto j2_0 = j2[0];
    auto j2_0_str = j2_0.get<string>();
    auto j2_1_str = j2[1].get<int>();
    auto j_n_1_str = j2[4].get<vector<float>>();
    cout_("{}{}", j_n_1_str, LR);


    cout_("j2_0_str:{}{}{}", j2_0_str, j2_1_str,  LR);


    json j3 = R"(
{
    "str": "abc",
    "int": 30,
    "vec_float": [1.2,2,3.2,4,5], 
    "map": {"a":10, "b":20, "c":30},
    "bool": true
}
)"_json;


for(auto [k,v] : j3.items())
{
    // cout_("key:{}\tvalue:{}{}" , k, v, LR);
    //cout << "key:" << k << "\tvalue:" << v << LR;
    cout_("key:{}\tvalue type:{}\tvalue: {} {}", k, v.type_name(), v.dump(), LR);
}
cout << LR;
for(auto e: j3)
{
    cout << e << endl;
}
auto j4 = j3; 

j3["bool"] = false;
cout_("j3 size: {}{}", j3.size(), LR);

if (j3.contains("map"))
{
    cout_("j3 map size: {}{}", j3["map"].size(), LR);
}
else
{
    cout_("no contain map_ key{}", LR);
}



assert(j3 != j4); 

ns::person p1 = {"abc", "addr_abc", 30};
j1 = p1;
ns::person p2 = j1; 

string j1_name_str = j1["name"];
assert(p1.name == p2.name && p1.address == p2.address && p1.age == p2.age);

j1["name"] = vector<int>{ 1,2,3,4,5 };
j1["address"] = { c_mss({ {"abc","def"}}), c_msi({{"abc",3333}}) };
j1["obj"] = { {"a", 10}, {"b", 20}  };

cout << j1.dump(4) << endl;
j1["address"].clear(); 

cout << j1.dump(4) << endl;

j1.erase("name");  // key name is removed

cout << "___ " << LR; 
cout << j1.dump(4) << endl;
assert(j1["address"].empty());
// key "name" is not exist , assert
assert(j1.find("name") == j1.end());

ci.fs_write_json("1.json", j1);
auto j1_ = ci.fs_read_json("1.json");
assert(j1==j1_); 


vector<int> vi(9,0xff); 
json jv = vi; 
cout << jv.dump() << endl;


j1 = R"(
{
    "a": 1,
    "b": [2,3,4],
    "c": {"d": 5, "e": 6}
}
)"_json;

int  j1_b_1_pointer =  j1["/b/1"_json_pointer];
// int  & j1_b_1_pointer_ =  j1["/b/1"_json_pointer]; // error 
// j1_b_1_pointer = 999; 
cout << j1_b_1_pointer <<endl; 
j1["/b/1"_json_pointer] = j1_b_1_pointer + 1e6;
cout << j1.dump(4) << endl;

json j1_patch = R"(
[
  { "op": "replace", "path": "/b/2", "value": 9999 },
  { "op": "add", "path": "/c", "value": {"new:":"world"} },
  { "op": "remove", "path": "/a"}
]
)"_json;

json j_apply_patch = j1.patch(j1_patch);
cout << j_apply_patch.dump(4) << endl;

cout << json::diff(j1, j_apply_patch).dump(4) << endl;



// a JSON value
json j_document = R"({
  "a": "b",
  "c": {
    "d": "e",
    "f": "g"
  }
})"_json;

// a patch
json j_patch = R"({
  "a":"z",
  "c": {
    "f": null
  }
})"_json;

// apply the patch
j_document.merge_patch(j_patch); 
// {
//  "a": "z",
//  "c": {
//    "d": "e"
//  }
// }

cout << j_document << endl; 


#endif 

#if 0

    json a;
	a["name"] = "abc"; 
	a["age"] = 30; 

	a["vec"] = { 1,2,3,4,5 };

    

	cout << a.dump(4) << endl;

    json j0 = { {"a",4}, {"b",5}, {"c","cvalue"} };

	cout << j0.dump(4) << endl;

	string jsonstr = R"(
{
    "a": 4,
    "b": 5,
    "c": "cvalue"
}
)";

    json j1 = json::parse(jsonstr);

    cout << j1.dump(4) << endl; 

    json j2 = R"(
{"a": 4, "b": 5, "c": "cvalue"}
)"_json; 

	cout << j2.dump(4) << endl;




	pair<string, int> p1 = { "abc", 10 };

	cout << p1.first << "\t" << p1.second << endl;


    // ci.fs_mkdir("./a/b/c/d");
    mss mm ={
        {"a", "10"},
        {"b", "20"},
        {"c", "30"}
	};
    
    for(auto & [k,v]: mm)
    {
      
        v = v + v;
        cout_("{}:{}{}", k, v, LR);
     
	}


    cout_("{}{}", ci.vec_group(ci.R(0,10), 3) , "\n");
    cout_("{}{}", ci.vec_flatten(ci.vec_group(ci.R(0,10), 3)) , "\n");

    vector<int> vi={3,3,4,5,6,4,5,3}; 

    auto vi_1 = ci.vec_range(vi,3);
	auto vi_2 = ci.vec_range(vi, 3, 8);
    assert(vi_1 == vi_2); 


    auto vi_0 = vi; 

    cout_("v==v? {} {} {}", ci.vec_match(vi_0,vi), vi_0 == vi, "\n");

    vector<int> vi_repeat = ci.vec_repeat(vi, 3); 

    cout_("{}{}{}", vi_repeat, ci.vec_rmdup(vi_repeat), "\n");
    cout << ci.str_repeat("abc_",4 ) << endl;;
    auto str_all = ci.str_repeat("abc_",4 );
    auto vstr = ci.split(str_all, "_"); 

    cout_("{}{}", ci.vec_match(vstr, m(R"(A)", "i")), "\n");

    

#endif 
#if 0
    vector<int> vi = { 1,2,3,4,8,-10,4,5,6,7 };
    auto [val, idx] = ci.vec_min(vi);
	cout << val << "\t" << idx << endl;


    unordered_map<string, int> mm = {
        {"a", 10},
        {"b", -5},
        {"c", 3},
        {"d", 8}
    };

    auto k_ = ci.map_keys(mm); 
    auto v_ = ci.map_values(mm);

    

    cout_("{}{}", ci.vec_to_table_templ( k_, v_ ),  "\n");
    auto b0 = ci.map_has(mm, string("a"));

     cout_("map has {}{}{}", "a", b0 ? " true" : "false", "\n");

     auto v__=  ci.vec_map(v_, [](int v) { return v * 2; });

     auto k__= ci.vec_grep(k_, [](string s) { return s != "b"; });
     cout_("{}{}", ci.vec_to_table_templ( k__ , v__), "\n");


     cout_("rand {:^11}{:^11}{:^11}{}{}", ci.rand_int(0, 3),  ci.rand_float(), ci.rand_pick(v_), ci.vec_shuffle(v_),  "\n");

     cout_("{}{}", ci.vec_combine(v_, v__), "\n");


#endif 
#if 0
    auto vstr = ci.read_to_vec("rel.bat");
    
	//cout << ci.vec_to_table({ vstr }) << endl;

    

	vstr = ci.read_all("rel.bat");


    ci.chomp(vstr); 


	vstr = ci.vec_grep_v_empty(vstr);

    cout_("{}\n", vstr); 

    for (auto& e : vstr)
    {
        // ci.chomp(e);
        
        {
            cout << "___" << e << "___" << endl;
        }
        
    }

    assert(0 == 1);

	string sbin = ci.read_to_str("rel.bat");

    cout << ci.b64(sbin) << endl; 
    cout << ci.b64_d(ci.b64(sbin)) << endl; 


 




#endif 
#if 0


    // pad table example
    vector<string> vstr = { "apple", "banana", "apricot", "cherry", "avocado" };
    vector<float> vf = { 0.1f, 0.2512f, 0.5f, 0.75f, 0.9f };
    vector<float> vf2 = { 0.1f, 0.2512f, 0.5f, 0.75f, 0.9f };
    auto vstr_1 = vstr; 
    auto vstr_2 = vstr; 


    // for(auto i:ci.R(vstr.size()))
    // {
    //     cout_("{:<18}{:<18}{:<18}\n", vstr[i], vstr_1[i], vstr_2[i]);
    // }

    cout << ci.vec_to_table({vstr, vstr_1, vstr_2}) << endl;;
    cout << "_________" << endl; 

    cout << ci.vec_to_table_templ(vstr, vf, vf2) << endl; 
    cout << "_________" << endl; 



    auto ss = ci.join(vstr, "+");
    cout << ss << endl; 

    auto vstr_ = ci.split(ss, "+");
    cout_("{}\n", vstr_);



#endif 

#if 0

    vector<string> vstr = { "apple", "banana", "apricot", "cherry", "avocado" };
    cout << ci.join(vstr, "+"); 

#endif 
#if 0
    cout << (string("abcd").substr(2)) << endl;

    const char* s = "0123456789"; 

    string ss = string(s+3, 4); 
    string sss = string(s + 3, s + 8); 

    cout << sss<< endl; 

#endif 
#if 0

    // lambda0_ start

    auto process_e_csv_folder = [](string dn, int flag_show = 0)
    {

        float normal_thres = 0.75f;
        float markpoint_add_val = 1.5f;
        int sz_time_index = 0; 

        // lambda_ start

        auto cut_csv_to_file = [](const std::vector<std::string>& data, const std::string& outputFilename, size_t startIndex, size_t endIndex) -> bool
        {
            // Validate indices
            if (startIndex > endIndex)
            {
                throw std::invalid_argument("startIndex must be <= endIndex");
            }

            if (endIndex >= data.size())
            {
                throw std::out_of_range("endIndex exceeds vector size");
            }

            // Create parent directories recursively if they don't exist
            fs::path filePath(outputFilename);
            if (filePath.has_parent_path())
            {
                fs::path parentPath = filePath.parent_path();
                if (!fs::exists(parentPath))
                {
                    try
                    {
                        fs::create_directories(parentPath);
                    }
                    catch (const fs::filesystem_error& e)
                    {
                        return false;
                    }
                }
            }

            // Open output file
            std::ofstream outFile(outputFilename);
            if (!outFile.is_open())
            {
                return false;
            }

            // Write the range [startIndex, endIndex] to file
            for (size_t i = startIndex; i <= endIndex; ++i)
            {
                outFile << data[i] << std::endl;
            }

            outFile.close();

            // output info
            cimg ci;
            auto old_filenamecsv = ci.replace_str(outputFilename, "raw_data_cut", "raw_data"); 


            auto log_msg = s_("{},{},{},{}", outputFilename,old_filenamecsv,startIndex,endIndex);

            cout << log_msg << endl;
            ci.fcoutln(g_arg_map["--log"], log_msg, "ios::app"); // append mode

            return true;
        };



        auto seed_point_wide_range = [](string fn, vector<float>& v, int seedidx, string direct, int cnt_thres = 2, int win_sz_thres = 10)
        {

            int ret_idx = seedidx;
            const float zero_thres = 0.15f;

            struct filter_para
            {
                int cnt_thres;
                int win_sz_thres;
                float zero_thres;
            };

            filter_para l_p = { 8, win_sz_thres, 0.11 };
            filter_para r_p = { 3, win_sz_thres / 2, 0.32 };

            if (rex(fn) % m(R"(type_4)"))
            {
                r_p = { 4, win_sz_thres, 0.32 };
            }

            if (rex(fn) % m(R"(type_5)"))
            {
                l_p = { 11, win_sz_thres*2, 0.04 };
                r_p = { 5, win_sz_thres/2, 0.32 };
            }


            if (rex(fn) % m(R"(type_8)"))
            {
                r_p = { 4, win_sz_thres , 0.32 };
            }


            if (rex(fn) % m(R"(type_9)"))
            {
                r_p = { 5, win_sz_thres / 2 , 0.32 };
            }


            if (direct == "left")
            {

                int cnt = 0;
                int prev_idx = seedidx;
                for (int i = seedidx; i >= 0; i -= l_p.win_sz_thres)
                {
                    auto ev = v[i];
                    if (ev < l_p.zero_thres)
                    {

                        prev_idx = i;
                        cnt++;
                    }
                    else
                    {
                        cnt = 0;
                    }


                    if (cnt > l_p.cnt_thres)
                    {
                        break;
                    }
                    ret_idx = i;
                }


                ret_idx += l_p.win_sz_thres * (cnt_thres - 1); // to avoid edge case
            }

            if (direct == "right")
            {
                int cnt = 0;
                int prev_idx = seedidx;
                for (int i = seedidx; i < v.size() - 1; i += r_p.win_sz_thres)
                {
                    auto ev = v[i];
                    if (ev < r_p.zero_thres)
                    {

                        prev_idx = i;
                        cnt++;
                    }
                    else
                    {
                        cnt = 0;
                    }


                    if (cnt > r_p.cnt_thres)
                    {
                        break;
                    }

                    ret_idx = i;

                }



            }

            return ret_idx;

        };

        auto assert_input_folder = [](string dn)
        {
            if (rex(dn) % m(R"(raw_data)"))
            {
                // ok
            }
            else
            {
                assertx("error, please ensure input folder contain raw_data", 0 == 1);
            }

            if (!std::filesystem::exists(dn))
            {
                assertx("error, not exist dn", 0 == 1);
            }

            // Check if it's a directory
            if (!std::filesystem::is_directory(dn))
            {
                cerr << "Error: Path is not a directory: " << dn << endl;
                return false;
            }


        };

        auto read_to_csvdata = [&normal_thres, &markpoint_add_val, &sz_time_index](vector<string> vfn_matched) -> unordered_map<string, csv_data>
        {

            assertx("vfn_matched size > 0", vfn_matched.size() > 0);
            cimg ci;

            if (rex(vfn_matched[0]) % m(R"(type_3)"))
            {
                normal_thres = 0.41f;
                markpoint_add_val = 5;
            }


            unordered_map<string, csv_data> csvdata;

            int & sz = sz_time_index;

            for (auto efn : vfn_matched)
            {
                //cout << efn << endl;

                auto& efc = csvdata[efn].efc;
                efc = ci.read_to_vec(efn);
                auto& e_csv_data = csvdata[efn];

                e_csv_data.l.reserve(efc.size());
                e_csv_data.r.reserve(efc.size());



                for (auto eline : efc)
                {
                    auto v_ele = ci.split(eline, ",");
                    auto leftv = std::atof(v_ele[0].c_str());
                    auto rightv = std::atof(v_ele[1].c_str());

                    e_csv_data.l.push_back(std::abs(leftv));
                    e_csv_data.r.push_back(std::abs(rightv));
                    e_csv_data.l_r0.push_back(leftv);
                    e_csv_data.r_r0.push_back(rightv);

                }


                e_csv_data.r = ci.vec_norm_by_minmax(e_csv_data.r);

                // cout << endl; 

                sz = e_csv_data.l.size();
            }

            return csvdata;
        };


        auto cal_peak_ref = [&normal_thres,&markpoint_add_val,  &sz_time_index, &flag_show](unordered_map<string, csv_data>& csvdata)
        {
            // loop to see all csvdata value
            cimg ci; 
            vector<float> v_sum_r = vector<float>(sz_time_index, 0.0);


            for (int i = 0; i < v_sum_r.size(); i++)
            {

                for (auto& e_pair : csvdata)
                {
                    auto& e_csv_data = e_pair.second;
                    v_sum_r[i] += e_csv_data.r[i];
                }

            }



            for (auto& e : v_sum_r)
            {
                e = abs(e);
            }

            v_sum_r = ci.vec_smooth(v_sum_r, 20);
            v_sum_r = ci.vec_smooth(v_sum_r, 7);



            // get min , max value from v_sum_r

            // v_sum_r = min_max_vector(v_sum_r);
            v_sum_r = ci.vec_norm_by_minmax(v_sum_r);

            float v_min = *std::min_element(v_sum_r.begin(), v_sum_r.end());
            float v_max = *std::max_element(v_sum_r.begin(), v_sum_r.end());





            auto maxIndex = std::distance(v_sum_r.begin(), std::max_element(v_sum_r.begin(), v_sum_r.end()));

            auto peak_valley_idx = -1;
            float maxpoint_ext_factor = 1.1f;
            for (auto i = 0; i < maxIndex * maxpoint_ext_factor; i++)
            {
                if (v_sum_r[i] > normal_thres)
                {
                    peak_valley_idx = i;
                    break;
                }
            }

            auto v_sum_copy = v_sum_r;


            vector<float> v_sum_tmp;

            v_sum_copy[peak_valley_idx] += markpoint_add_val;
            std::copy(v_sum_copy.begin() + peak_valley_idx - 100, v_sum_copy.begin() + peak_valley_idx + 200, std::back_inserter(v_sum_tmp));




            if (flag_show)  ci.P(v_sum_tmp, true);


            return peak_valley_idx;
        };


        // lambda_ end

        cimg ci; 


        assert_input_folder(dn);

        auto vfn_matched = ci.get_folder_content(dn, R"(\.csv$)", "files", false);


        // for(auto & efn: fn_list) { efn = ci.norm_path(efn); }


        auto log_msg = s_( "====================");
        cout << log_msg << endl;
        ci.fcoutln(g_arg_map["--log"], log_msg, "ios::app"); // append mode


        auto csvdata = read_to_csvdata(vfn_matched);


        auto peak_valley_idx = cal_peak_ref(csvdata);

        //cal_range_ultra(csvdata, peak_valley_idx);
        cal_range_by_ref(csvdata, peak_valley_idx, normal_thres, markpoint_add_val, flag_show, seed_point_wide_range);


        for (auto & epair : csvdata)
        {
            auto fn = epair.first;
            auto& e_csv_data = epair.second;

            auto efn_cut = ci.replace_str(fn, "raw_data", "raw_data_cut");

            cut_csv_to_file(e_csv_data.efc, efn_cut, e_csv_data.peak_valley_idx_left, e_csv_data.peak_valley_idx_right); 

        }

    };

    // lambda0_ end




    // ci.P(v_sum_r, true);



    // --folder ./t1/raw_data
    //

    // if exist ./log.log, then delete it!

    unordered_map<string,string> arg_map_default = 
    {
        {   "--folder", "./t1/raw_data"   },
        {   "--log", "./cut_raw_data.log" },
        {   "--filter", R"()" },
    };



            string help_msg = s_(
                    R"(
Usage: program [options]

Options:
{}
\t--folder {}\t\tSpecify the input folder path.
\t--filter {}\t\tfilter the  folder path.
)",

                    "<this_exe.exe>",
                    arg_map_default["--folder"],
                    arg_map_default["--filter"]
                    );


    g_arg_map = ci.parse_args(argc, argv, arg_map_default, help_msg);

    cout_ ("{}\n", g_arg_map); 




    if (std::filesystem::exists(g_arg_map["--log"]))
    {
        std::filesystem::remove(g_arg_map["--log"]);
    }


    auto folder_raw_data = s_("{}", g_arg_map["--folder"]);



    cout << "filter:\t" << g_arg_map["--filter"] << endl;


    auto rex_filter_r = rex(g_arg_map["--filter"]);

    if (g_arg_map["--filter"].empty())
    {
        rex_filter_r = rex(R"(type_\d.202[5678]_)"); 
    }
    else
    {
        rex_filter_r % s(R"(\W)", R"(.)", "g");
    }


    auto v_csv_folder = ci.get_folder_content(folder_raw_data, rex_filter_r.str(), "folder", true);

    // auto esrc_fn = "d:/jd/t/git/pulse_peak_find/t1/raw_data/type_9/2025_10_12/2025_10_12_01_sensor_VTin1_o.csv";
    // auto esave_fn = ci.replace_str(esrc_fn, "raw_data", "raw_data_cut");





    for (auto & e_csv_folder : v_csv_folder)
    {
        e_csv_folder = ci.norm_path(e_csv_folder);


        if (!g_arg_map["--filter"].empty())
        {

            auto rex_filter_r = rex(g_arg_map["--filter"]);

            rex_filter_r % s(R"(\W)", R"(.)", "g");

            //  cout << e_csv_folder << endl;  


            if (rex(e_csv_folder) % m(rex_filter_r.str()))
            {
                // ok
            }
            else
            {
                continue;
            }
        }

        auto log_msg = s_("process folder:\t{}", e_csv_folder);
        cout << log_msg << endl;
        ci.fcoutln(g_arg_map["--log"], log_msg, "ios::app"); // append mode

        process_e_csv_folder(e_csv_folder);
        cout << ""<<endl;
        //break;
    }

    // cout log path absolute path
    cout << s_("log path:\t{}", std::filesystem::absolute(g_arg_map["--log"]).string()) << endl;


    // end TODO

#endif 

#if 0

    vector<float> va = {4,5,5,8.8888}; 

    auto va_s = ci.serial_p_2_str(va.data(), va.size()* sizeof(float));
    ci.str_to_bin_file("va.bin", va_s); 
    auto va_new_s = ci.read_to_str("va.bin");

    vector<float> va_copy ={}; 
    va_copy.resize(va_new_s.size() / sizeof(float));

    memcpy(va_copy.data(), va_new_s.data(), va_new_s.size());

    cout_("{}\n", va_copy);


    

    struct abc {
        int x;
        int y;
        int z; 
        float f0;
        double vd[22];
    };





    abc a1;
    a1.f0 = 99.99f;
    auto a1_s = ci.serial_struct_2_str(a1);
    
    ci.str_to_bin_file("a1.bin", a1_s);


    auto a1_str = ci.read_to_str("a1.bin");

    abc a2 = ci.str_to_struct<abc>(a1_str); 

    cout << a2.f0 << endl;

    string s="hello, world!";
    ci.str_to_bin_file("d:/jd/t/hello.bin", s);
    auto s2 = ci.read_to_str("d:/jd/t/hello.bin");
    cout << s2 << endl;


    vector<abc> vabc =
    {
        {1,2,3,4.4f,{0}},
        {11,22,33,44.44f,{0}},
        {111,222,333,444.44f,{0}}
    
    };

    string vabc_s = ci.serial_p_2_str(vabc.data(), vabc.size() * sizeof(abc));


    ci.str_to_bin_file("vabc.bin", vabc_s);

    auto vabc_s2 = ci.read_to_str("vabc.bin");
    vector<abc> vabc2;
    vabc2.resize(vabc_s2.size() / sizeof(abc));
    memcpy(vabc2.data(), vabc_s2.data(), vabc_s2.size());

    for(auto &eabc : vabc2)
    {
        cout_("{}, {}, {}, {}\n", eabc.x, eabc.y, eabc.z, eabc.f0);
    }

    cout_("{}\n", vabc2);

#endif 
#if 0

    auto ans_cmd = ci.run_cmd("date /T"); 

    cout<< ans_cmd["stdout"] << endl; 
    for (auto &ep: ans_cmd)
    {
        cout_("{},{}\n", ep.first, ep.second);
    }

#endif 
#if 0

    ci.fs_touch("./a/b/c/d.txt"); 
    ci.fcoutln("./a/b/c/d.txt", "hello world!", "ios::app");

    cout <<  ci.fs_du("./a/b/c/d.txt") << endl; 

    cout <<    ci.fs_f("./a/b/c/d.txt") << endl; 
    cout << ci.fs_d("./a/b/c/d.txt") << endl;
    // cout << ci.fs_stat("./a/b/c/d.txt") << endl;
    auto ans_stat = ci.fs_stat("./a/b/c/d.txt");

    for(auto & e_stat : ans_stat)
    {
        cout << e_stat.first << " : " << e_stat.second << endl; 
    }

    auto du_dir = ci.fs_du("./a");

    ci.fs_cp("./a/b/c/d.txt", "./a/b/c/d_copy.txt");
    ci.fs_cp("./a/b/c/d.txt", "./a/b/c/d_copy_.txt");
    ci.fs_mv("./a/b/c/d_copy_.txt", "./a/b/c/d_moved.txt");

    cout << "mv src: " <<  ci.fs_f("./a/b/c/d_copy_.txt") << endl;

    cout << "mv : " <<  ci.fs_f("./a/b/c/d_moved.txt") << endl;


    cout << ci.fs_du("./a/b/c/d_copy.txt") << endl;

    ci.fs_cp("./a", "./a_copy");
    cout << ci.fs_du("./a_copy") << endl;

    cout_("ls:{}\n", ci.fs_ls("./a_copy"));
    cout_("ls_r:{}\n", ci.fs_ls_r("./a_copy", R"(mov\w)"));


    ci.fs_rm("./a/b/c/d.txt");
    ci.fs_rm("./a_copy"); 

#endif 
#if 0

    ci.r_i("d:/jd/t/1.jpg"); 
    ci.s_i(); 

#endif 
#if 0

	auto ts0 = ci.ts0();
	ci.td_sleep(2.2);

	auto ts1 = ci.ts1(ts0);



    // cout << ts1.s_d << endl; 

#endif 

#if 0
cout << ci.endwith("abc.svs", ".svs") << endl; 
cout << ci.toupper(".abC") << endl;
cout << ci.tolower(".abC") << endl;

ci.read_img("d:/jd/t/h24193301009.jpg"); 
auto imghist = ci.hist_img(ci.img);
ci.img = imghist;
ci.s_i();


// auto orig_size = ci.img.size();

// auto img = ci.pad2square(ci.img);

// auto img_o = ci.reverse_pad2square(img, orig_size);

// ci.img = img_o;
// ci.s_i(); 


#endif 

#if 0


    auto t_fun_0 = [](){
    cout << "t_fun_0" << endl;

    auto t_fun_00 = [](){
        cout << "t_fun_00" << endl;
    };

    auto t_fun_01 = [&t_fun_00](){
        t_fun_00();
        cout << "t_fun_01" << endl;
    };


    t_fun_01();

    };



    t_fun_0();
#endif     
#if 0

    vector<cv::Mat> vimg; 

    for (auto i : ci.R(1, 10))
    {
    
		cout << i << endl;
        auto fn = s_("d:/jd/t/git/pulse_peak_find/cut_type_{}_2025_10_09.png",i);
        ci.read_img(fn); 
        ci.resize(0.6);
        vimg.push_back(ci.img.clone()); 
    }

    auto vimg_all = ci.vconcat(vimg); 

    auto fullpathimg = ci.abs_path("./vimg_all.jpg"); 

    cv::imwrite(fullpathimg, vimg_all); 
    cout << "full img path: " << fullpathimg << endl;
#endif 

#if 0
    string myabc = "\t\t\tabc"; 

	myabc = ci.replace_str(myabc, "Ab", "_TAB_", "i");

	cout << myabc << endl;
#endif 

#if 0


            unordered_map<string, string> arg_map_default = {
                {"--folder", "./t0/raw_data"},
                {"--filter", R"(202[5-9].*\.csv)"},
                {"--log", "./log.log"},
            };

            string help_msg = s_(
                    R"(opt:
\t{},\n\t{}\n\t{}
)",
                    "abc","def", 342
                    ); 


    help_msg = s_("\n\t{},\n\t{},\n\t{}", "abc","def", 342);

    // unordered_map<string, string> map_ak_av = ci.parse_args(argc, argv);
    auto map_ak_av = ci.parse_and_verify_args(argc, argv, arg_map_default, help_msg);
    cout_("{}\n", map_ak_av); 


#endif 



    // cut_type_9_2025_10_09.png
    

#if 0

    cout << "get folder list test:" << endl;
    auto vf = ci.get_folder_content("d:/jd/t/git/pulse_peak_find/t1/raw_data", R"(raw_data.*type_1.*2025)", "file",  true);
    // vf = ci.get_folder_content("d:/jd/t/git/pulse_peak_find/t1/raw_data", R"(type_1)", "all",  true);
    for (auto& ename : vf)
    {
        if (rex(ename) % m(R"(_09)"))
        {
            cout << ename << endl;
        }
    }



#endif 



#if 0
    vector<cv::Mat> vimg = {


    };


    for (auto i : ci_0.R(9))
    {
        cout << i << endl;

        auto efn = s_("d:/jd/t/git/pulse_peak_find/type_{}.png", i + 1); 
        ci.read_img(efn);

        ci.resize(0.4);
        ci.puttext(s_("type_{}", i + 1));

        vimg.push_back(ci.img.clone());
    }


    auto img_big = ci.vconcat(vimg, 3);

    cv::imwrite("d:/jd/t/git/pulse_peak_find/all_type_10_02.jpg", img_big);


#endif 



#if 0
    vector<int> a = {}; 

    cout << a.empty() << endl; 

    vector<int> b = { 1,2,3 }; 
    cout << b.empty() << endl; 

#endif 

#if 0

    ci.read_img("d:/jd/t/git/WsiCtl/build/t0/1.jpg"); 

    cv::Mat imgcut = ci.img({ 444,888 }, { 288,999 }); 
    // ci.s_i(imgcut);

    cv::imwrite("d:/jd/t/1.jpg", imgcut); 


    ci_0.img = imgcut.clone(); 

    ci_0.cvtcolor("GRAY"); 

    auto gray = ci_0.img.clone();


    ci_0.threshold(111, 255);

    ci_0.img = ~ci_0.img; 

    auto img_t = ci_0.img.clone(); 
    img_t.setTo(0);



    auto img_contour = ci_0.find_contours(); 

    auto v_con_roi = ci_0.v_contours;
    v_con_roi.clear();


    for(auto &ec : ci_0.v_contours)
    {
        // get shape factor for each contour
        auto area = cv::contourArea(ec);
        auto peri = cv::arcLength(ec, true);
        auto shape_factor = 4 * CV_PI * area / (peri * peri + 1e-5);

        // cout << shape_factor << endl;

        if (shape_factor > 0.71f && area > 15*15)
        {
            v_con_roi.push_back(ec);
        }
    }

    auto v_con_roi_pick = v_con_roi;
    v_con_roi_pick.clear();

    for(int i=0;i<v_con_roi.size();i++)
    {
        auto e_c = v_con_roi[i];

        if (i == 3)
        {
            v_con_roi_pick.push_back(e_c);
        }

    }


    // auto img_mask_t = ci.img_contour_mask.clone().setTo(0);

    auto v_con_roi_pick_r0 = v_con_roi_pick; 
    auto img_t_r0 = img_t.clone();



    for (auto offset : ci.R(-5, 6))
    {
        // offset = 2;
        v_con_roi_pick = v_con_roi_pick_r0;

        for (auto& ec : v_con_roi_pick)
        {
            for (auto& ep : ec)
            {
                ep.x += offset;
                ep.y += 0;
            }

        }
        img_t = img_t_r0.clone();

        cv::drawContours(img_t, v_con_roi_pick, -1, cv::Scalar(255), cv::FILLED);

        imwrite("d:/jd/t/3.jpg", img_t);

        cv::Mat gray_t = gray & img_t;

        auto bb = cv::boundingRect(v_con_roi_pick[0]);
        // loop gray_t by bb 

        int sum = 0;
        int cnt = 0;
        for (int r = bb.y; r < bb.y + bb.height; r++)
        {
            for (int c = bb.x; c < bb.x + bb.width; c++)
            {
                auto& e_gray = gray_t.ptr<uchar>(r)[c];
                if (e_gray == 0)
                {
                    continue;
                }

                sum += e_gray;
                cnt += 1;

            }
        }

        float avg = sum * 1.0f / cnt;

        std::cout << "offset:" << offset << ", avg:" << avg << endl;

    }












    //ci.s_i(img_contour);   


#endif 

#if 0

    ci.read_img("d:/jd/t/t0/000000000009.jpg"); 

    // ci.s_i();
    int rows = ci.img.rows;
    int cols = ci.img.cols;


    cout << "rows:" << rows << ", cols:" << cols << endl;

    auto v_str = ci.read_to_vec("d:/jd/t/t0/000000000009.txt");

    cout << v_str[0] << endl; 

    auto v_x_y = ci.split(v_str[3], " ");

    vector<cv::Point2i> v_pt; 

    v_pt.reserve(v_x_y.size() / 2 + 1);


    for(int i=1;i<v_x_y.size()-1;i+=2)
    {
        v_pt.push_back(
                cv::Point2i( stof(v_x_y[i]) * cols , stof(v_x_y[i+1]) * rows)
                );
    }


    cout << endl;

    auto img = ci.img.clone(); 
    img.setTo(0);

    for(auto &ep : v_pt)
    {
        cout << ep.x << "," << ep.y << endl; 
        img.ptr<v3b>(ep.y)[ep.x] = v3b(255,255,255);
    }


    ci.s_i(img); 








#endif 


#if 0

    rex  ids("AB3C");

    rex_match idm = m(R"(B)");

    bool ism = ids % idm;
    cout << ism << endl;

    ids % s(R"(\d)", R"(__)");


    cout << ids.str() << endl;


    rex ids_2("AB3C5dc"); 

    rex_match mresult = m(R"(^.*(\d)\D(\d).*$)");
    ids_2 % mresult;
    cout << mresult.count() << ","<< mresult[2] << "__" << mresult[1] << endl;

    //cout << mresult << endl;







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

    // Æ½ï¿½ï¿½ ci.img Ö±ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ img_dst
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
    // img_square = pad2square(img, cv::Scalar(0, 0, 0)); 
    // reverse size from img_square to old img 
    // img = rev_pad2square(img_square,img_old_size); 








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



            // ï¿½ï¿½ï¿½Ã»Ò¶ï¿½Í¼ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½?
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

    vector<Mat> vimg; // ï¿½Þ¸ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Îªï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½

    for (int e : ci.R(0, N)) // Ê¹ï¿½ï¿½ std::views::iota ï¿½ï¿½ï¿½É·ï¿½Î§
    {
        cout << e << endl;
        auto e_fn = s_("d:/jd/t/p/masked_{}.png", e);

        ci.read_img(e_fn); 

        vimg.push_back(ci.img.clone()); // È·ï¿½ï¿½ vimg ï¿½ï¿½Ò»ï¿½ï¿½ï¿½ï¿½Ð§ï¿½ï¿½ vector
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
    auto step = img.step[0]; // ï¿½Þ¸ï¿½Îªï¿½ï¿½È¡ï¿½ï¿½Ò»Í¨ï¿½ï¿½ï¿½Ä²ï¿½ï¿½ï¿½

    cout << step << endl;

    // get widthstep for img 
    int widthstep = img.step; // ï¿½Þ¸ï¿½Îªï¿½ï¿½È¡ï¿½ï¿½Ò»Í¨ï¿½ï¿½ï¿½Ä²ï¿½ï¿½ï¿½

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



    auto vstr = ci.read_to_vec(fn_all_jpg);




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

    // Ê¾ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½  
    std::vector<Point> contour = {
        Point(100, 100), Point(200, 100),
        Point(200, 200), Point(100, 222),  Point(100, 100),
    };

    vector< std::vector<Point>> v_contours {contour}; 


    test_const_contour(v_contours); 


    // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½  
    std::vector<Point> shrunkenContour = shrinkC(contour,0.3);

    // ï¿½ï¿½Í¼ï¿½ï¿½ï¿½Ï»ï¿½ï¿½ï¿½Ô­Ê¼ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??  
    Mat image = Mat::zeros(300, 300, CV_8UC3);



    cv::drawContours(image, v_contours, -1, Scalar(255, 0, 0), cv::FILLED); // Ô­Ê¼ï¿½ï¿½ï¿½ï¿½  

    // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??  
    // polylines(image, shrunkenContour, true, Scalar(0, 255, 0), 2); // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½  


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
            // ï¿½ï¿½ï¿½Êµï¿½ï¿½ï¿½Î»ï¿½ï¿½ï¿½ï¿½ï¿½Ó»ï¿½ï¿½Æ¾ï¿½ï¿½ÎµÄ´ï¿½ï¿½ï¿½
            // cv::rectangle(ci.img, bb, cv::Scalar(0, 0, 0), 1); // Ê¹ï¿½Ãºï¿½É«ï¿½ï¿½ï¿½Æ¾ï¿½ï¿½ï¿½

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

        // cv::rectangle(ci.img, bb, cv::Scalar(0, 0, 0), 1); // Ê¹ï¿½Ãºï¿½É«ï¿½ï¿½ï¿½Æ¾ï¿½ï¿½ï¿½
        // draw rectangle and filled with 255
        cv::rectangle(ci.img, bb, cv::Scalar(255), cv::FILLED);


    }


    // ci.threshold(254); 

    cv::threshold(img, img, 254, 255, cv::THRESH_BINARY);

    // ci.img; 

    // ï¿½ï¿½Í¼ci.img ï¿½ï¿½Ò»ï¿½Î¿ï¿½ï¿½ï¿½ï¿½ã¡£
    //int kernelSize = 7;
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    //// ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
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

    auto vfn = ci.read_to_vec(fn);
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

    // ï¿½ï¿½ï¿½ï¿½Ö±ï¿½ï¿½ y0 ï¿½ï¿½Öµ  
    float y0 = 150.0; // ï¿½ï¿½ï¿½ï¿½ï¿?? y ï¿½ï¿½ï¿½ï¿½Öµ  

    // ï¿½ï¿½ï¿½ã½»ï¿½ï¿½  
    std::vector<Point2f> intersectionPoints = getIntersectionPoints(c0, y0);

    // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??  
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
    // ï¿½ï¿½ï¿½ï¿½Ã¿ï¿½ï¿½Í¨ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶È·ï¿½Öµ
    std::vector<cv::Mat> gradientMagnitudes;
    for (const auto& channel : channels) {
        cv::Mat Gx, Gy;
        cv::Sobel(channel, Gx, CV_64F, 1, 0, 9, 5.0); // Ë®Æ½ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶ï¿½
        cv::Sobel(channel, Gy, CV_64F, 0, 1, 9, 5.0); // ï¿½ï¿½Ö±ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶ï¿½
        cv::Mat gradientMagnitude;
        cv::magnitude(Gx, Gy, gradientMagnitude);
        gradientMagnitudes.push_back(gradientMagnitude);
    }



    cv::normalize(gradientMagnitudes[0], gradientMagnitudes[0], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(gradientMagnitudes[1], gradientMagnitudes[1], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(gradientMagnitudes[2], gradientMagnitudes[2], 0, 255, cv::NORM_MINMAX, CV_8U);



    // È¡ï¿½Ý¶ï¿½, È¡ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½? * 3
    cv::Mat gradientMagnitude;
    (cv::max)(gradientMagnitudes[0], gradientMagnitudes[1], gradientMagnitude);
    (cv::max)(gradientMagnitude, gradientMagnitudes[2], gradientMagnitude);
    cv::multiply(gradientMagnitude, 3, gradientMagnitude);


    cv::imwrite("d:/jd/t/t0/gradientMagnitude.png", gradientMagnitude);



    std::vector<std::vector<cv::Point>> contours;

    cv::Mat binaryImage;

    for (int i = 70; i <= 150; i += 20)
    {
        // ï¿½ï¿½Öµï¿½ï¿½
        cv::threshold(gradientMagnitude, binaryImage, i, 255, cv::THRESH_BINARY);


        cv::imwrite("d:/jd/t/t0/binaryImage" + to_string(i) + ".png", binaryImage);

        // context->saveMiddleImg(binaryImage, "binaryImage" + TC_Common::tostr(i) + ".jpg");
        // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??
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

    // ï¿½ï¿½ï¿½ï¿½ï¿½Ø¸ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½, Ê¹ï¿½ï¿½mapï¿½Å»ï¿½ï¿½ï¿½ï¿½ï¿½Ð§ï¿½ï¿½
    std::vector<std::vector<cv::Point>> uniqueContours;
    for (const auto& contour : contours) {
        // ï¿½ï¿½ï¿½ãµ±Ç°ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Äµï¿½
        cv::Moments m = cv::moments(contour);
        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);

        bool isDuplicate = false;
        // ï¿½ï¿½ï¿½Ñ±ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È½ï¿½ï¿½ï¿½ï¿½Äµï¿½ï¿½ï¿½ï¿½
        for (size_t i = 0; i < uniqueContours.size(); i++) {
            cv::Moments m2 = cv::moments(uniqueContours[i]);
            cv::Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);

            // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Äµï¿½ï¿½ï¿½ï¿½Ð¡ï¿½ï¿½10ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½,ï¿½ï¿½Îªï¿½ï¿½ï¿½Ø¸ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½, ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
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




            uchar updiff = (vFeatureImgroi[3].ep - vFeatureImgroi[2].ep) / 3.0f * 1.1f; // floodfill diff ï¿½Ï½ï¿½,ï¿½ï¿½ï¿½ï¿½1.1fï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??,ï¿½ï¿½Ð¡ï¿½áµ¼ï¿½ï¿½ï¿½Òµï¿½cell coreÆ«Ð¡ï¿½ï¿½ï¿½ï¿½Ö®

            // cv::floodFill(imgroi, coreMinPos, 255, 0, cv::Scalar::all(10), cv::Scalar::all(updiff), FLOODFILL_FIXED_RANGE);


            // ï¿½ï¿½imgroiï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ÖµÎª255ï¿½Ä²ï¿½ï¿½Ö¿ï¿½ï¿½ï¿½ï¿½ï¿½img_r1ï¿½ï¿½bbï¿½ï¿½ï¿½ï¿½  
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
    assert(img.channels() == 3); // È·ï¿½ï¿½Í¼ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Í¨ï¿½ï¿½ï¿½ï¿½
    vector<cv::Mat> vbgr;;
    cv::split(img, vbgr); // ï¿½ï¿½Í¼ï¿½ï¿½ï¿½ï¿½Îªï¿½ï¿½ï¿½ï¿½Í¨ï¿½ï¿½

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
            // ï¿½ï¿½ï¿½ï¿½Ã¿ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
            uchar pixel_value = bbimg_copy.ptr<uchar>(r)[c];
            // ï¿½ï¿½ï¿½ï¿½Ä³ï¿½Ö²ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Öµï¿½ï¿½ï¿½ï¿½
            if (pixel_value > 66) {
                bbimg_copy.ptr<uchar>(r)[c] = 255; // ï¿½ï¿½ï¿½ï¿½Îªï¿½ï¿½É«
            } else {
                bbimg_copy.ptr<uchar>(r)[c] = 0;   // ï¿½ï¿½ï¿½ï¿½Îªï¿½ï¿½É«
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
        //cv::Sobel(gray, Gx, CV_64F, 1, 0, 3); // Ë®Æ½ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶ï¿½
        //cv::Sobel(gray, Gy, CV_64F, 0, 1, 3); // ï¿½ï¿½Ö±ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶ï¿½
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

            ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä±ï¿½Ö¾ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ä·½Ê½ï¿½ï¿½ï¿½ï¿½ï¿½ÃµÄ±ï¿½Ö¾ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??
            4 : ï¿½ß½ï¿½ï¿½ï¿½ï¿??4 - ï¿½Ú½Ó¡ï¿½
            8 : ï¿½ß½ï¿½ï¿½ï¿½ï¿??8 - ï¿½Ú½Ó¡ï¿½
            FLOODFILL_FIXED_RANGE : ï¿½ï¿½É«Æ¥ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½É«ï¿½ï¿½Î§ï¿½ï¿½ï¿½Ì¶ï¿½ï¿½ï¿½ï¿½Óµï¿½ï¿½ï¿½É«ï¿½ï¿??
            FLOODFILL_MASK_ONLY : Ö»ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä£ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿??
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
    // ï¿½ï¿½ imgroi2_mat ï¿½ï¿½ï¿½Ã»ï¿½ ci.img ï¿½ï¿½ï¿½ï¿½Í¬ ROI

    auto imgroi = ci.img(roi);
    auto imgroi_copy = imgroi.clone();

    imgroi_copy.at<uchar>(0, 11) = 0;
    imgroi_copy.at<uchar>(1, 11) = 0;
    imgroi_copy.at<uchar>(2, 11) = 0;
    imgroi_copy.at<uchar>(3, 11) = 0;


    // Ê¹ï¿½ï¿½ cv::Mat ï¿½ï¿½ï¿½ï¿½ cv::MatExpr
    cv::Mat imgroi2_mat = imgroi & imgroi_copy;

    // Ê¹ï¿½ï¿½ .at<uchar>() ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Öµ
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
        // cv::drawContours(mask, contours_roi, static_cast<int>(i), cv::Scalar(255), cv::FILLED); // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??


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
        cv::drawContours(mask, contours_roi, static_cast<int>(i), cv::Scalar(255), cv::FILLED); // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿??

        // get boundbox for contours_roi[i]
        cv::Rect bbox = cv::boundingRect(contours_roi[i]);



        deal_bbox_diff(diffimg, mask, d_diffimg, bbox,2.0f ,D_DIFF_THRES);

        // deal_bbox_diff(diffimg, mask, diffimg, bbox);



        // loop all bbox's pixel value and count the average 



    }
    //ci.s_i(diffimg);

    //diffimg += 100;



    cv::Mat min_kernel; // ï¿½ï¿½ï¿½Ú´æ´¢ï¿½ï¿½Ð¡Öµï¿½ï¿½Í¼ï¿½ï¿½
    cv::Mat temp; // ï¿½ï¿½Ê±Í¼ï¿½ï¿½

    // Ê¹ï¿½ï¿½ 3x3 ï¿½Ä¾ï¿½ï¿½ï¿½ï¿½ï¿½
    cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

    // ï¿½ï¿½Ê¼ï¿½ï¿½ min_kernel
    min_kernel = cv::Mat::zeros(d_diffimg.size(), d_diffimg.type());

    // ï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½
    for (int r = 1; r < d_diffimg.rows - 1; r++) {
        for (int c = 1; c < d_diffimg.cols - 1; c++) {
            // ï¿½ï¿½È¡ 3x3 ï¿½ï¿½ï¿½ï¿½
            temp = d_diffimg(cv::Rect(c - 1, r - 1, 3, 3));
            // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ð¡Öµ
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



    // Ê¹ï¿½Ã±ï¿½×¼ï¿½ï¿½ï¿½Ðµï¿½ M_PI
    double pi_value = M_PI; // ï¿½ï¿½ï¿½ï¿½Ö±ï¿½ï¿½Ê¹ï¿½ï¿½ 3.14159265358979323846
#endif 

#if 0

    ci.read_img("image_g_r.png"); 
    ci.img;

    cv::Mat binimg; 
    cv::threshold(ci.img, binimg,  155, 255, CV_)


        ci.s_i();












#if 0

    vector<cv::Mat> channels = { ci.img }; 

    // ï¿½ï¿½ï¿½ï¿½Ã¿ï¿½ï¿½Í¨ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶È·ï¿½Öµ
    std::vector<cv::Mat> gradientMagnitudes;
    for (const auto& channel : channels) {
        cv::Mat Gx, Gy;
        cv::Sobel(channel, Gx, CV_64F, 1, 0, 3); // Ë®Æ½ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶ï¿½
        cv::Sobel(channel, Gy, CV_64F, 0, 1, 3); // ï¿½ï¿½Ö±ï¿½ï¿½ï¿½ï¿½ï¿½Ý¶ï¿½
        cv::Mat gradientMagnitude;
        cv::magnitude(Gx, Gy, gradientMagnitude);
        gradientMagnitudes.push_back(gradientMagnitude);
    }


    cv::normalize(gradientMagnitudes[0], gradientMagnitudes[0], 0, 255, cv::NORM_MINMAX, CV_8U);


    // È¡ï¿½Ý¶ï¿½, È¡ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½? * 3
    cv::Mat gradientMagnitude = gradientMagnitudes[0];

    cv::multiply(gradientMagnitude, 3, gradientMagnitude);

    cout << "gradientMagnitude.type() = " << gradientMagnitude.type() << endl;

    cv::imwrite("gradientMagnitude.jpg", gradientMagnitude);

    cv::Mat binaryImage;
    cv::threshold(gradientMagnitude, binaryImage, 70, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);


    // ï¿½ï¿½ï¿½ï¿½Ã¿ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Èµï¿½Í¹ï¿½ï¿½
    std::vector<std::vector<cv::Point>> convexHulls;
    for (const auto& contour : contours) {

        if (!filterContour(contour)) {
            continue;
        }

        convexHulls.push_back(contour);
    }

    // ï¿½ï¿½ï¿½ï¿½É«, ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ò»ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ê¶ï¿½ï¿½Ï¸ï¿½ï¿½ï¿½ï¿½
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
    auto id_s = ci.read_to_str("d:/jd/t/t0/1.jpg"); 
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
    auto ev =  ci.vec_sum(vf);

    cout << ev << endl;

    cout << ci.vec_mean(vf) << endl; 

    cout_("{}\n", ci.vec_diff(vf)) ;

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

        auto vp_vn = ci.vec_combine(vp, vn);


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
        vector<string> vs = ci.read_to_vec(fn);
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

    auto idbin = ci.read_to_str(fn);
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


    int min_distance = 10; // ï¿½ï¿½Ð¡ï¿½ï¿½ï¿½ï¿½  
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

    // ï¿½ï¿½ï¿½ï¿½ HSV Í¨ï¿½ï¿½
    std::vector<cv::Mat> v_hsv;
    cv::split(HSV, v_hsv);





    auto gray = v_hsv[2].clone();
    ci.img = gray; 
    ci.norm_img();


    ci.img = v_hsv[1];
    ci.norm_img();




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


    ci.norm_img();
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
    ci.read_img("D:/jd/t/smb_share/t/test_rgb2y/ï¿½ï¿½Ôµï¿½Ö¸ï¿½/01_03.jpg");


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
    ci.read_img("D:/jd/t/smb_share/t/test_rgb2y/ï¿½ï¿½Ôµï¿½Ö¸ï¿½/01_03.jpg"); 


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

    const char* shm_name = "MySharedMemory"; // Ê¹ï¿½ï¿½ ANSI ï¿½Ö·ï¿½ï¿½ï¿½  
    const int SIZE = 4096;

    // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ú´ï¿½  
    HANDLE shm = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, SIZE, shm_name);
    if (shm == NULL) {
        std::cerr << "Could not create file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ú´ï¿½Ó³ï¿½äµ½ï¿½ï¿½Ç°ï¿½ï¿½ï¿½ÌµÄµï¿½Ö·ï¿½Õ¼ï¿½  
    void* ptr = MapViewOfFile(shm, FILE_MAP_ALL_ACCESS, 0, 0, SIZE);
    if (ptr == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        return 1;
    }

    // Ð´ï¿½ï¿½ï¿½ï¿½ï¿½Ýµï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ú´ï¿½  
    const char* message = "Hello, Shared Memory!";

    // Ê¹ï¿½ï¿½ strcpy_s ï¿½ï¿½ï¿?? strcpy  
    // Ä¿ï¿½ê»ºï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ SIZEï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ê¹ï¿½ï¿½ SIZE ï¿½ï¿½Îªï¿½ï¿½ï¿½ï¿½  
    errno_t err = strcpy_s(static_cast<char*>(ptr), SIZE, message);
    if (err != 0) {
        std::cerr << "Error copying to shared memory: " << err << std::endl;
        return 1;
    }

    // ï¿½ï¿½È¡ï¿½ï¿½ï¿½ï¿½ï¿½Ú´ï¿½ï¿½Ðµï¿½ï¿½ï¿½ï¿½ï¿½  
    std::cout << static_cast<char*>(ptr) << std::endl;

    // ï¿½ï¿½ï¿½ï¿½  
    UnmapViewOfFile(ptr);
    CloseHandle(shm);

    system("pause");

    return 0;

#endif
#if 0


    const char* shm_name = "MySharedMemory";
    const int SIZE = 4096;

    // ï¿½ò¿ª¹ï¿½ï¿½ï¿½ï¿½Ú´ï¿½  
    HANDLE shm = OpenFileMappingA(FILE_MAP_READ, FALSE, shm_name);
    if (shm == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    // Ó³ï¿½ä¹²ï¿½ï¿½ï¿½Ú´æµ½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ÌµÄµï¿½Ö·ï¿½Õ¼ï¿??  
    void* ptr = MapViewOfFile(shm, FILE_MAP_READ, 0, 0, SIZE);
    if (ptr == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(shm);
        return 1;
    }

    // ï¿½ï¿½È¡ï¿½ï¿½ï¿½ï¿½ï¿½Ú´ï¿½ï¿½Ðµï¿½ï¿½ï¿½ï¿½ï¿½  
    std::cout << "Data from shared memory: " << static_cast<char*>(ptr) << std::endl;

    // ï¿½ï¿½ï¿½ï¿½  
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
    vector<string> fc = ci.split(cml, "\n");

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

    int sliderValue = 0; // ??Ê¼ÖµÎª0
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
    // ?Í·???Ô´
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

    string obj = ci.read_to_str("D:/jd/t/di_4431.bin");


    ObjectParaSmall* obj_ptr = (ObjectParaSmall*)obj.data();

    cout << obj_ptr->DnaIndex << endl;

#endif

#if 0

    string fn_string = "D000011411_y.jpg D000011410_y.jpg D000011409_y.jpg D000011408_y.jpg D000011407_y.jpg D000011406_y.jpg D000011405_y.jpg D000011404_y.jpg D000011403_y.jpg D000011402_y.jpg D000011401_y.jpg D000011400_y.jpg D000011311_y.jpg D000011310_y.jpg D000011309_y.jpg D000011308_y.jpg D000011307_y.jpg D000011306_y.jpg D000011305_y.jpg D000011304_y.jpg D000011303_y.jpg D000011302_y.jpg D000011301_y.jpg D000011300_y.jpg D000011211_y.jpg D000011210_y.jpg D000011209_y.jpg D000011208_y.jpg D000011207_y.jpg D000011206_y.jpg D000011205_y.jpg D000011204_y.jpg D000011203_y.jpg D000011202_y.jpg D000011201_y.jpg D000011200_y.jpg D000011111_y.jpg D000011110_y.jpg D000011109_y.jpg D000011108_y.jpg D000011107_y.jpg D000011106_y.jpg D000011105_y.jpg D000011104_y.jpg D000011103_y.jpg D000011102_y.jpg D000011101_y.jpg D000011100_y.jpg D000011011_y.jpg D000011010_y.jpg D000011008_y.jpg D000011007_y.jpg D000011005_y.jpg D000011004_y.jpg D000011002_y.jpg D000011001_y.jpg D000011000_y.jpg D000010911_y.jpg D000010910_y.jpg D000010909_y.jpg D000010908_y.jpg D000010907_y.jpg D000010906_y.jpg D000010905_y.jpg D000010904_y.jpg D000010903_y.jpg D000010902_y.jpg D000010901_y.jpg D000010900_y.jpg D000010811_y.jpg D000010810_y.jpg D000010809_y.jpg D000010808_y.jpg D000010807_y.jpg D000010806_y.jpg D000010805_y.jpg D000010804_y.jpg D000010803_y.jpg D000010802_y.jpg D000010801_y.jpg D000010800_y.jpg D000010711_y.jpg D000010710_y.jpg D000010708_y.jpg D000010707_y.jpg D000010705_y.jpg D000010704_y.jpg D000010702_y.jpg D000010701_y.jpg D000010700_y.jpg D000010611_y.jpg D000010610_y.jpg D000010609_y.jpg D000010608_y.jpg D000010607_y.jpg D000010606_y.jpg D000010605_y.jpg D000010604_y.jpg D000010603_y.jpg D000010602_y.jpg D000010601_y.jpg D000010600_y.jpg D000010511_y.jpg D000010510_y.jpg D000010509_y.jpg D000010508_y.jpg D000010507_y.jpg D000010506_y.jpg D000010505_y.jpg D000010504_y.jpg D000010503_y.jpg D000010502_y.jpg D000010501_y.jpg D000010500_y.jpg D000010411_y.jpg D000010410_y.jpg D000010408_y.jpg D000010407_y.jpg D000010405_y.jpg D000010404_y.jpg D000010402_y.jpg D000010401_y.jpg D000010400_y.jpg D000010311_y.jpg D000010310_y.jpg D000010309_y.jpg D000010308_y.jpg D000010307_y.jpg D000010306_y.jpg D000010305_y.jpg D000010304_y.jpg D000010303_y.jpg D000010302_y.jpg D000010301_y.jpg D000010300_y.jpg D000010211_y.jpg D000010210_y.jpg D000010209_y.jpg D000010208_y.jpg D000010207_y.jpg D000010206_y.jpg D000010205_y.jpg D000010204_y.jpg D000010203_y.jpg D000010202_y.jpg D000010201_y.jpg D000010200_y.jpg D000010111_y.jpg D000010110_y.jpg D000010109_y.jpg D000010108_y.jpg D000010107_y.jpg D000010106_y.jpg D000010105_y.jpg D000010104_y.jpg D000010103_y.jpg D000010102_y.jpg D000010101_y.jpg D000010100_y.jpg D000010011_y.jpg D000010010_y.jpg D000010009_y.jpg D000010008_y.jpg D000010007_y.jpg D000010006_y.jpg D000010005_y.jpg D000010004_y.jpg D000010003_y.jpg D000010002_y.jpg D000010001_y.jpg D000010000_y.jpg D000010403_y.jpg D000010703_y.jpg D000011003_y.jpg D000011006_y.jpg D000011009_y.jpg D000010709_y.jpg D000010409_y.jpg D000010406_y.jpg D000010706_y.jpg";

    auto v_fn = ci.split(fn_string, " ");




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



    // ??Ò»???? 0-255 ??Î§??
    cv::Mat normalizedImage;
    cv::normalize(img_y, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    //cv::normalize(img_y, normalizedImage, 0, 255, cv::NORM_RELATIVE, CV_8U);

    // ×ª??Îª uchar ???Íµ?Í¼??
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

    // Ê¹???Ô¶?????È¨Ïµ??????É«Í¼??×ª??Îª?Ò¶?Í¼??
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

    auto ci_str_ = ci.read_to_str("1.bin");
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

    string  dir = "D:/jd/t/dl/?Ç¶?/g0/";

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

    string  dir = "D:/jd/t/dl/?Ç¶?/g0/";

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
    cv::Mat kernel(3, 3, CV_32S, fpArray); // ??????×ª??Îª OpenCV ?? Mat ??Ê½

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

// ????Ã¿??Í¨?ï¿½ï¿½?Ö±??Í¼
std::vector<cv::Mat> hist(3);
int histSize = 256;
float range[] = { 0, 256 };
const float* histRange = { range };
for (int i = 0; i < 3; i++) {
    cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist[i], 1, &histSize, &histRange);
}

// ????Ö±??Í¼
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

// ???Óº???????
cv::line(histImage, cv::Point(0, hist_h), cv::Point(hist_w, hist_h), cv::Scalar(0, 0, 0), 1, 8, 0);
cv::line(histImage, cv::Point(0, hist_h), cv::Point(0, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
for (int i = 0; i < histSize; i += 32) {
    cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, hist_h - 5), cv::Scalar(0, 0, 0), 1, 8, 0);
    cv::line(histImage, cv::Point(bin_w * i, hist_h), cv::Point(bin_w * i, 0), cv::Scalar(0, 0, 0), 1, 8, 0);
    std::stringstream ss;
    ss << i;
    cv::putText(histImage, ss.str(), cv::Point(bin_w * i, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// ????Ö±??Í¼Í¼??
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
            //ci.img.at<Vec3b>(r, c) = Vec3b(0, 0, 0); // ????Îª????É«
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

auto sbuf2 = ci.read_to_str("1.bin");

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
string frombin = ci.read_to_str("1.bin");

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

string bin_content = ci.read_to_str(fn);
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







auto v_c = ci.read_to_str("d:/jd/t/1.bin");

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
string bin_content = ci.read_to_str(fn);

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

auto vs = ci.split(id_s, ":");
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

