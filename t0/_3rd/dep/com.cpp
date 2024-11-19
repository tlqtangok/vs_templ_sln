//
// Created by jd on 20240517.
//

#if 1
#include "com.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>

#endif 


#if 1
// glo_ cpp start 

int glo::gx = 888;

int glo::argc = 0;
char** glo::argv = nullptr;
string  glo::abs_exe_path = "NULL_PATH";
string  glo::abs_exe_dirpath = "NULL_PATH";
string glo::abs_log_file = "NULL_PATH";
com* glo::pcom = nullptr;

// glo_ cpp end 

// com_ cpp start 
com::com(void)
{
    map_config_k_v = {};
}


string com::get_exe_abs_path()
{

    std::string absolutePath;
    try {
        // 获取当前可执行文件的路径
        std::filesystem::path exePath = std::filesystem::absolute(std::filesystem::path(glo::argv[0]));

        // 将路径转换为绝对路径的字符串
        absolutePath = exePath.string();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    glo::abs_exe_path = absolutePath;
    return absolutePath;
}

string com::get_exe_dir()
{
    std::filesystem::path exe_abs_path_name = get_exe_abs_path();
    string exe_dirname = exe_abs_path_name.parent_path().string();
    glo::abs_exe_dirpath = exe_dirname;
    return exe_dirname;
}

string com::get_current_timestamp()
{
#if 0
    // 获取当前时间
    SYSTEMTIME st;
    GetLocalTime(&st);

    // 创建一个string对象来存储格式化后的时间字符串
    string strTimestamp;

    // 格式化时间字符串
    strTimestamp.Format(_T("%04d%02d%02d_%02d%02d"),
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute);

#endif 

    std::stringstream ss;

    std::time_t now = std::time(nullptr);
    std::tm timeInfo;

    if (localtime_s(&timeInfo, &now) == 0) {
        
        ss << std::put_time(&timeInfo, "%Y%m%d_%H%M");

        //std::cout << ss.str() << std::endl;
    }
    else {
        std::cerr << "Failed to get local time." << std::endl;
    }


    //std::stringstream ss;
   // ss << std::put_time(&timeInfo, "%Y%m%d_%H%M");
    return ss.str();
}

#if 0
std::wstring com::cstring2wstring(string id_cstring)
{
    std::wstring id_wstring(id_cstring.GetString());
    return id_wstring;

}

std::string com::wstring2string(wstring id_wstring)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(id_wstring);
}

std::string com::cstring2string(string id_cstring)
{
    auto id_wstring = cstring2wstring(id_cstring);
    auto id_string = wstring2string(id_wstring);
    return id_string;

}
#endif 



void com::logger(const string & id_s)
{
    const auto F_W = (ios::out | ios::app);

    std::string fn(glo::abs_log_file);

    //wcout << fn << endl; 
    //wcout << fn << endl;

    ofstream if_(fn, F_W);
    if (!if_.is_open()) {
        cout << "- make sure the file path is accessible!" << endl;
    }
    assert(if_.is_open());
    auto ts = get_current_timestamp();
    if_ << "[" << ts << "]" << " " << id_s << endl;
    if_.flush();
    if_.close();
    //cout << fn << endl;
}

void com::m_s(string msg)
{
    //AfxMessageBox(msg);
    cout << msg << endl;
}
void com::glo_init(int argc, char** argv)
{
    //    glo::gx = 999; 
    
    static uint64_t s_glo_init = 0;
    if (s_glo_init++ == 0)
    {
#if 1
        glo::argc = argc;
        glo::argv = argv;
        //glo::argv0 = argv[0]; 
        get_exe_dir();  // glo::abs_exe_dirpath

        glo::abs_log_file = glo::abs_exe_dirpath + "\\" + string(COM_LOG_FILE);
        // cout << "- log file: " << glo::abs_log_file << endl;
        glo::pcom = this;
        //cout << CW2A(glo::abs_log_file) << endl;

        if (file_exist(glo::abs_log_file))
        {
            trim_log_file((int)20e6 /*max_log_file_size*/, 200 /*save_line_num*/);
        }
#endif 
        logger("");
    }

}

string com::d2s(double d_)
{
    string st;
    //st.Format(L"%.2f", (double)d_);
    return st;
}
string com::f2s(float d_)
{
    string st;
    //st.Format(L"%.2f", d_);
    return st;
}
string com::i2s(int d_)
{
    string st;
    //st.Format(L"%d", d_);
    return st;
}

int com::file_exist(string fn)
{
    //CFileStatus status;
    int ret_code = 0; 
    if (std::filesystem::exists(fn))
    {
        ret_code = 1;
    }
    return ret_code;
}

vector<string> com::get_e_str_left_right(string str)
{

    std::vector<std::string> result;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, '=')) {
        // 去除空格
        token.erase(0, token.find_first_not_of(" \t\n\r"));
        token.erase(token.find_last_not_of(" \t\n\r") + 1);
        result.push_back(token);
    }
    return result;

}

int  com::get_file_size(string filePath)
{
    std::uintmax_t fileSize = 0;
    try {
        // 使用 std::filesystem::file_size 函数获取文件大小
        fileSize = std::filesystem::file_size(filePath);
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return fileSize;

}
void com::read_file_2_vec(const string& filePath, std::vector<string>& lines)
{
    if (std::filesystem::exists(filePath)) {
        std::ifstream file(filePath);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                lines.push_back(line);
            }
            file.close();
        }
        else {
            std::cerr << "Failed to open file: " << filePath << std::endl;
        }
    }
    else {
        std::cerr << "File does not exist: " << filePath << std::endl;
    }

}

void com::read_config(string fn_config)
{

    map_config_k_v.clear();

    vector<string> vcs{};
    read_file_2_vec(fn_config, vcs);

    //unordered_map<string, string, stringHash, stringEqual> map_param_k_v;
    for (auto& eline : vcs)
    {
        //wcout << eline << endl; 
        if (eline[0] == '#' || eline.size() <= 2)
        {
            continue;
        }
        auto el_er = get_e_str_left_right(eline);
        auto  e_left = el_er[0];
        auto e_right = el_er[1];
        map_config_k_v[e_left] = e_right;
    }

    for (auto& e_map : map_config_k_v)
    {
        //wcout << e_map.first.GetString() << ":" << e_map.second.GetString() << endl;
    }

}

string com::serial_config(string config_path)
{


    std::ofstream file(config_path);
    if (file.is_open()) {
        for (const auto& pair : map_config_k_v) {
            file << pair.first << "=" << pair.second << std::endl;
        }
        file.close();
        std::cout << "Map contents have been written to file." << std::endl;
    }
    else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    return config_path;
}

void com::trim_log_file(int maxFileSize, int save_line_num)
{

    auto logfile = glo::abs_log_file;

    auto filesize = get_file_size(logfile);

    int filesize_thr = maxFileSize;
    const int fileline_thr = save_line_num;

    if (filesize > filesize_thr)
    {
        vector<string> fc{};
        read_file_2_vec(logfile, fc);

        int start_line_no = 0;
        start_line_no = (int)fc.size() - fileline_thr;
        if (start_line_no < 0)
        {
            start_line_no = 0;
        }

        auto vec_slice = vector<string>(fc.begin() + start_line_no, fc.end());
        serial_vec(vec_slice, logfile);
    }

}

string com::serial_vec(vector<string>& vs, string fn)
{
    std::ofstream file(fn);
    if (file.is_open()) {
        for (const auto& str : vs) {
            file << str << std::endl;
        }
        file.close();
        std::cout << "Vector content has been written to file: " << fn << std::endl;
    }
    else {
        std::cerr << "Failed to open file: " << fn << std::endl;
    }
    return fn;
}
int com::hashstr(const string& str)
{
    std::hash<std::string> hasher;
    size_t hashValue = hasher(str);
    return hashValue;
}

// com_ cpp end
#endif 
