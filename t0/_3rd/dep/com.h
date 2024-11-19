//
// Created by jd on 20240517.
//

#pragma once


#if 1
#include <filesystem>
#include <unordered_map>
#include <vector>

#endif 


using namespace std;



// com_ hpp start 
#define assertx(exp, true_false) assert((exp, true_false))
#define COM_LOG_FILE "log.log"
#define ecl ((*(glo::pcom)).logger)



template<typename T>
string to_string_(T n)
{
    ostringstream ss;
    ss << n;
    return ss.str();
}



class com
{
public:
    com(void);
    string get_exe_abs_path();
    string get_exe_dir();
    string get_current_timestamp();
    //string get_current_timestamp(int mill);
#if 0
    std::wstring cstring2wstring(string id_cstring);
    std::string wstring2string(wstring id_wstring);
    std::string cstring2string(string id_cstring);
#endif 
    void logger(const string & id_s);
    void m_s(string msg);
    void glo_init(int argc, char** argv);

    string d2s(double d_);
    string f2s(float d_);
    string i2s(int d_);
    int file_exist(string fn);

    vector<string> get_e_str_left_right(string str);

    int get_file_size(string filePath);

    void read_file_2_vec(const string& filePath, std::vector<string>& lines);

    unordered_map<string, string> map_config_k_v;

    void read_config(string fn_config);
    string serial_config(string config_path);
    void trim_log_file(int maxFileSize, int save_line_num);

    string serial_vec(vector<string>& vs, string fn);
    int hashstr(const string& str);
    
    // template
    template<typename T>
    inline static string tostring_(const T& t)
    {
        ostringstream sBuffer;
        sBuffer << t;
        return sBuffer.str();
    }


};
// com_ hpp end


// glo_ hpp start 
class com; 

class  glo
{
public:

    static int gx;
    static int argc;
    static string abs_exe_path;
    static string abs_exe_dirpath;
    static string abs_log_file;
    static char** argv;
    static com* pcom;
};

// glo_ hpp end 




