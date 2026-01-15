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
#include <cmath> 
#include <regex>
#include <rex.h>  // jd define regex

// #define PCRE2_CODE_UNIT_WIDTH 8
// #include <pcre2.h>   // .\vcpkg install pcre2:x64-windows && .\vcpkg integrate install 

#define byte uchar
#define WIDTH_ALIGN(_w, _align) \
    ((_w % _align != 0) ? (_w / _align + 1) * _align : _w)
#define WIDTH_4(_w) WIDTH_ALIGN(_w, 4)

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


#include "cimg_imp.inc"

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

// main_
int main(int argc, char** argv)
{
    //  global var  // 
    com& ec = s_com;
    ec.glo_init(argc, argv);

    cimg ci; cimg ci_0; cimg ci_1; cimg ci_2;
    // -------- //
#if 1

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
                efc = ci.read_txt_to_vec_str(efn);
                auto& e_csv_data = csvdata[efn];

                e_csv_data.l.reserve(efc.size());
                e_csv_data.r.reserve(efc.size());



                for (auto eline : efc)
                {
                    auto v_ele = ci.split_str_2_vec(eline, ',');
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

#endif 


    return 0;
    
}

