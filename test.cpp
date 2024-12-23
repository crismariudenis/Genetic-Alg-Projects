#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

std::ifstream fin("totalLogs");
int main()
{

    std::string name, type;
    double a, b, c;

    std::vector<std::pair<double, double>> error_percentages;
    std::vector<std::pair<double, double>> time_percentages;

    while (fin >> name)
    {
        fin >> type >> a >> b >> c;
        if (type == "error" && name == "Average")
        {
            double percent_b = (a != 0) ? ((b - a) / a) * 100 : 0;
            double percent_c = (a != 0) ? ((c - a) / a) * 100 : 0;
            error_percentages.push_back({percent_b, percent_c});
        }

        if (type == "time" && name == "Average")
        {
            double percent_b = (a != 0) ? ((b - a) / a) * 100 : 0;
            double percent_c = (a != 0) ? ((c - a) / a) * 100 : 0;
            time_percentages.push_back({percent_b, percent_c});
        }
    }

    double error_percent_b_sum = 0, error_percent_c_sum = 0;
    for (const auto &percent : error_percentages)
    {
        error_percent_b_sum += percent.first;
        error_percent_c_sum += percent.second;
    }
    double avg_error_percent_b = error_percentages.empty() ? 0 : error_percent_b_sum / error_percentages.size();
    double avg_error_percent_c = error_percentages.empty() ? 0 : error_percent_c_sum / error_percentages.size();

    double time_percent_b_sum = 0, time_percent_c_sum = 0;
    for (const auto &percent : time_percentages)
    {
        time_percent_b_sum += percent.first;
        time_percent_c_sum += percent.second;
    }
    double avg_time_percent_b = time_percentages.empty() ? 0 : time_percent_b_sum / time_percentages.size();
    double avg_time_percent_c = time_percentages.empty() ? 0 : time_percent_c_sum / time_percentages.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average Error Percentages (Column 2 vs Column 1, Column 3 vs Column 1):" << std::endl;
    std::cout << "B has " << std::abs(avg_error_percent_b) << "% " << (avg_error_percent_b > 0 ? "more" : "less") << " error than A" << std::endl;
    std::cout << "C has " << std::abs(avg_error_percent_c) << "% " << (avg_error_percent_c > 0 ? "more" : "less") << " error than A" << std::endl;

    std::cout << "\nAverage Time Percentages (Column 2 vs Column 1, Column 3 vs Column 1):" << std::endl;
    std::cout << "B is " << std::abs(avg_time_percent_b) << "% " << (avg_time_percent_b > 0 ? "slower" : "faster") << " than A" << std::endl;
    std::cout << "C is " << std::abs(avg_time_percent_c) << "% " << (avg_time_percent_c > 0 ? "slower" : "faster") << " than A" << std::endl;

    return 0;
}