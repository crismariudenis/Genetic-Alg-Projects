#include <fstream>
#include <iostream>
#include <string>
std::ifstream fin("testlogs");
int main()
{
    std::string x, type;
    double total_correctness_ratio = 0.0;
    double total_speed_ratio = 0.0;
    int error_count = 0;
    int time_count = 0;
    while (fin >> x)
    {
        fin >> type;
        double d[5];
        for (int i = 0; i < 5; i++)
            fin >> d[i];
        if (type == "error")
        {
            if (d[0] == 0 && d[4] == 0) // Both are zero, consider them equal
            {
                total_correctness_ratio += 1.0, error_count++;
            }
            else
            {
                double correctness_ratio = d[4] / d[0];
                total_correctness_ratio += correctness_ratio;
                error_count++;
            }
        }
        else if (type == "time")
        {
            if (d[0] == 0 && d[4] == 0) // Both are zero, consider them equal
            {
                total_correctness_ratio += 1.0, error_count++;
            }
            else
            {
                double speed_ratio = d[0] / d[4];
                total_speed_ratio += speed_ratio;
                time_count++;
            }
        }
    }

    if (error_count > 0)
    {
        double average_correctness_ratio = total_correctness_ratio / error_count;
        double correctness_percentage = (average_correctness_ratio - 1.0) * 100.0;
        std::cout << "Average Correctness Ratio: " << correctness_percentage << "% more correct" << std::endl;
    }
    else
    {
        std::cout << "No valid error data found." << std::endl;
    }

    if (time_count > 0)
    {
        double average_speed_ratio = total_speed_ratio / time_count;
        double speed_percentage = (average_speed_ratio - 1.0) * 100.0;
        std::cout << "Average Speed Ratio: " << speed_percentage << "% faster" << std::endl;
    }
    else
    {
        std::cout << "No valid time data found." << std::endl;
    }
}