#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <format>
std::ifstream fin("test");
int count;
int main()
{
    std::string name, junk;
    char c;
    double totalTB = 0, totalTF = 0;
    double totalTBV = 0, totalTFV = 0;
    while (fin >> c)
    {
        double bc, fc;
        fin >> name >> junk >> c >> bc >> c >> fc;
        if (junk == "time" && name == "Average")
            totalTB += bc, totalTF += fc;
        if (junk == "error" && name == "Average")
            totalTBV += bc, totalTFV += fc;
    }
    std::cout << std::format("Time improvement: {:.5f}, {:.5f} {:.5f}\n", totalTF / totalTB * 100, totalTB, totalTF);
    std::cout << std::format("Error improvement: {:.5f}, {:.5f} {:.5f}\n", totalTBV / totalTFV * 100, totalTBV, totalTFV);
}