#include <vector>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <sstream>

std::string exec(const char *cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, void (*)(FILE *)> pipe(popen(cmd, "r"), [](FILE *f) -> void
                                                 { std::ignore = pclose(f); });

    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return result;
}

double sd(const std::vector<double> &v)
{
    std::string cmd{"R -s -e \"cat(sd(c("};
    for (int i = 0; i < v.size(); i++)
        cmd += std::to_string(v[i]) + ',';
    cmd.back() = ')';
    cmd += "))\"";

    std::string output = exec(cmd.c_str());
    return std::stod(output);
}
double median(const std::vector<double> &v)
{
    std::string cmd{"R -s -e \"cat(median(c("};
    for (int i = 0; i < v.size(); i++)
        cmd += std::to_string(v[i]) + ',';
    cmd.back() = ')';
    cmd += "))\"";

    std::string output = exec(cmd.c_str());
    return std::stod(output);
}
std::array<double, 5> quantile(const std::vector<double> &v)
{
    std::string cmd{"R -s -e \"cat(quantile(c("};
    for (int i = 0; i < v.size(); i++)
        cmd += std::to_string(v[i]) + ',';
    cmd.back() = ')';
    cmd += ",names=FALSE))\"";

    std::string output = exec(cmd.c_str());
    std::istringstream iss{output};
    std::array<double, 5> ans;
    for (auto &x : ans)
        iss >> x;
    return ans;
}
