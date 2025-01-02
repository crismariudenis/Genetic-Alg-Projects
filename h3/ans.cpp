#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
struct LogEntry
{
    std::string problem;
    double avgValue = -1;
    double stdValue = -1;
    double minValue = -1;
    double maxValue = -1;
    double avgTime = -1;
    double stdTime = -1;
    double minTime = -1;
    double maxTime = -1;
    std::string additionalData;
};
std::map<std::string, int64_t> trueValues = {
    {"./data/easy/berlin52.tsp", 7542},
    {"./data/easy/rat99.tsp", 1211},
    {"./data/easy/st70.tsp", 675},
    {"./data/medium/a280.tsp", 2579},
    {"./data/medium/bier127.tsp", 118282},
    {"./data/medium/ch130.tsp", 6110},
    {"./data/medium/pr124.tsp", 59030},
    {"./data/hard/brd14051.tsp", 469385},
    {"./data/hard/d18512.tsp", 645238},
    {"./data/hard/pla33810.tsp", 66048945},
};
std::map<std::string, std::vector<LogEntry>> LOGS;
void parseLogs(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    std::vector<LogEntry> logs;
    std::string line;
    LogEntry logEntry;

    while (std::getline(file, line))
    {
        if (line.find("Problem:") != std::string::npos)
        {
            if (!logEntry.problem.empty())
            {
                logs.push_back(logEntry);
                LOGS[logEntry.problem].push_back(logEntry);
                logEntry = LogEntry();
            }
            logEntry.problem = line.substr(line.find(":") + 2);
        }
        else if (line.find("Avg Value:") != std::string::npos)
        {
            std::istringstream iss(line);
            std::string temp;
            iss >> temp >> temp >> logEntry.avgValue >> temp;
            iss >> temp >> temp >> logEntry.stdValue >> temp;
            iss >> temp >> temp >> logEntry.minValue >> temp;
            iss >> temp >> temp >> logEntry.maxValue >> temp;
        }
        else if (line.find("Avg Time:") != std::string::npos)
        {
            std::istringstream iss(line);
            std::string temp;
            iss >> temp >> temp >> logEntry.avgTime >> temp;
            iss >> temp >> temp >> logEntry.stdTime >> temp;
            iss >> temp >> temp >> logEntry.minTime >> temp;
            iss >> temp >> temp >> logEntry.maxTime >> temp;
        }
        else if (!line.empty())
        {
            logEntry.additionalData = line;
        }
    }

    file.close();
}

int main()
{
    std::string filePath = "logs";
    parseLogs(filePath);

    for (auto &[key, vec] : LOGS)
    {
        std::cout << "\033[31m" << key << "\033[0m" << std::string(50, ' ');
        std::cout << "\033[33m" << trueValues[key] << "\033[0m\n";
        std::sort(vec.begin(), vec.end(), [](const LogEntry &a, const LogEntry &b)
                  {
                      if (a.minValue != b.minValue)
                          return a.minValue < b.minValue;
                      return a.avgValue < b.avgValue;
                  });
        for (auto &log : vec)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Avg Value: " << log.avgValue << ", Standard Value: " << log.stdValue
                      << ", Min Value: " << log.minValue << ", Max Value: " << log.maxValue << std::endl;

            if (!log.additionalData.empty())
            {
                std::cout << "Additional Data: " << log.additionalData << std::endl;
            }
            std::cout << std::endl;
        }
    }
    return 0;
}