#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <functional>
#include <chrono>
#include <iomanip>
#include <format>
#include <string>
#include <random>
#include <thread>
#include <filesystem>
#include "R.h"
#include "defaultGA.h"

void recursive()
{
    std::string folderPath = "./data/easy";
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".tsp")
        {
            std::string filePath = entry.path().string();
            GeneticMain g{filePath};
            g.benchmark(100);
        }
    }
}

int main()
{
    // recursive();
    GeneticMain g{"./data/easy/rat99.tsp"};
    g.benchmark(100);
    // g.benchmark(30);
    // g.benchmark(30);
    // g.benchmark(30);
}
