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
    std::string folderPath = "./data/medium/";

    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".tsp")
        {

            std::string filePath = entry.path().string();
            GeneticMain g{filePath};
            g.benchmark(30);
        }
    }
}
void recursive1()
{
    std::string folderPath = "./data/hard/";
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".tsp")
        {
            std::string filePath = entry.path().string();
            std::cout << "Finished: " << filePath << '\n';
            GeneticMain g{filePath};
            g.benchmark(30);
        }
    }
}

int main()
{
        while(true)
        recursive1();

    //   std::cout << "Finished the easy problems\n";
    // recursive1();
    // for (int i = 0; i < 100; i++)
    // {
    //     GeneticMain g{"./data/easy/rat99.tsp"};
    //     g.benchmark(30);
    // }
    // for (int i = 0; i < 100; i++)
    // {
    //     GeneticMain g{"./data/easy/st70.tsp"};
    //     g.benchmark(30);
    // }

    // recursive();
    // GeneticMain g{"./data/easy/rat99.tsp"};
    // g.benchmark(30);

    // g.benchmark(30);
    // g.benchmark(30);
    // g.benchmark(30);
}
