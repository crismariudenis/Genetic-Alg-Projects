#include <iostream>
#include <vector>
#include <chrono>
#include <format>
#include <functional>
#include <random>
#include <filesystem>
#include "assert.h"
#include "R.h"
#include "TSP_parse.h"
class Timer
{

public:
    Timer() { startTime = std::chrono::high_resolution_clock::now(); }

    double getTime()
    {
        auto endTime = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::time_point_cast<std::chrono::milliseconds>(startTime).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::milliseconds>(endTime).time_since_epoch().count();

        auto duration = end - start;

        return duration * 1e-3;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};
struct Individual
{
    double fitness;
    std::vector<int> path;
};

std::mt19937_64 gen;

double eval(const Individual &b, Graph &graph)
{
    auto &path = b.path;
    assert(path.size() == graph.size());
    double cost = 0;
    for (int i = 0; i < path.size() - 1; i++)
        cost += graph.dist(path[i], path[i + 1]);
    cost += graph.dist(path.back(), path[0]);
    return cost;
}

Individual getRandomPath(Graph &graph)
{
    Individual ind;
    ind.path.resize(graph.size());
    std::iota(ind.path.begin(), ind.path.end(), 0);
    std::shuffle(ind.path.begin(), ind.path.end(), gen);
    return ind;
}

Individual getNeighbor(Individual &curr)
{
    Individual neighbor = curr;
    auto &path = neighbor.path;
    std::uniform_int_distribution<> dis(0, 3);

    int r = dis(gen);

    std::uniform_int_distribution<> randIdx(0, path.size() - 1);
    int idx1 = randIdx(gen);
    int idx2 = randIdx(gen);

    if (r == 0) // swap 2 cities
    {
        std::swap(path[idx1], path[idx2]);
    }
    else if (r == 1) // inversion mutation
    {
        std::swap(idx1, idx2);
        std::reverse(path.begin() + idx1, path.begin() + idx2 + 1);
    }
    else if (r == 2) // scramble mutation
    {
        if (idx1 > idx2)
            std::swap(idx1, idx2);
        std::shuffle(path.begin() + idx1, path.begin() + idx2 + 1, gen);
    }
    else if (r == 3) // rotation mutation
    {
        if (idx1 > idx2)
            std::swap(idx1, idx2);
        std::rotate(path.begin() + idx1, path.begin() + idx1 + 1, path.begin() + idx2 + 1);
    }

    return neighbor;
}
std::pair<double, double> newSimulatedAnealing(double T, double alpha, Graph &graph)
{
    Timer t;
    double ans = std::numeric_limits<double>::max();
    Individual curr = getRandomPath(graph);
    curr.fitness = eval(curr, graph);
    Individual best = curr;
    int samePath = 0;
    int sameBest = 0;

    while (samePath < std::min(5 * (int)graph.size(), 5000) && sameBest < std::min(50 * (int)graph.size(), 15000))
    {
        Individual neighbor = getNeighbor(curr);
        neighbor.fitness = eval(neighbor, graph);

        double delta = neighbor.fitness - curr.fitness;
        if (delta < 0 || std::exp(-delta / T) > std::uniform_real_distribution<>(0.0, 1.0)(gen))
        {
            curr = neighbor;
            samePath = 0;
        }
        else
            samePath++;

        if (curr.fitness < best.fitness)
        {
            best = curr;
            sameBest = 0;
        }
        else
        {
            sameBest++;
        }

        T *= alpha;
    }

    ans = best.fitness;
    return {ans, t.getTime()};
}
std::pair<double, double> simulatedAnealing(int epochs, Graph &graph)
{
    Timer t;
    double ans = std::numeric_limits<double>::max();
    for (int e = 1; e <= epochs; e++)
    {
        Individual curr = getRandomPath(graph);
        double bestVal = eval(curr, graph);
        double T = 1;
        std::uniform_real_distribution<> realDis(0.0, 1.0);
        while (T > 1e-5)
        {
            std::pair<int, int> swaped{-1, -1};
            for (int i = 0; i < curr.path.size(); i++)
                for (int j = i + 1; j < curr.path.size(); j++)
                {
                    std::swap(curr.path[i], curr.path[j]);

                    double ngVal = eval(curr, graph);

                    if (bestVal > ngVal)
                    {
                        bestVal = ngVal;
                        swaped = {i, j};
                    }
                    else if (realDis(gen) < std::exp(-std::abs(bestVal - ngVal) / T))
                    {
                        bestVal = ngVal;
                        swaped = {i, j};
                    }

                    std::swap(curr.path[i], curr.path[j]);
                }

            if (swaped.first == -1) // the local best
                break;
            else
                std::swap(curr.path[swaped.first], curr.path[swaped.second]);
            T *= 0.9;
        }
        if (e % 10 == 0)
            std::cout << e << " " << T << " " << bestVal << '\n';
        ans = std::min(bestVal, ans);
    }
    return {ans, t.getTime()};
}
// https://medium.com/@francis.allanah/travelling-salesman-problem-using-simulated-annealing-f547a71ab3c6
class SAMain
{

public:
    double trueValue = 0;
    TSP tsp;
    std::string file;
    SAMain(std::string file, double trueValue = 0) : trueValue(trueValue), file(file)
    {
        tsp.init(this->file);
    }
    void benchmark(int nrSamples)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);

        double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::lowest();
        double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::lowest();

        std::random_device rd;
        gen.seed(rd());
        std::uniform_real_distribution<> tempDist(1.0, 100.0);
        std::uniform_real_distribution<> alphaDist(0.8, 0.99);
        double T = 100;
        double alpha = 0.99;
        for (int i = 0; i < nrSamples; i++)
        {
            auto [v, t] = newSimulatedAnealing(T, alpha, tsp.graph);
            errors[i] = v;
            times[i] = t;
            minT = std::min(minT, times[i]);
            maxT = std::max(maxT, times[i]);
        }

        minE = *std::min_element(errors.begin(), errors.end());
        maxE = *std::max_element(errors.begin(), errors.end());
        std::cout << "SA Problem: " << file << '\n';
        std::cout << std::format("   Avg Value: {:.3f}, Standard Value: {:.3f}, Min Value: {:.3f}, Max Value: {:.3f}\n", median(errors), sd(errors), minE, maxE);
        std::cout << std::format("   Avg Time: {:.3f}, Standard Time: {:.3f}, Min Time: {:.3f}, Max Time: {:.3f}\n", median(times), sd(times), minT, maxT);
        std::cout << '\n';
    }
};

void recursive()
{
    std::string folderPath = "./data/easy/";

    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".tsp")
        {

            std::string filePath = entry.path().string();
            if (filePath == "./data/easy/berlin52.tsp")
                continue;
            SAMain sa{filePath};
            sa.benchmark(30);
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
            SAMain sa{filePath};
            sa.benchmark(30);
        }
    }
}
int main()
{
    // for (int i = 0; i < 10; i++)
    //     recursive1();

    for (int i = 0; i < 30; i++)
    {
        SAMain sa{"./data/easy/rat99.tsp"};
        sa.benchmark(30);
    }
    for (int i = 0; i < 50; i++)
    {
        SAMain sa{"./data/medium/a280.tsp"};
        sa.benchmark(30);
    }

    for (int i = 0; i < 50; i++)
    {
        SAMain sa{"./data/medium/ch130.tsp"};
        sa.benchmark(30);
    }
}