#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <functional>
#include <format>
#include <assert.h>
#include <unordered_map>
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

class Population
{
    struct Individual
    {
        double fitness;
        std::vector<int> path;
    };
    int popSize;
    std::vector<Individual> pop;
    std::mt19937_64 gen;
    size_t bestIndiv = 0;
    double bestFitness;
    double elitism = 0.1;
    int crossoverPoints = 5;

    double mutationRate;

    Graph &graph;
    std::vector<bool> genBin(size_t size)
    {
        std::vector<bool> bin(size);

        int sz = 0;
        while (sz != bin.size())
        {
            uint64_t val = gen();
            int bits = 64;
            while (bits-- > 3 && sz < bin.size())
            {
                bin[sz++] = val & 1;
                val >>= 1;
            }
        }
        return bin;
    }

public:
     std::vector<int> translatePath(const Individual &b)
    {

        std::vector<bool> used(graph.size(), 0);
        // unhash or something the path
        std::vector<int> path(graph.size());
        for (int i = 0; i < b.path.size(); i++)
        {
            int cnt = b.path[i] + 1;
            for (int j = 0; j < used.size(); j++)
            {
                if (used[j] == 0)
                    cnt--;

                if (cnt == 0 and used[j] == 0)
                {
                    used[j] = 1;
                    path[i] = j;
                    break;
                }
            }
        }
        // add the last one
        for (int i = 0; i < used.size(); i++)
            if (!used[i])
            {
                path.back() = i;
                break;
            }
        return path;
    }
     double eval(const Individual &b)
    {
        auto path = translatePath(b);
        assert(path.size() == graph.size());
        double cost = 0;
        for (int i = 0; i < path.size() - 1; i++)
            cost += graph.dist(path[i], path[i + 1]);
        cost += graph.dist(path.back(), path[0]);
        return cost;
    }
    Population(int popSize, Graph &graph) : popSize(popSize), graph(graph)
    {
        std::random_device rd;
        gen.seed(rd());

        mutationRate = 4 / graph.size();
        mutationRate = 0.02;
        pop.resize(popSize);
        for (auto &x : pop)
        {
            x.path.resize(graph.size() - 1);
            for (int i = 0; i < x.path.size(); i++)
            {
                std::uniform_int_distribution<> dis(0, graph.size() - 1 - i);
                x.path[i] = dis(gen);
            }
        }
    }
    void print()
    {
        for (int i = 0; i < pop.size(); i++)
        {
            std::cout << "Member #" << i << " : ";
            for (auto &x : pop[i].path)
                std::cout << x << " ";
            std::cout << '\n';
        }
    }
    void computeFitness()
    {
        double minFitness = std::numeric_limits<double>::max(), maxFitness = std::numeric_limits<double>::lowest();
        for (int i = 0; i < pop.size(); i++)
        {
            pop[i].fitness = eval(pop[i]);
            minFitness = std::min(pop[i].fitness, minFitness);
            maxFitness = std::max(pop[i].fitness, maxFitness);
        }
        bestFitness = minFitness;
        double eps = 1e-9;

        double sum = 0;
        for (auto &p : pop)
        {
            // p.fitness = std::pow(((maxFitness - p.fitness) / (maxFitness - minFitness + eps) + 1), 2);
            p.fitness = 1 / (p.fitness + eps);
            sum += p.fitness;
        }

        for (auto &p : pop)
        {
            p.fitness /= sum;
        }

        // std::cout << bestFitness << '\n';
    }

    void addElit(std::vector<Individual> &newPop)
    {
        size_t size = pop.size() * elitism;

        for (int i = 0; i < size; i++)
            newPop[i].fitness = -1;

        for (auto x : pop)
        {
            for (int i = 0; i < size; i++)
            {
                if (x.fitness >= newPop[i].fitness)
                    std::swap(x, newPop[i]);
            }
        }
    }

    std::pair<double, double> run(int generations)
    {
        Timer t;

        double ans = std::numeric_limits<double>::max();

        std::vector<Individual> newPop;
        newPop.resize(popSize);

        for (int g = 0; g < generations; g++)
        {

            computeFitness();

            for (int i = pop.size() * elitism; i < pop.size(); i += 2)
            {
                size_t p1 = selectParent();
                size_t p2 = p1;

                while (p2 == p1)
                    p2 = selectParent();

                crossover(p1, p2, newPop[i], newPop[i + 1]);
                mutate(newPop[i]);
                mutate(newPop[i + 1]);
            }

            addElit(newPop);
            for (int i = 0; i < pop.size(); i++)
                pop[i] = newPop[i];

            ans = std::min(bestFitness, ans);
        }
        // auto v = translatePath(pop[0]);
        // std::cout << "path: ";
        // for (auto x : v)
        //     std::cout << x << " ";
        // std::cout << '\n';

        return {ans, t.getTime()};
    }

    size_t selectParent()
    {
        double r = (double)gen() / std::numeric_limits<uint64_t>::max();
        double sum = 0;
        for (int i = 0; i < pop.size() - 1; i++)
        {
            sum += pop[i].fitness;
            if (sum >= r)
            {
                return i;
            }
        }

        // for avoiding rounding errors
        return pop.size() - 1;
    }
    void crossover(int p1, int p2, Individual &ind1, Individual &ind2)
    {
        ind1.path.resize(pop[p1].path.size());
        ind2.path.resize(pop[p1].path.size());

        std::uniform_int_distribution<> dis(0, pop[p1].path.size() - 1);

        std::vector<int> points(crossoverPoints);

        // Randomly select crossover points
        for (int i = 0; i < points.size(); i++)
        {
            points[i] = dis(gen);
        }

        // Sort points to make sure the order is ascending
        std::sort(points.begin(), points.end());

        // Perform crossover at each segment defined by the points
        bool swap = false;
        int last_point = 0;
        for (int i = 0; i < points.size(); i++)
        {
            int point = points[i];

            // Swap segments
            if (swap)
            {
                for (int j = last_point; j <= point; j++)
                {
                    ind1.path[j] = pop[p1].path[j];
                    ind2.path[j] = pop[p2].path[j];
                }
            }
            else
            {
                for (int j = last_point; j <= point; j++)
                {
                    ind1.path[j] = pop[p2].path[j];
                    ind2.path[j] = pop[p1].path[j];
                }
            }

            // Toggle swap
            swap = !swap;
            last_point = point + 1; // Set the new segment start
        }

        // Handle the final segment after the last crossover point
        if (swap)
        {
            for (int i = last_point; i < pop[p1].path.size(); i++)
            {
                ind1.path[i] = pop[p1].path[i];
                ind2.path[i] = pop[p2].path[i];
            }
        }
        else
        {
            for (int i = last_point; i < pop[p1].path.size(); i++)
            {
                ind1.path[i] = pop[p2].path[i];
                ind2.path[i] = pop[p1].path[i];
            }
        }
    }

    void mutate(Individual &ind)
    {
        std::uniform_int_distribution<> dis(0, graph.size());

        for (int i = 0; i < ind.path.size(); i++)
        {
            // Todo: Check if maybe -1 work as well
            if ((double)gen() / std::numeric_limits<uint64_t>::max() < mutationRate)
                ind.path[i] = (dis(gen) + ind.path[i]) % (graph.size() - 1 - i);
        }
    }
};

class GeneticMain
{

public:
    double trueValue = 0;
    TSP tsp;
    std::string file;
    GeneticMain(std::string file, double trueValue = 0) : trueValue(trueValue), file(file)
    {
        tsp = TSP{file};
    }
    void benchmark(int nrSamples)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);

        double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::lowest();
        double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::lowest();
        for (int i = 0; i < nrSamples; i++)
        {
            Population pop{200, tsp.graph};
            auto [v, t] = pop.run(2000);
            errors[i] = std::abs(v - trueValue);
            errors[i] = v;
            times[i] = t;
            minE = std::min(minE, errors[i]);
            maxE = std::max(maxE, errors[i]);

            minT = std::min(minT, times[i]);
            maxT = std::max(maxT, times[i]);
        }

        std::cout << std::format("Problem: {}, Avg Error: {:.3f}, Standard Error: {:.3f}, Min Error: {:.3f}, Max Error: {:.3f}\n", file, median(errors), sd(errors), minE, maxE);
        std::cout << std::format("Problem: {}, Avg Time: {:.3f}, Standard Time: {:.3f}, Min Time: {:.3f}, Max Time: {:.3f}\n", file, median(times), sd(times), minT, maxT);
        std::cout << '\n';
    }
};
