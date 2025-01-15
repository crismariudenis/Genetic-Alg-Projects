#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <functional>
#include <format>
#include <assert.h>
#include <unordered_map>
#include "TSP_parse.h"
#include <algorithm>
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
    double elitism = 0.05;
    int crossoverPoints = 2;
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

    std::vector<int> decodePermutation(const std::vector<int> &vec) const
    {
        int n = vec.size();
        std::vector<int> perm(n);
        std::vector<int> positions(n);

        std::iota(positions.begin(), positions.end(), 0);

        for (int i = 0; i < n; ++i)
        {
            perm[i] = positions[vec[i]];
            positions.erase(positions.begin() + vec[i]);
        }

        return perm;
    }

    double eval(const Individual &b)
    {
        auto path = decodePermutation(b.path);
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

        mutationRate = 0.1;
        pop.resize(popSize);
        for (auto &x : pop)
        {
            x.path.resize(graph.size());

            // Generate random inversion sequence
            for (int j = 0; j < x.path.size(); ++j)
            {
                std::uniform_int_distribution<> dis(0, x.path.size() - j - 1);
                x.path[j] = dis(gen);
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
    int consecutiveBest = 0;
    int maxConsecutiveBest = 0;
    void computeFitness()
    {
        double minFitness = std::numeric_limits<double>::max(), maxFitness = std::numeric_limits<double>::lowest();
        for (int i = 0; i < pop.size(); i++)
        {
            pop[i].fitness = eval(pop[i]);
            minFitness = std::min(pop[i].fitness, minFitness);
            maxFitness = std::max(pop[i].fitness, maxFitness);
        }
        if (bestFitness == minFitness)
            consecutiveBest++;
        else
            consecutiveBest = 0;
        maxConsecutiveBest = std::max(consecutiveBest, maxConsecutiveBest);

        bestFitness = minFitness;
        double eps = 1e-9;

        double sum = 0;
        for (auto &p : pop)
        {
            p.fitness = std::pow(((maxFitness - p.fitness) / (maxFitness - minFitness + eps) + 1), 1);
            // p.fitness = 1 / (p.fitness + eps);
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
            int count = 100;
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

    size_t selectParent1()
    {
        int tournament_size = 5; // You can adjust the tournament size as needed
        std::uniform_int_distribution<> dis(0, pop.size() - 1);

        // Select the first competitor
        int best = dis(gen);
        for (int i = 1; i < tournament_size; i++)
        {
            int competitor = dis(gen);
            if (pop[competitor].fitness < pop[best].fitness)
            {
                best = competitor;
            }
        }
        return best;
    }

    void crossover(int p1, int p2, Individual &ind1, Individual &ind2)
    {
        ind1.path.resize(pop[p1].path.size());
        ind2.path.resize(pop[p1].path.size());

        std::uniform_int_distribution<> dis(0, pop[p1].path.size() - 1);

        std::vector<int> points(crossoverPoints + 1);

        // Randomly select crossover points
        for (int i = 0; i < points.size() - 1; i++)
        {
            points[i] = dis(gen);
        }
        points.back() = pop[p1].path.size() - 1;

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
    }

    void mutate(Individual &ind)
    {
        std::uniform_int_distribution<> dis(0, graph.size());

        std::uniform_real_distribution<> realDis(0.0, 1.0); // Uniform distribution for floating-point numbers

        if (realDis(gen) < mutationRate) // Use the real distribution for mutation check
        {
            std::uniform_int_distribution<> dis(0, ind.path.size() - 1);
            int i = dis(gen);
            int range = graph.size() - 1 - i;
            if (range > 0)
            {
                std::uniform_int_distribution<> validRange(0, range);
                ind.path[i] = (validRange(gen) + ind.path[i]) % range;
            }
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
            auto [v, t] = pop.run(4000);
            errors[i] = v;
            times[i] = t;
            minE = std::min(minE, errors[i]);
            maxE = std::max(maxE, errors[i]);

            minT = std::min(minT, times[i]);
            maxT = std::max(maxT, times[i]);
        }
        std::cout << "Problem: " << file << '\n';
        std::cout << std::format("   Avg Value: {:.3f}, Standard Value: {:.3f}, Min Value: {:.3f}, Max Value: {:.3f}\n", median(errors), sd(errors), minE, maxE);
        std::cout << std::format("   Avg Time: {:.3f}, Standard Time: {:.3f}, Min Time: {:.3f}, Max Time: {:.3f}\n", median(times), sd(times), minT, maxT);
        std::cout << '\n';
    }
};
