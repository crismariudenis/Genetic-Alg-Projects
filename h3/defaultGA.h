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
#include <set>
#include <unordered_set>
#include <queue>
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
    double greedyInit = 0.1;
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
    double eval(const Individual &b)
    {
        auto &path = b.path;
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
            std::uniform_real_distribution<> realDis(0.0, 1.0);

            if (realDis(gen) < greedyInit)
            {
                greedyInitializtion(x);
            }
            else
            {
                std::iota(x.path.begin(), x.path.end(), 0);
                std::shuffle(x.path.begin(), x.path.end(), rd);
            }
        }
    }
    // https://www.enggjournals.com/ijet/docs/IJET17-09-02-188.pdf
    void greedyInitializtion(Individual &x)
    {
        std::vector<bool> viz(graph.size(), 0);
        std::uniform_int_distribution<> dis(0, graph.size() - 1);
        std::uniform_real_distribution<> realDis(0.0, 1.0);

        int start = dis(gen);
        int begin = start;
        int indx = 0;
        viz[start] = 1;
        x.path[indx++] = start;
        double totalCost = 0;
        while (true)
        {

            double minCost = std::numeric_limits<double>::max();
            double poz = start;
            for (int i = 0; i < graph.size(); i++)
            {
                if (viz[i] == 0 and i != start and graph.dist(i, start) < minCost)
                {
                    minCost = graph.dist(i, start);
                    poz = i;
                }
            }

            if (start == poz)
                break;
            x.path[indx++] = poz;
            totalCost += graph.dist(start, poz);
            start = poz;
            viz[poz] = 1;
        }
        totalCost += graph.dist(begin, start);
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

        std::vector<std::pair<double, int>> fitnessIndices(pop.size());

        for (int i = 0; i < pop.size(); i++)
            fitnessIndices[i] = {pop[i].fitness, i};

        std::sort(fitnessIndices.begin(), fitnessIndices.end(),
                  [](const std::pair<double, int> &a, const std::pair<double, int> &b)
                  {
                      return a.first > b.first;
                  });

        for (int i = 0; i < size; i++)
        {
            int bestIndex = fitnessIndices[i].second;
            newPop[i] = pop[bestIndex];
        }

        // exit(0);
    }

    double comparePaths(const std::vector<int> &p1, const std::vector<int> &p2)
    {
        typedef std::set<std::pair<int, int>> edge_set;
        auto getEdges = [](const std::vector<int> &p)
        {
            edge_set s;
            for (int i = 0; i < p.size() - 1; i++)
            {
                int a = std::min(p[i], p[i + 1]);
                int b = std::max(p[i], p[i + 1]);
                s.insert({a, b});
            }
            int b = std::max(p[0], p.back());
            int a = std::min(p[0], p.back());
            s.insert({a, b});
            return s;
        };

        auto s1 = getEdges(p1);
        auto s2 = getEdges(p2);

        edge_set intersection;
        edge_set union_set = s1;

        for (const auto &edge : s2)
        {
            union_set.insert(edge);
            if (s1.find(edge) != s1.end())
            {
                intersection.insert(edge);
            }
        }

        // Calculate similarity as the size of the intersection divided by the size of the union
        double similarity = (double)(intersection.size()) / union_set.size();
        return similarity;
    }

    std::pair<double, double> run(int generations)
    {
        Timer t;

        double ans = std::numeric_limits<double>::max();
        bool updatedMutation = false;
        std::vector<Individual> newPop;
        newPop.resize(popSize);
        for (int g = 0; g < generations; g++)
        {
            computeFitness();
            addElit(newPop);
            for (int i = pop.size() * elitism; i < pop.size(); i += 2)
            {
                size_t p1 = selectParent();
                size_t p2 = p1;
                while (p1 != p2)
                    p2 = selectParent();
                // crossover(p1, p2, newPop[i], newPop[i + 1]);
                pmxCrossover(p1, p2, newPop[i], newPop[i + 1]);

                mutate(newPop[i]);
                mutate(newPop[i + 1]);
            }
            for (int i = 0; i < pop.size(); i++)
                pop[i] = newPop[i];

            ans = std::min(bestFitness, ans);

            if (consecutiveBest > 50)
                mutationRate = 0.5;
            else
                mutationRate = 0.1;

            if (consecutiveBest > 100)
            {
                for (size_t i = pop.size() * elitism; i < pop.size(); ++i)
                {
                    std::shuffle(pop[i].path.begin(), pop[i].path.end(), gen);
                }
                size_t size = pop.size() * elitism;

                for (int i = 0; i < size; i++)
                    for (int j = i + 1; j < size; j++)
                    {
                        double similarity = comparePaths(newPop[i].path, newPop[j].path) * 100;
                        if (similarity == 100)
                        {
                            mutate(newPop[j]);
                        }
                    }

                // consecutiveBest = 0;
            }
        }
        // std::cout << maxConsecutiveBest;
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
    // https://itnext.io/the-genetic-algorithm-and-the-travelling-salesman-problem-tsp-31dfa57f3b62
    void pmxCrossover(int p1, int p2, Individual &ind1, Individual &ind2)
    {
        ind1 = pop[p1];
        ind2 = pop[p2];
        auto &parent1 = pop[p1].path;
        auto &parent2 = pop[p2].path;

        std::uniform_int_distribution<> dis(0, parent1.size() - 1);
        int start = dis(gen);
        int end = dis(gen);

        if (start > end)
            std::swap(start, end);

        // Copy the segment from the first parent to the offspring
        for (int i = start; i <= end; i++)
        {
            ind1.path[i] = parent2[i];
            ind2.path[i] = parent1[i];
        }

        // Fill the rest of the offspring
        auto fillRemaining = [&](const std::vector<int> &source, std::vector<int> &target, int start, int end)
        {
            std::unordered_set<int> used(target.begin() + start, target.begin() + end + 1);

            int currentPos = 0;
            for (int value : source)
            {
                if (used.find(value) == used.end())
                {
                    // Skip positions in the copied segment
                    while (currentPos >= start && currentPos <= end)
                        currentPos++;

                    target[currentPos++] = value;
                    used.insert(value);
                }
            }
        };

        fillRemaining(parent1, ind1.path, start, end);
        fillRemaining(parent2, ind2.path, start, end);
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
        std::uniform_real_distribution<> realDis(0.0, 1.0);
        if (realDis(gen) < mutationRate)
        {
            std::uniform_int_distribution<> dis(0, ind.path.size() - 1);

            // Generate two random indices
            int idx1 = dis(gen);
            int idx2 = dis(gen);

            // Ensure idx1 < idx2
            if (idx1 > idx2)
                std::swap(idx1, idx2);

            // Perform a left rotation of the subarray
            std::rotate(ind.path.begin() + idx1, ind.path.begin() + idx1 + 1, ind.path.begin() + idx2 + 1);
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
            auto [v, t] = pop.run(3000);

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
