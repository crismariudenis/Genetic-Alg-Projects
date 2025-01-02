#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <functional>
#include <format>
#include <assert.h>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <queue>
#include "TSP_parse.h"
#include <thread>
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
    double allTimeBestFitness = std::numeric_limits<double>::max();
    int crossoverPoints = 2;
    double mutationRate;
    int nrThreads = 3;

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
    double initialMutationRate = 0.1;
    double elitism = 0.1;
    double greedyInit = 0.1;

    int elitNr = 1;
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
        pop.resize(popSize);
    }
    void init()
    {
        std::random_device rd;
        gen.seed(rd());
        elitNr = elitism * popSize;
        mutationRate = initialMutationRate;
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
        std::vector<bool> viz(graph.size());
        assert(viz.size() == graph.size());
        std::uniform_int_distribution<> dis(0, graph.size() - 1);
        std::uniform_real_distribution<> realDis(0.0, 1.0);

        int start = dis(gen);
        int begin = start;
        int indx = 0;
        assert(start < viz.size());
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
            assert(indx < viz.size());
            x.path[indx++] = poz;

            totalCost += graph.dist(start, poz);
            start = poz;
            assert(poz < viz.size());
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
        allTimeBestFitness = std::min(bestFitness, allTimeBestFitness);

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
        size_t size = elitNr;

        std::vector<std::pair<double, int>> fitnessIndices(pop.size());
        assert(newPop.size() == pop.size());
        newPop.resize(pop.size());

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
        int intersection_size = 0;
        int union_size = s1.size();

        for (int i = 0; i < p2.size() - 1; i++)
        {
            int a = std::min(p2[i], p2[i + 1]);
            int b = std::max(p2[i], p2[i + 1]);
            std::pair<int, int> edge = {a, b};
            if (s1.find(edge) != s1.end())
            {
                intersection_size++;
            }
            else
            {
                union_size++;
            }
        }
        int a = std::min(p2[0], p2.back());
        int b = std::max(p2[0], p2.back());
        std::pair<int, int> edge = {a, b};
        if (s1.find(edge) != s1.end())
            intersection_size++;
        else
            union_size++;

        // Calculate similarity as the size of the intersection divided by the size of the union
        double similarity = (double)intersection_size / union_size;
        return similarity;
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

    Individual newSimulatedAnealing(double T, double alpha, Individual state)
    {
        double ans = std::numeric_limits<double>::max();
        Individual curr = state;
        curr.fitness = eval(curr);
        Individual best = curr;
        int samePath = 0;
        int sameBest = 0;

        while (samePath < std::min(5 * (int)graph.size(), 500) && sameBest < std::min(50 * (int)graph.size(), 1500))
        {
            Individual neighbor = getNeighbor(curr);
            neighbor.fitness = eval(neighbor);

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
        return best;
    }
    std::pair<double, double>
    run(int generations)
    {
        Timer t;

        double ans = std::numeric_limits<double>::max();
        std::vector<Individual> newPop(pop.size());
        for (auto &x : newPop)
        {
            x.path.resize(graph.size());
        }
        for (int g = 0; g < generations; g++)
        {
            computeFitness();
            addElit(newPop);
            for (int i = elitNr; i < pop.size(); i += 2)
            {
                size_t p1 = selectParent();
                size_t p2 = p1;
                while (p1 != p2)
                    p2 = selectParent();
                // crossover(p1, p2, newPop[i], newPop[i + 1]);
                if (i + 1 < pop.size())
                {
                    pmxCrossover(p1, p2, newPop[i], newPop[i + 1]);
                    mutate(newPop[i + 1]);
                }
                else
                    crossover(p1, p2, newPop[i]); // handle the case when non elit are an even number

                mutate(newPop[i]);
            }
            for (int i = 0; i < pop.size(); i++)
                pop[i] = newPop[i];

            ans = std::min(allTimeBestFitness, ans);

            if (consecutiveBest > 50)
                mutationRate = std::min(1.0, 5 * initialMutationRate);
            else
                mutationRate = initialMutationRate;

            if (consecutiveBest % 50 == 0)
            {
                size_t size = elitNr;

                // Apply SA to refine the population
                auto r = genBin(size);
                double T = 100;
                double alpha = 0.9;
                for (int i = 0; i < size; i++)
                {
                    if (r[i])
                        pop[i] = newSimulatedAnealing(T, alpha, pop[i]);
                }
                // for (int i = size; i < pop.size(); i++)
                //     std::shuffle(pop[i].path.begin(), pop[i].path.end(), gen);
            }

            elitNr = std::clamp(elitNr, 1, (int)(elitism * popSize));
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

        if (p1 >= pop.size() || p2 >= pop.size())
        {
            std::cerr << "Parent index out of bounds: p1 = " << p1 << ", p2 = " << p2 << std::endl;
            return;
        }
        auto &parent1 = pop[p1].path;
        auto &parent2 = pop[p2].path;

        // std::cout << ind1.path.size() << " " << ind2.path.size() << " " << pop[p1].path.size() << " " << pop[p1].path.size() << '\n';
        ind1.path = pop[p1].path;
        ind2.path = pop[p2].path;

        if (parent1.size() != ind1.path.size())
            std::cout << "pmxCrossover: p1 path size = " << pop[p1].path.size() << ", p2 path size = " << pop[p2].path.size() << std::endl;

        assert(0 <= parent1.size() - 1 && "Invalid range for uniform_int_distribution");

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

    void crossover(int p1, int p2, Individual &ind1)
    {
        ind1.path.resize(pop[p1].path.size());

        std::uniform_int_distribution<>
            dis(0, pop[p1].path.size() - 1);

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
                }
            }
            else
            {
                for (int j = last_point; j <= point; j++)
                {
                    ind1.path[j] = pop[p2].path[j];
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
        double r = realDis(gen);
        if (r < 0.5)
        {
            inversionMutation(ind);
            return;
        }
        else if (false)
        {
            scrambleMutation(ind);
            return;
        }
        if (realDis(gen) < mutationRate)
        {
            assert(0 <= ind.path.size() - 1 && "Invalid range for uniform_int_distribution");

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
    void inversionMutation(Individual &ind)
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

            // Reverse the subarray
            std::reverse(ind.path.begin() + idx1, ind.path.begin() + idx2 + 1);
        }
    }
    void scrambleMutation(Individual &ind)
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

            // Scramble the subarray
            std::shuffle(ind.path.begin() + idx1, ind.path.begin() + idx2 + 1, gen);
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
        tsp.init(this->file);
    }
    void benchmark(int nrSamples)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);

        double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::lowest();
        double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::lowest();

        std::uniform_real_distribution<> realDis(0, 0.5);
        std::mt19937_64 gen;

        std::random_device rd;
        gen.seed(rd());

        double elit = realDis(gen), mutation = realDis(gen), greedy = realDis(gen);
        for (int i = 0; i < nrSamples; i++)
        {
            Population pop{200, tsp.graph};
            pop.initialMutationRate = mutation;
            pop.elitism = elit;
            pop.init();
            auto [v, t] = pop.run(3000);
            errors[i] = v;
            times[i] = t;
            minT = std::min(minT, times[i]);
            maxT = std::max(maxT, times[i]);
        }
        std::sort(errors.begin(), errors.end());
        std::vector<double> temp(30);
        for (int i = 0; i < 30; i++)
        {
            temp[i] = errors[i];
        }

        minE = *std::min_element(temp.begin(), temp.end());
        maxE = *std::max_element(temp.begin(), temp.end());
        std::cout << "Problem: " << file << '\n';
        std::cout << std::format("   Avg Value: {:.3f}, Standard Value: {:.3f}, Min Value: {:.3f}, Max Value: {:.3f}\n", median(temp), sd(temp), minE, maxE);
        std::cout << std::format("   Avg Time: {:.3f}, Standard Time: {:.3f}, Min Time: {:.3f}, Max Time: {:.3f}\n", median(times), sd(times), minT, maxT);
        std::cout << elit << " " << mutation << '\n';
        std::cout << '\n';
    }
};
