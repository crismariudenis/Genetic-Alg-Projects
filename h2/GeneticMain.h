#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <functional>
#include <format>
#include <unordered_map>
static double data[64][64];

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
        std::vector<bool> bin;
    };
    bool useGrayCode = false;
    int popSize, binSize, n;
    double min, max;
    std::vector<Individual> pop;
    std::mt19937_64 gen;
    size_t bestIndiv = 0;
    double bestFitness;

    double mutationRate;
    std::function<double(const std::vector<double> &)> f;
    std::vector<bool> genBin()
    {
        std::vector<bool> bin(binSize);

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

    std::vector<bool> grayToBinary(const std::vector<bool> &gray)
    {
        std::vector<bool> bin(gray.size());

        bin[0] = gray[0];
        for (size_t i = 1; i < gray.size(); ++i)
        {
            bin[i] = bin[i - 1] ^ gray[i];
        }
        return bin;
    }

    std::vector<double> binToVec(const std::vector<bool> &bin)
    {
        std::vector<double> v(bin.size() / n);
        int maxVal = std::pow(2, n) - 1;

        std::vector<bool> binary = bin;
        if (useGrayCode)
        {
            binary = grayToBinary(bin);
        }

        for (int i = 0; i < v.size(); i++)
        {
            int val = 0;
            for (int j = 0; j < n; j++)
                val = (val << 1) + binary[i * n + j];

            v[i] = min + (double)val * (max - min) / maxVal;
        }
        return v;
    }
    int grayToBinary(int gray)
    {
        int binary = gray;
        while (gray >>= 1)
        {
            binary ^= gray;
        }
        return binary;
    }

    double eval(const std::vector<bool> &b)
    {
        return f(binToVec(b));
    }

public:
    Population(int popSize, int binSize, int n, double min, double max, std::function<double(const std::vector<double> &)> f) : popSize(popSize), binSize(binSize), n(n), min(min), max(max), f(f)
    {
        std::mt19937_64 temp;
        temp.seed(std::chrono::system_clock::now().time_since_epoch().count() * 6969);
        temp.discard(69420);
        gen.seed(temp());

        mutationRate = 1.0 / (binSize);
        pop.resize(popSize);
        for (auto &p : pop)
        {
            p.bin = genBin();
        }
    }
    void print()
    {
        for (int i = 0; i < pop.size(); i++)
        {
            std::cout << "Member #" << i << " : ";
            for (auto x : pop[i].bin)
                std::cout << x;
            std::cout << '\n';
        }
    }
    void computeFitness()
    {
        double minFitness = std::numeric_limits<double>::max(), maxFitness = std::numeric_limits<double>::min();
        double sum = 0;
        for (int i = 0; i < pop.size(); i++)
        {
            pop[i].fitness = eval(pop[i].bin);
            if (minFitness > pop[i].fitness)
            {
                minFitness = pop[i].fitness;
                bestIndiv = i;
                bestFitness = minFitness;
            }
            maxFitness = std::max(pop[i].fitness, maxFitness);
        }
        double eps = 1e-9;

        for (auto &p : pop)
        {
            p.fitness = std::pow(((maxFitness - p.fitness) / (maxFitness - minFitness + eps) + 1), 2);
            // p.fitness = 1 / (p.fitness + eps);
            sum += p.fitness;
        }

        for (auto &p : pop)
        {
            p.fitness /= sum;
        }
    }

    std::vector<bool> binaryToGray(const std::vector<bool> &bin)
    {
        std::vector<bool> gray(bin.size());
        gray[0] = bin[0];
        for (size_t i = 1; i < bin.size(); ++i)
        {
            gray[i] = bin[i] ^ bin[i - 1];
        }
        return gray;
    }

    std::pair<double, double> run(int generations)
    {
        Timer t;

        double ans = std::numeric_limits<double>::max();

        for (int g = 0; g < generations; g++)
        {

            // if (!useGrayCode
            //     && g >= generations * 0.9
            // )
            // {
            //     for (auto &p : pop)
            //     {
            //         p.bin = binaryToGray(p.bin);
            //     }
            //     useGrayCode = true;
            // }
            computeFitness();
            std::vector<Individual> newPop;
            newPop.resize(popSize);

            // set best Individual
            newPop[0] = pop[bestIndiv];
            ans = std::min(bestFitness, ans);

            for (int i = 1; i < popSize; i++)
            {
                size_t p1 = selectParent();
                size_t p2 = p1;

                while (p2 == p1)
                    p2 = selectParent();

                newPop[i] = crossover(p1, p2);
                mutate(newPop[i]);
            }
            pop = newPop;
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
    Individual crossover(int p1, int p2)
    {
        Individual ind = pop[p1];

        // do this to generate random numb with 0.5
        std::vector<bool> rand = genBin();
        for (int i = 0; i < rand.size(); i++)
        {
            if (rand[i])
                ind.bin[i] = pop[p2].bin[i];
        }
        return ind;
    }
    void mutate(Individual &ind)
    {
        int cnt = 0;
        for (int i = 0; i < ind.bin.size(); i++)
        {
            if ((double)gen() / std::numeric_limits<uint64_t>::max() < mutationRate)
                ind.bin[i] = !ind.bin[i];
        }
    }
};

class GeneticMain
{
    int D = 2; // nr of dimensions
    int p = 5; // precision 10^-p
    int N;
    int n;
    double min, max;
    std::function<double(const std::vector<double> &)> f;
    std::vector<int> dimensions = {5, 10, 30};

public:
    std::vector<double> trueValues = {0, 0, 0};
    GeneticMain(double min, double max, int p, std::function<double(const std::vector<double> &)> f) : min(min), max(max), p(p), f(f)
    {
        N = (max - min) * std::pow(10, p);
        n = std::ceil(std::log2(N));
    }
    void benchmark(int nrSamples)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);
        int index = 5;
        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::min();
            double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::min();
            for (int i = 0; i < nrSamples; i++)
            {
                Population pop{100, D * n, n, min, max, f};
                auto [v, t] = pop.run(1000);
                errors[i] = std::abs(v - trueValues[d]);
                times[i] = t;
                minE = std::min(minE, errors[i]);
                maxE = std::max(maxE, errors[i]);

                minT = std::min(minT, times[i]);
                maxT = std::max(maxT, times[i]);
            }
            data[d * 4][index] = median(errors);
            data[d * 4 + 1][index] = sd(errors);
            data[d * 4 + 2][index] = minE;
            data[d * 4 + 3][index] = maxE;

            data[12 + d * 4][index] = median(times);
            data[12 + d * 4 + 1][index] = sd(times);
            data[12 + d * 4 + 2][index] = minT;
            data[12 + d * 4 + 3][index] = maxT;

            std::cout << std::format("Dimensions: {}, Avg Error: {:.5f}, Standard Error: {:.5f}, Min Error: {:.5f}, Max Error: {:.5f}\n", dimensions[d], data[d * 4][index], data[d * 4 + 1][index], minE, maxE);
            std::cout << std::format("Dimensions: {}, Avg Time: {:.5f}, Standard Time: {:.5f}, Min Time: {:.5f}, Max Time: {:.5f}\n", dimensions[d], data[12 + d * 4][index], data[12 + d * 4 + 1][index], minT, maxT);
        }
        std::cout << '\n';
    }
};
