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
#include "R.h"
#include "GeneticMain.h"

class HillClimb
{

    int D = 2; // nr of dimensions
    int p = 5; // precision 10^-p
    int N;
    int n;
    double min, max;
    std::function<double(const std::vector<double> &)> f;
    std::vector<int> dimensions = {5, 10, 30};

    std::mt19937_64 gen;
    std::vector<bool> vecToBin(const std::vector<double> &v)
    {
        std::vector<bool> bin(v.size() * n);

        for (int i = 0; i < v.size(); i++)
        {
            int y = (v[i] - min) * std::pow(10, p);
            for (int j = 0; j < n; j++)
            {
                bin[i * n + (n - 1) - j] = y & 1;
                y >>= 1;
            }
        }
        return bin;
    }

    std::vector<double> binToVec(const std::vector<bool> &bin)
    {
        std::vector<double> v(bin.size() / n);
        int maxVal = std::pow(2, n) - 1;
        for (int i = 0; i < v.size(); i++)
        {
            int val = 0;
            for (int j = 0; j < n; j++)
                val = (val << 1) + bin[i * n + j];
            v[i] = min + (double)val * (max - min) / maxVal;
        }
        return v;
    }

    double inline eval(const std::vector<bool> &b)
    {
        return f(binToVec(b));
    }

public:
    enum class HC_TYPE
    {
        BEST,
        FIRST,
        WORST,
        SA
    };
    std::vector<double> trueValues = {0, 0, 0};
    HillClimb(double min, double max, int p, std::function<double(const std::vector<double> &)> f) : min(min), max(max), p(p), f(f)
    {
        N = (max - min) * std::pow(10, p);
        n = std::ceil(std::log2(N));
        std::mt19937_64 temp;
        temp.seed(std::chrono::system_clock::now().time_since_epoch().count() * 1000);
        temp.discard(69420);
        gen.seed(temp());
    }
    ~HillClimb()
    {
        std::vector<std::string> row = {"Average", "SDev", "Min", "Max"};

        for (int i = 0; i < 24; i++, std::cout << '\n')
        {
            if (i % 4 == 0)
                std::cout << "D = " << dimensions[i / 4 % 3] << " & ";
            else
                std::cout << "     &   ";
            std::cout << std::format("{} {} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\{}", row[i % 4], (i / 12 ? "time" : "error"), data[i][0], data[i][1], data[i][2], data[i][3], i % 4 == 3 ? "\n" : "");
        }
    }
    std::vector<bool> genBin()
    {
        std::vector<bool> bin(D * n);

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

    std::pair<double, double> simulatedAnealing(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();
        for (int e = 1; e <= epochs; e++)
        {
            std::vector<bool> bin = genBin();
            double bestVal = eval(bin);
            double T = 10000;
            while (T > 1e-5)
            {
                int index = -1;

                for (int i = 0; i < bin.size(); i++)
                {
                    bin[i] = !bin[i];

                    double ngVal = eval(bin); // value for the vector with the bit i flipped
                    if (bestVal > ngVal)
                    {
                        bestVal = ngVal;
                        index = i;
                    }
                    else if ((double)gen() / std::numeric_limits<uint64_t>::max() < std::exp(-std::abs(bestVal - ngVal) / T))
                    {
                        bestVal = ngVal;
                        index = i;
                    }
                    bin[i] = !bin[i];
                }

                if (index == -1) // bin is the local best
                    break;
                else
                    bin[index] = !bin[index];
                T *= 0.9;
            }
            ans = std::min(bestVal, ans);
        }
        return {ans, t.getTime()};
    }

    std::pair<double, double> run(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();

        for (int e = 1; e <= epochs; e++)
        {
            std::vector<bool> bin = genBin();
            double bestVal = eval(bin);
            while (true)
            {
                int index = -1;

                for (int i = 0; i < bin.size(); i++)
                {
                    bin[i] = !bin[i];

                    double ngVal = eval(bin); // value for the vector with the bit i flipped
                    if (bestVal > ngVal)
                    {
                        bestVal = ngVal;
                        index = i;
                    }
                    bin[i] = !bin[i];
                }

                if (index == -1) // bin is the local best
                    break;
                else
                    bin[index] = !bin[index];
            }
            ans = std::min(bestVal, ans);
        }
        return {ans, t.getTime()};
    }

    std::pair<double, double> runWorst(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();

        for (int e = 1; e <= epochs; e++)
        {
            std::vector<bool> bin = genBin();
            double bestVal = eval(bin);
            double currVal = bestVal;
            while (true)
            {
                int index = -1;

                for (int i = 0; i < bin.size(); i++)
                {
                    bin[i] = !bin[i];

                    double ngVal = eval(bin); // value for the vector with the bit i flipped

                    if (currVal > ngVal and bestVal > ngVal)
                    {
                        bestVal = ngVal;
                        index = i;
                    }

                    bin[i] = !bin[i];
                }

                if (index == -1) // bin is the local best
                    break;
                else
                    bin[index] = !bin[index];
            }
            ans = std::min(bestVal, ans);
        }
        return {ans, t.getTime()};
    }

    std::pair<double, double> runFirstImprove(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();

        for (int e = 1; e <= epochs; e++)
        {
            std::vector<bool> bin = genBin();
            double bestVal = eval(bin);
            while (true)
            {
                int i;
                for (i = 0; i < bin.size(); i++)
                {
                    bin[i] = !bin[i];

                    double ngVal = eval(bin); // value for the vector with the bit i flipped
                    if (bestVal > ngVal)
                    {
                        bestVal = ngVal;
                        break;
                    }
                    bin[i] = !bin[i];
                }
                if (i == bin.size())
                    break;
            }
            ans = std::min(bestVal, ans);
        }
        return {ans, t.getTime()};
    }

    void benchmark(int nrSamples, int epochs, HC_TYPE type = HC_TYPE::BEST)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);

        int index = 0;
        std::function<std::pair<double, double>(int)> runf;
        switch (type)
        {
        case HC_TYPE::BEST:
            std::cout << "Best Improvement:\n";
            runf = [this](int index)
            { return run(index); };
            index = 0;
            break;
        case HC_TYPE::FIRST:
            std::cout << "First Improvement:\n";
            runf = [this](int index)
            { return runFirstImprove(index); };
            index = 1;
            break;
        case HC_TYPE::SA:
            std::cout << "Simulated Anealing:\n";
            runf = [this](int index)
            { return simulatedAnealing(index); };
            index = 2;
            break;
        case HC_TYPE::WORST:
            std::cout << "Worst Improvement:\n";
            runf = [this](int index)
            { return run(index); };
            index = 3;
            break;
        }

        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::min();
            double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::min();
            for (int i = 0; i < nrSamples; i++)
            {
                auto [v, t] = runf(epochs);
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

int epochs = 1000;
void main_rast()
{
    //? p.fitness = std::pow(((maxFitness - p.fitness) / (maxFitness - minFitness + eps) + 1), 2);
    std::cout << "____________Rastrigin Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 10 * v.size();
        auto pi = std::acos(-1);
        for (int i = 0; i < v.size(); i++)
            ans += v[i] * v[i] - 10 * std::cos(2 * pi * v[i]);
        return ans;
    };
    // HillClimb hc{-5.12, 5.12, 5, f};

    // hc.benchmark(30, epochs);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::FIRST);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::SA);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::WORST);

    GeneticMain g{-5.12, 5.12, 5, f};
    g.benchmark(30);
    std::cout << '\n';
}

void main_micha()
{
    //? p.fitness = std::pow(((maxFitness - p.fitness) / (maxFitness - minFitness + eps) + 1), 3);

    std::cout << "____________Michalewics Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 0;
        int m = 10;
        auto pi = std::acos(-1);
        for (int i = 0; i < v.size(); i++)
            ans -= std::sin(v[i]) * std::pow(std::sin(((i + 1) * v[i] * v[i]) / pi), 2 * m);
        return ans;
    };

    // HillClimb hc{0, std::acos(-1), 5, f};
    // hc.trueValues = {-4.687658, -9.66015, -29.6308839};

    // hc.benchmark(30, epochs);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::FIRST);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::SA);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::WORST);

    GeneticMain g{0, std::acos(-1), 5, f};
    // g.trueValues = {-4.687658, -9.66015, -29.6308839}; // https://arxiv.org/pdf/2003.09867
    g.benchmark(30);

    std::cout << '\n';
}

void main_dejong()
{
    //? p.fitness = 1 / (p.fitness + eps);

    std::cout << "____________De Jong 1 Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 0;
        for (int i = 0; i < v.size(); i++)
            ans += v[i] * v[i];
        return ans;
    };

    // HillClimb hc{-5.12, 5.12, 5, f};
    // hc.benchmark(30, epochs);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::FIRST);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::SA);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::WORST);

    GeneticMain g{-5.12, 5.12, 5, f};
    g.benchmark(30);

    std::cout << '\n';
}

void main_schwefel()
{
    //? p.fitness = std::pow(((maxFitness - p.fitness) / (maxFitness - minFitness + eps) + 1), 2);

    std::cout << "____________Schwefel Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 418.9829 * v.size();

        for (int i = 0; i < v.size(); i++)
            ans -= v[i] * std::sin(std::sqrt(std::abs(v[i])));

        return ans;
    };
    // HillClimb hc{-500, 500, 5, f};

    // hc.benchmark(30, epochs);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::FIRST);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::SA);
    // hc.benchmark(30, epochs, HillClimb::HC_TYPE::WORST);

    GeneticMain g{-500, 500, 5, f};
    g.benchmark(30);

    std::cout << '\n';
}

// Todo remove f'
int main()
{
    // main_dejong();
    // main_rast();
    main_micha();
    // main_schwefel();
}
