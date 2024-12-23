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
    std::vector<double> trueValues = {0, 0, 0};
    double data[64][64];
    HillClimb(double min, double max, int p, std::function<double(const std::vector<double> &)> f) : min(min), max(max), p(p), f(f)
    {
        N = (max - min) * std::pow(10, p);
        n = std::ceil(std::log2(N));
        std::mt19937_64 temp;
        temp.seed(std::chrono::system_clock::now().time_since_epoch().count() * 1000);
        temp.discard(69420);
        gen.seed(temp());

        genBin();
    }
    ~HillClimb()
    {
        std::vector<std::string> row = {"Average", "Standard", "Min", "Max"};

        for (int i = 0; i < 32; i++, std::cout << '\n')
        {
            if (i % 4 == 0)
                std::cout << "D = " << dimensions[i / 4 % 3] << " & ";
            else
                std::cout << "     &   ";
            std::cout << std::format("{} {} & {:.5f} & {:.5f} & {:.5f} \\\\{}", row[i % 4], (i / 12 ? "time" : "error"), data[i][0], data[i][1], data[i][2], i % 4 == 3 ? "\n" : "");
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

    std::pair<double, double> runFirstImprove(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();

        std::default_random_engine re(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> unif(min, max);

        std::vector<double> v(D); // Generate initial sample
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

    void benchmark(int nrSamples, int epochs)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);
        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::min();
            double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::min();
            for (int i = 0; i < nrSamples; i++)
            {
                auto [v, t] = run(epochs);
                errors[i] = std::abs(v - trueValues[d]);
                times[i] = t;
                minE = std::min(minE, errors[i]);
                maxE = std::max(maxE, errors[i]);

                minT = std::min(minT, times[i]);
                maxT = std::max(maxT, times[i]);
            }
            data[d * 4][0] = median(errors);
            data[d * 4 + 1][0] = sd(errors);
            data[d * 4 + 2][0] = minE;
            data[d * 4 + 3][0] = maxE;

            data[12 + d * 4][0] = median(times);
            data[12 + d * 4 + 1][0] = sd(times);
            data[12 + d * 4 + 2][0] = minT;
            data[12 + d * 4 + 3][0] = maxT;

            std::cout << std::format("Dimensions: {}, Avg Error: {:.5f}, Standard Error: {:.5f}, Min Error: {:.5f}, Max Error: {:.5f}\n", dimensions[d], data[d * 4][0], data[d * 4 + 1][0], minE, maxE);
            std::cout << std::format("Dimensions: {}, Avg Time: {:.5f}, Standard Time: {:.5f}, Min Time: {:.5f}, Max Time: {:.5f}\n", dimensions[d], data[12 + d * 4][0], data[12 + d * 4 + 1][0], minT, maxT);
        }
        std::cout << '\n';
    }
    void benchmark2(int nrSamples, int epochs)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);
        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::min();
            double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::min();

            for (int i = 0; i < nrSamples; i++)
            {
                auto [v, t] = runFirstImprove(epochs);
                errors[i] = std::abs(v - trueValues[d]);
                times[i] = t;
                minE = std::min(minE, errors[i]);
                maxE = std::max(maxE, errors[i]);

                minT = std::min(minT, times[i]);
                maxT = std::max(maxT, times[i]);
            }
            data[d * 4][1] = median(errors);
            data[d * 4 + 1][1] = sd(errors);
            data[d * 4 + 2][1] = minE;
            data[d * 4 + 3][1] = maxE;

            data[12 + d * 4][1] = median(times);
            data[12 + d * 4 + 1][1] = sd(times);
            data[12 + d * 4 + 2][1] = minT;
            data[12 + d * 4 + 3][1] = maxT;

            std::cout << std::format("Dimensions: {}, Avg Error: {:.5f}, Standard Error: {:.5f}, Min Error: {:.5f}, Max Error: {:.5f}\n", dimensions[d], data[d * 4][1], data[d * 4 + 1][1], minE, maxE);
            std::cout << std::format("Dimensions: {}, Avg Time: {:.5f}, Standard Time: {:.5f}, Min Time: {:.5f}, Max Time: {:.5f}\n", dimensions[d], data[12 + d * 4][1], data[12 + d * 4 + 1][1], minT, maxT);
        }
        std::cout << '\n';
    }
    void benchmark3(int nrSamples, int epochs)
    {
        std::vector<double> errors(nrSamples), times(nrSamples);
        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), maxT = std::numeric_limits<double>::min();
            double minE = std::numeric_limits<double>::max(), maxE = std::numeric_limits<double>::min();

            for (int i = 0; i < nrSamples; i++)
            {
                auto [v, t] = simulatedAnealing(epochs);
                errors[i] = std::abs(v - trueValues[d]);
                times[i] = t;
                minE = std::min(minE, errors[i]);
                maxE = std::max(maxE, errors[i]);

                minT = std::min(minT, times[i]);
                maxT = std::max(maxT, times[i]);
            }
            data[d * 4][2] = median(errors);
            data[d * 4 + 1][2] = sd(errors);
            data[d * 4 + 2][2] = minE;
            data[d * 4 + 3][2] = maxE;

            data[12 + d * 4][2] = median(times);
            data[12 + d * 4 + 1][2] = sd(times);
            data[12 + d * 4 + 2][2] = minT;
            data[12 + d * 4 + 3][2] = maxT;

            std::cout << std::format("Dimensions: {}, Avg Error: {:.5f}, Standard Error: {:.5f}, Min Error: {:.5f}, Max Error: {:.5f}\n", dimensions[d], data[d * 4][2], data[d * 4 + 1][2], minE, maxE);
            std::cout << std::format("Dimensions: {}, Avg Time: {:.5f}, Standard Time: {:.5f}, Min Time: {:.5f}, Max Time: {:.5f}\n", dimensions[d], data[12 + d * 4][2], data[12 + d * 4 + 1][2], minT, maxT);
        }
        std::cout << '\n';
    }
};

int epochs = 1000;
void main_rast()
{
    std::cout << "____________Rastrigin Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 10 * v.size();
        auto pi = std::acos(-1);
        for (int i = 0; i < v.size(); i++)
            ans += v[i] * v[i] - 10 * std::cos(2 * pi * v[i]);
        return ans;
    };
    HillClimb hc{-5.12, 5.12, 5, f};
    std::cout << "Best Improvement:\n";
    hc.benchmark(30, epochs);
    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);
    std::cout << "Simulated Anealing:\n";
    hc.benchmark3(30, epochs);
    std::cout << '\n';
}

void main_micha()
{
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

    HillClimb hc{0, std::acos(-1), 5, f};
    // https://arxiv.org/pdf/2003.09867
    hc.trueValues = {-4.687658, -9.66015, -29.6308839};

    std::cout << "Best Improvement:\n";
    hc.benchmark(30, epochs);
    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);
    std::cout << "Simulated Anealing:\n";
    hc.benchmark3(30, epochs);
    std::cout << '\n';
}

void main_dejong()
{
    std::cout << "____________De Jong 1 Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 0;
        for (int i = 0; i < v.size(); i++)
            ans += v[i] * v[i];
        return ans;
    };

    HillClimb hc{-5.12, 5.12, 5, f};
    std::cout << "Best Improvement:\n";
    hc.benchmark(30, epochs);
    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);
    std::cout << "Simulated Anealing:\n";
    hc.benchmark3(30, epochs);
    std::cout << '\n';
}

void main_schwefel()
{
    std::cout << "____________Schwefel Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 418.9829 * v.size();

        for (int i = 0; i < v.size(); i++)
            ans -= v[i] * std::sin(std::sqrt(std::abs(v[i])));

        return ans;
    };
    HillClimb hc{-500, 500, 5, f};

    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);
    // std::cout << "Simulated Anealing:\n";
    // hc.benchmark3(30, epochs);
    // std::cout << "Best Improvement:\n";
    // hc.benchmark(30, epochs);
    // std::cout << '\n';
}

int main()
{
    // main_rast();
    // main_micha();
    // main_dejong();
    main_schwefel();
}
