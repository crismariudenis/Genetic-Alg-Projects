#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <functional>
#include <chrono>
#include <iomanip>
#include <format>
#include <string>
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
    std::vector<int> dimensions = {2, 5, 10};

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
        for (int i = 0; i < v.size(); i++)
        {
            int val = 0;
            for (int j = 0; j < n; j++)
                val = (val << 1) + bin[i * n + j];
            v[i] = val / std::pow(10, p) + min;
        }
        return v;
    }

    double inline eval(const std::vector<double> &v)
    {
        return f(v);
    }

public:
    double data[20][20];
    HillClimb(double min, double max, int p, std::function<double(const std::vector<double> &)> f) : min(min), max(max), p(p), f(f)
    {
        N = (max - min) * std::pow(10, p);
        n = std::ceil(std::log2(N));
    }
    ~HillClimb()
    {
        std::vector<std::string> row = {"Minimum", "Maximum", "Average"};

        for (int i = 0; i < 18; i++, std::cout << '\n')
        {
            if (i % 3 == 0)
                std::cout << "D = " << dimensions[i / 3 % 3] << " & ";
            else
                std::cout << "     &   ";
            std::cout << std::format("{} {} & {:.5f} & {:.5f} \\\\{}", row[i % 3], (i / 9 ? "time" : "value"), data[i][0], data[i][1], i % 3 == 2 ? "\n" : "");
        }
    }

    std::pair<double, double> run(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();

        std::default_random_engine re(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> unif(min, max);

        std::vector<double> v(D); // Generate initial sample
        for (int e = 1; e <= epochs; e++)
        {
            for (auto &x : v)
                x = unif(re);
            std::vector<bool> bin = vecToBin(v);
            while (true)
            {
                std::vector<bool> ngh = bin;
                std::vector<bool> best = bin;
                auto V = binToVec(best);

                for (int i = 0; i < bin.size(); i++)
                {
                    if (i != 0)
                        ngh[i - 1] = !ngh[i - 1];

                    ngh[i] = !ngh[i];
                    if (eval(binToVec(best)) > eval(binToVec(ngh)))
                        best = ngh;
                }

                if (eval(binToVec(best)) < eval(binToVec(bin)))
                    bin = best;
                else
                    break; // bin is the local maximum
            }
            ans = std::min(eval(binToVec(bin)), ans);
        }
        return {ans, t.getTime()};
    }
    //Todo: abtract its own section but give number.
    // Todo: dont talk aobut the algorithms that the class knows. maybe in the last paragraph
    //Todo: stop copying and just cach bin value
    //Todo: find more tricks to not create more binary vectors
    //Todo: for D=2 9.66000 [de unde l-am luat]
    // Todo: add standdard deviasion daca nu spatoiu remove min & max. Median,distante inte quartila. DO IT IN R. maybe add it to makefile
    //Todo: change the title to numerical function
    std::pair<double, double> runFirstImprove(int epochs)
    {
        Timer t;
        double ans = std::numeric_limits<double>::max();

        std::default_random_engine re(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> unif(min, max);

        std::vector<double> v(D); // Generate initial sample
        for (int e = 1; e <= epochs; e++)
        {
            for (auto &x : v)
                x = unif(re);
            std::vector<bool> bin = vecToBin(v);
            while (true)
            {
                std::vector<bool> ngh = bin;
                std::vector<bool> oldBin = bin;
                for (int i = 0; i < bin.size(); i++)
                {
                    if (i != 0)
                        ngh[i - 1] = !ngh[i - 1];

                    ngh[i] = !ngh[i];
                    if (eval(binToVec(bin)) > eval(binToVec(ngh)))
                    {
                        bin = ngh;
                        break;
                    }
                }
                if (oldBin == bin)
                    break;
            }
            ans = std::min(eval(binToVec(bin)), ans);
        }
        return {ans, t.getTime()};
    }

    void benchmark(int nrSamples, int epochs)
    {
        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), totalT = 0, maxT = std::numeric_limits<double>::min();
            double minV = std::numeric_limits<double>::max(), totalV = 0, maxV = std::numeric_limits<double>::min();

            for (int i = 0; i < nrSamples; i++)
            {
                // auto [v, t] = runFirstImprove(epochs);
                auto [v, t] = run(epochs);

                minT = std::min(t, minT);
                totalT += t;
                maxT = std::max(t, maxT);

                minV = std::min(v, minV);
                totalV += v;
                maxV = std::max(v, maxV);
            }
            data[d * 3][0] = minV;
            data[d * 3 + 1][0] = maxV;
            data[d * 3 + 2][0] = totalV / nrSamples;

            data[9 + d * 3][0] = minT;
            data[9 + d * 3 + 1][0] = maxT;
            data[9 + d * 3 + 2][0] = totalT / nrSamples;

            std::cout << std::format("Dimensions: {}, Avg Time: {:.5f}, Min Time: {:.5f}, Max Time: {:.5f}\n", dimensions[d], totalT / nrSamples, minT, maxT);
            std::cout << std::format("Dimensions: {}, Avg Value: {:.5f}, Min Value: {:.5f}, Max Value: {:.5f}\n", dimensions[d], totalV / nrSamples, minV, maxV);
        }
        std::cout << '\n';
    }
    void benchmark2(int nrSamples, int epochs)
    {
        for (int d = 0; d < dimensions.size(); d++)
        {
            D = dimensions[d];

            double minT = std::numeric_limits<double>::max(), totalT = 0, maxT = std::numeric_limits<double>::min();
            double minV = std::numeric_limits<double>::max(), totalV = 0, maxV = std::numeric_limits<double>::min();

            for (int i = 0; i < nrSamples; i++)
            {
                auto [v, t] = runFirstImprove(epochs);
                // auto [v, t] = run(epochs);

                minT = std::min(t, minT);
                totalT += t;
                maxT = std::max(t, maxT);

                minV = std::min(v, minV);
                totalV += v;
                maxV = std::max(v, maxV);
            }
            data[d * 3][1] = minV;
            data[d * 3 + 1][1] = maxV;
            data[d * 3 + 2][1] = totalV / nrSamples;

            data[9 + d * 3][1] = minT;
            data[9 + d * 3 + 1][1] = maxT;
            data[9 + d * 3 + 2][1] = totalT / nrSamples;

            std::cout << std::format("Dimensions: {}, Avg Time: {:.5f}, Min Time: {:.5f}, Max Time: {:.5f}\n", dimensions[d], totalT / nrSamples, minT, maxT);
            std::cout << std::format("Dimensions: {}, Avg Value: {:.5f}, Min Value: {:.5f}, Max Value: {:.5f}\n", dimensions[d], totalV / nrSamples, minV, maxV);
        }
        std::cout << '\n';
    }
};

int epochs = 500;
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
    //std::cout << "First Improvement:\n";
    //hc.benchmark2(30, epochs);
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
    std::cout << "Best Improvement:\n";
    hc.benchmark(30, epochs);
    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);

    std::cout << '\n';
}

void main_dixon()
{
    std::cout << "____________Dixon & Price Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = (v[0] - 1) * (v[0] - 1);

        for (int i = 1; i < v.size(); i++)
        {
            ans += (i + 1) * (2 * v[i] * v[i] - v[i - 1]) * (2 * v[i] * v[i] - v[i - 1]);
        }
        return ans;
    };

    HillClimb hc{-10, 10, 5, f};
    std::cout << "Best Improvement:\n";
    hc.benchmark(30, epochs);
    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);

    std::cout << '\n';
}

void main_griewank()
{
    std::cout << "____________Griewank Function____________\n";
    auto f = [](const std::vector<double> &v)
    {
        double ans = 1;

        for (int i = 0; i < v.size(); i++)
            ans += v[i] * v[i];
        ans /= 4000;
        double prod = 1;
        for (int i = 0; i < v.size(); i++)
            prod *= std::cos(v[i] / std::sqrt(i + 1));
        return ans - prod;
    };

    HillClimb hc{-600, 600, 5, f};
    std::cout << "Best Improvement:\n";
    hc.benchmark(30, epochs);
    std::cout << "First Improvement:\n";
    hc.benchmark2(30, epochs);

    std::cout << '\n';
}

int main()
{
    main_rast();
    //main_micha();
   // main_dixon();
    //main_griewank();
}
