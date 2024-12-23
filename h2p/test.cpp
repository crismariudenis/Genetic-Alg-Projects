#include <fstream>
#include <iostream>
#include <string>
#include <vector>
std::ifstream fin("logslogs");

int main()
{

    std::string s;
    double err = 0;
    double time = 0;
    int e = 0, t = 0;
    while (fin >> s)
    {
        double a, b;
        fin >> a >> b;
        if (s[0] == 'E')
        {
            err += a / b;
            e++;
        }
        else
            time += a / b, t++;
    }
    std::cout << err / e << " " << time / t << '\n';
}
/// 40% less error 190% slower