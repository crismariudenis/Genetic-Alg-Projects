#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>

using namespace std;

// Function to extract the largest number from a string
int extractLargestNumber(const string &str)
{
    regex re("\\d+");
    smatch match;
    int largestNumber = -1;

    string::const_iterator searchStart(str.cbegin());
    while (regex_search(searchStart, str.cend(), match, re))
    {
        int number = stoi(match[0]);
        if (number > largestNumber)
        {
            largestNumber = number;
        }
        searchStart = match.suffix().first;
    }

    return largestNumber;
}

int main()
{
    ifstream inputFile("date");
    if (!inputFile.is_open())
    {
        cerr << "Error opening file" << endl;
        return 1;
    }

    vector<string> strings;
    string line;
    while (getline(inputFile, line))
    {
        strings.push_back(line);
    }
    inputFile.close();

    // Sort the strings based on the largest number they contain
    sort(strings.begin(), strings.end(), [](const string &a, const string &b)
         { return extractLargestNumber(a) > extractLargestNumber(b); });

    // Print all the sorted strings
    for (const auto &str : strings)
    {
        cout << str << endl;
    }

    return 0;
}