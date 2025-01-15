#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <format>
class Graph
{

    enum TYPE
    {
        EUC_2D,
        CEIL_2D
    } type;

    inline int nint(double x)
    {
        return (int)(x + 0.5);
    }

public:
    std::vector<std::pair<double, double>> data;
    Graph() = default;
    Graph(int size, std::string t)
    {
        data.resize(size);
        if (t == "EUC_2D")
            type = EUC_2D;
        else if (t == "CEIL_2D")
            type = CEIL_2D;
    }

    int dist(int node1, int node2)
    {
        double xd, yd;
        switch (type)
        {
        case CEIL_2D:
            xd = data[node1].first - data[node2].first;
            yd = data[node1].second - data[node2].second;
            return std::ceil(std::sqrt(xd * xd + yd * yd));

        default:
            xd = data[node1].first - data[node2].first;
            yd = data[node1].second - data[node2].second;
            return nint(std::sqrt(xd * xd + yd * yd));
        }
    }
    size_t size()
    {
        return data.size();
    }
    auto &operator[](int idx)
    {
        return data[idx];
    }
};

class TSP
{
    std::string file;
    std::ifstream fin;
    struct Header
    {
        std::string name;
        std::string comment;
        std::string type;
        uint32_t dimensions;
        std::string edge_weight_type;
    } header;

    std::string trim(const std::string &str)
    {
        size_t first = str.find_first_not_of(" \t");
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, (last - first + 1));
    }

    void parseHeader()
    {
        std::string line;
        while (std::getline(fin, line))
        {
            if (line.empty())
                continue;

            if (line == "NODE_COORD_SECTION")
                break;

            std::istringstream iss(line);

            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos)
            {
                std::string key = trim(line.substr(0, colon_pos));
                std::string value = trim(line.substr(colon_pos + 1));

                if (key == "NAME")
                    header.name = value;
                else if (key == "COMMENT")
                    header.comment = value;
                else if (key == "TYPE")
                    header.type = value;
                else if (key == "DIMENSION")
                    header.dimensions = std::stoi(value);
                else if (key == "EDGE_WEIGHT_TYPE")
                    header.edge_weight_type = value;
            }
        }
    }

    void parseGraph()
    {
        graph = Graph(header.dimensions, header.edge_weight_type);
        for (int i = 0; i < header.dimensions; i++)
        {
            int index;
            fin >> index >> graph[i].first >> graph[i].second;
        }
    }

public:
    void printHeader() const
    {
        std::cout << "NAME: " << header.name << std::endl;
        std::cout << "COMMENT: " << header.comment << std::endl;
        std::cout << "TYPE: " << header.type << std::endl;
        std::cout << "DIMENSION: " << header.dimensions << std::endl;
        std::cout << "EDGE_WEIGHT_TYPE: " << header.edge_weight_type << std::endl;
    }
    void printNodes() const
    {

        for (int i = 0; i < graph.data.size(); i++)
        {
            std::cout << i + 1 << " " << graph.data[i].first << " " << graph.data[i].second << '\n';
        }
    }

public:
    Graph graph;
    TSP() = default;
    void init(std::string file)
    {
        this->file = file;
        fin = std::ifstream(file);
        parseHeader();
        parseGraph();
        assert(graph.size() != 0);
    }
    ~TSP()
    {
        if (fin.is_open())
        {
            fin.close(); // Close the file explicitly
        }
    }
};