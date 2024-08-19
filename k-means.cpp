#include <bits/stdc++.h>
#include <random>
#include <chrono>

using namespace std;
vector<vector<double>> xtrain;
double dcal(vector<double>&x,vector<double>&y)
{
    double result=0.0;
    for(int i=0; i<y.size(); i++)
    {
        result+=pow(x[i]-y[i],2);
    }
    return result;
}
class k_means
{

private:
    int n; // the number of rows
    int m; // the number of columns
    int k; // number of clusters
    vector<vector<double>> x; // coordinates of all points
    vector<int>nearstcluster;
    set<int>centroids;
public:
    k_means(int k,int n, int m, const vector<vector<double>>& input_x)
        : k(k), n(n), m(m), x(input_x),nearstcluster(vector<int>(n)) {}

    void random_choose_clusters()
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        mt19937 eng(seed);
        uniform_int_distribution<> distr(0, n-1);

        while(centroids.size()!=k)
        {
            centroids.insert(distr(eng));
        }
    }

    void choose_cluster()
    {
        for (int i = 0; i < n; i++)
        {
            double mind = 1e18;
            int minc = -1;
            for (int centroid : centroids)
            {
                double d = dcal(x[centroid], x[i]);
                if (d < mind)
                {
                    mind = d;
                    minc = centroid;
                }
            }
            nearstcluster[i] = minc;
        }

    }
    void update_centroids()
    {
        vector<vector<double>> new_centroids(k, vector<double>(m, 0.0));
        for(int centroid:centroids)
        {
            for(int i=0; i<m; i++)
            {
                double result=0.0;
                int s=0;
                for(int j=0; j<n; j++)
                {
                    if(nearstcluster[j]==centroid)
                    {
                        result+=x[j][i];
                        s++;
                    }
                }
                if(s>0)
                {
                    x[centroid][i]=result/s;

                }
            }
        }
    }
    void build(int t)
    {
        random_choose_clusters();
        while(t)
        {
            choose_cluster();
            update_centroids();
            t--;
        }
    }
    void get_clusters()
    {
        for (int i = 0; i < n; i++)
        {
            cout << "Point " << i << " is in cluster " << nearstcluster[i] << "\n";
        }
    }
};


signed main()
{
    // Example data points (4 points in 2D space)
    vector<vector<double>> data =
    {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0}
    };

    int n = data.size();
    int m = data[0].size();
    xtrain = data;

    // Create k-means object with 2 clusters
    k_means kmeans(2, n, m, xtrain);

    // Run the k-means algorithm for a fixed number of iterations
    kmeans.build(100000);

    // Print the cluster assignments
    kmeans.get_clusters();

    return 0;

}
