// This is a simple example to showcase the use of this class, this example give an approximate for the relation y=2*x

#include <bits/stdc++.h>

using namespace std;
int rows,cols;
vector<vector<double>> xtrain;
vector<double> ytrain;

class linear_regression_class
{
private:
    int n, m;
    vector<vector<double>> x;
    vector<vector<double>> w;
    vector<double> y;
    double b=0.0;
    double alpha;

public:
    linear_regression_class(int n, int m, const vector<vector<double>>& input_x, const vector<double>& input_y, double alpha)
        : n(n), m(m), x(input_x), w(1, vector<double>(m)), y(input_y), alpha(alpha) {}

    double f(vector<double>xx, int o)
    {
        double result = 0.0;
        for(int i=0; i<o; i++)
        {
            result+=xx[i]*w[0][i];
        }
        result+=b;

        return result;
    }
    double J()
    {
        double result=0.0;
        for(int i=0; i<n; i++)
        {
            double est=f(x[i],m);
            result+=pow((est-y[i]),2);
        }
        result/=n;
        return result;
    }
    double dJb()
    {
        double result=0.0;
        for(int i=0; i<n; i++)
        {
            double est=f(x[i],m);
            result+=(est-y[i]);
        }
        result*=2;
        result/=n;
        return result;
    }
    vector<double> dJw()
    {
        vector<double>finalresult(m);
        double result;
        for(int j=0; j<m; j++)
        {
            result=0.0;
            for(int i=0; i<n; i++)
            {
                double est=f(x[i],m);
                result+=(est-y[i])*x[i][j];
            }
            result*=2;
            result/=n;
            finalresult[j]=result;
        }
        return finalresult;
    }
    void update()
    {
        b= b-alpha*dJb();
        vector<double>djw(m);
        djw=dJw();
        for(int i=0; i<m; i++)
        {
            w[0][i]= w[0][i]-alpha*djw[i];
        }
    }
    void train(int t)
    {
        while(t)
        {
            update();
            t--;
        }
    }
};
void load_data()
{
    rows = 4;
    cols = 1;
    int k = 0;
    vector<vector<double>> xxtrain(rows, vector<double>(cols));
    vector<double> yytrain(rows);

    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            xxtrain[j][i] = k;
            k++;
        }
    }
    for (int j = 0; j < rows; j++)
    {
        yytrain[j] = j*2;
    }
    xtrain=xxtrain;
    ytrain=yytrain;
}
signed main()
{
    load_data();
    linear_regression_class test(rows, cols, xtrain, ytrain,0.005);
    test.train(1200);

    cout <<test.f({1},cols) << endl;

    return 0;
}
