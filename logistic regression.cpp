// This example shows the implementation and a simple usage of the Logistic Regression class, this example classifies the positive and negative numbers (it is just a very simple example)
#include <bits/stdc++.h>

using namespace std;
int rows,cols;
vector<vector<double>> xtrain;
vector<double> ytrain;
double sig(double k)
{
    return 1/(1+exp(-k));
}
class logistic_regression_class
{
private:
    int n, m;
    vector<vector<double>> x;
    vector<vector<double>> w;
    vector<double> y;
    double b=0.0;
    double alpha;

public:
    logistic_regression_class(int n, int m, const vector<vector<double>>& input_x, const vector<double>& input_y, double alpha)
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
            est=sig(est);
            result+=y[i]*log(est)+(1-y[i])*log(1-est);
        }
        result/=(-n);
        return result;
    }

    double dJb()
    {
        double result=0.0;
        for(int i=0; i<n; i++)
        {
            double est=f(x[i],m);
            est=sig(est);
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
                est=sig(est);
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
    rows = 20;
    cols = 1;
    int k = 0;
    vector<vector<double>> xxtrain(rows, vector<double>(cols));
    vector<double> yytrain(rows);

    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            xxtrain[j][i] = k;
            k++;
        }
        k=0;
        for(int j=10;j<rows;j++)
        {
            xxtrain[j][i]=k;
            k--;
        }
    }
    for (int j = 0; j < rows/2; j++)
    {
        yytrain[j] = 1;
    }
    for(int j=rows/2;j<rows;j++)
    {
        yytrain[j]=0;
    }
    xtrain=xxtrain;
    ytrain=yytrain;
}
signed main()
{
    load_data();
    logistic_regression_class test(rows, cols, xtrain, ytrain,0.005);
    test.train(120);

    cout <<sig(test.f({999},cols)) << endl;

    return 0;
}
