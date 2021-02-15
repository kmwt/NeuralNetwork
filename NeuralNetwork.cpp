#include <iostream>
#include<cmath>
#include<vector>
#include<random>
#include <limits>
using namespace std;

vector<double> operator+(const vector<double>& x, const vector<double>& y) {
	vector<double> z(x.size());
	for (size_t d = 0; d < x.size(); d++) z[d] = x[d] + y[d];
	return z;
}
vector<double> operator-(const vector<double>& x, double y) {
	vector<double> z(x.size());
	for (size_t d = 0; d < x.size(); d++)
		z[d] = x[d] - y;
	return z;
}
vector<double> operator-(const vector<double>& x) {
	vector<double> z(x.size());
	for (size_t d = 0; d < x.size(); d++) z[d] = -x[d];
	return z;
}
vector<double> operator-(const vector<double>& x, const vector<double>& y) {
	return x + (-y);
}
vector<double> operator*(double a, const vector<double>& x) {
	vector<double> z(x.size());
	for (size_t d = 0; d < z.size(); d++) z[d] = a * x[d];
	return z;
}
vector<double> operator*(const vector<double>& x, const vector<double>& y) {
	vector<double> z(x.size());
	for (size_t d = 0; d < x.size(); d++) z[d] = x[d] * y[d];
	return z;
}
vector<double> operator/(const vector<double>& x, double a) {
	vector<double> z(x.size());
	for (size_t d = 0; d < x.size(); d++) z[d] = x[d] / a;
	return z;
}
vector<double> operator+(const vector<double>& x, double a) {
    vector<double> z(x.size());
    for (size_t d = 0; d < x.size(); d++) z[d] = x[d] + a;
    return z;
}
vector<double> operator/(const vector<double>& x, const vector<double>& y) {
    vector<double> z(x.size());
    for (size_t d = 0; d < x.size(); d++) z[d] = x[d] / y[d];
    return z;
}

double Sum(const vector<double>& x) {
	double z = 0.;
	for (size_t d = 0; d < x.size(); d++) z += x[d];
	return z;
}

double Dot(const vector<double>& x, const vector<double>& y) {
	return Sum(x * y);
}

class Activation {
public:
	virtual double f(double x) const { return x; }
	virtual double df(double x) const { return 1.; }
	double operator()(double x) const { return f(x); }
};
class Linear : public Activation{};
class Sigmoid : public  Activation {
public:
	virtual double f(double x) const override {
		return 1. / (1. + exp(-x));
	}
	virtual double df(double x) const override {
		double p = f(x);
		return p * (1. - p);
	}
};

class Error {
public:
	virtual double f(double y, double p) const {
		double error = y - p;
		return error * error / 2.;
	}
	virtual double df(double y, double p) const {
		return p - y;
	}
	double f(const vector<double>& Y, const vector<double>& P) const {
		double error = 0.;
		for (size_t n = 0; n < Y.size(); n++) error += f(Y[n], P[n]);
		return error / ((double)Y.size());
	}
	double operator()(const vector<double>& Y, const vector<double>& P) const {
		return f(Y, P);
	}
};
class MSE : public Error {};
class BCE : public Error {
private:
	double __epsilon;
public:
	BCE() { __epsilon = numeric_limits<double>::epsilon(); }
	virtual double f(double y, double p) const override {
		if (0. < p && p < 1.) return -(y * log(p) + (1. - y) * log(1. - p));
		else if (p == 0) p = __epsilon;
		else if (p == 1) p = 1. - __epsilon;
		return -(y * log(p) + (1. - y) * log(1. - p));
	}
	virtual double df(double y, double p) const override {
		if (0. < p && p < 1.) return -y / p + (1. - y) / (1. - p);
		else if (p == 0) p = __epsilon;
		else if (p == 1) p = 1. - __epsilon;
		return -y / p + (1. - y) / (1. - p);
	}
};

class Optimizer {
protected:
	vector<double> __lr_w;
	double __lr_b;
public:
	virtual void set_length(size_t  length) {
		for (size_t d = 0; d < length; d++)
			__lr_w.push_back(__lr_b);
	}
	virtual vector<double> operator()(const vector<double>& grad) { return __lr_w * grad; }
	virtual double operator()(double grad) { return __lr_b * grad; }
};
class GD : public Optimizer {
public:GD(double lr = 0.01) { __lr_b = lr; }
};
class Adam : public Optimizer {
private:
	vector<double> __sqrt(vector<double> x) {
		vector<double> y(x.size());
		for (size_t d = 0; d < y.size(); d++) y[d] = sqrt(x[d]);
		return y;
	}
protected:
	double __beta_1, __beta_2, __epsilon;
	int __t_w, __t_b;
	vector<double> __m_w, __v_w;
	double __m_b, __v_b;
public:
	Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8) {
		__lr_b = lr;
		__beta_1 = beta_1;
		__beta_2 = beta_2;
		__epsilon = epsilon;

		__t_w = __t_b = 0;
		__m_b = __v_b = 0;
	}
	void set_length(size_t  length) override {
		for (size_t d = 0; d < length; d++) {
			__m_w.push_back(0);
			__v_w.push_back(0);
		}
	}
	vector<double> operator()(const vector<double>& grad) override {
		__t_w++;
		__m_w = __beta_1 * __m_w + (1. - __beta_1) * grad;
		__v_w = __beta_2 * __v_w + (1. - __beta_2) * grad * grad;
		vector<double> m = __m_w / (1. - pow(__beta_1, __t_w));
		vector<double> v = __v_w / (1. - pow(__beta_2, __t_w));
		return __lr_b * m / __sqrt(v + __epsilon);
	}
	double operator()(double grad) override {
		__t_b++;
		__m_b = __beta_1 * __m_b + (1. - __beta_1) * grad;
		__v_b = __beta_2 * __v_b + (1. - __beta_2) * grad * grad;
		double m = __m_b / (1. - pow(__beta_1, __t_b));
		double v = __v_b / (1. - pow(__beta_2, __t_b));
		double tmp = __lr_b * m / sqrt(v + __epsilon);
		return tmp;
	}
};

class Unit {
protected:
	vector<double> __weight;
	double __bias;
	size_t __length;
	Activation* __activation;
	Error* __error;
	Optimizer* __optimizer;
	void __init__(const vector<double>&, double, Activation&);
	void __update(const vector<double >&, double);
	double __sum(const vector<double>&) const;
public:
	Unit(size_t, Activation&);
	double predict(const vector<double>&) const;
	vector<double> predict(const vector<vector<double> >&) const;
	void compile( Optimizer&,  Error&);
	double train_on_batch(const vector<vector<double> >&, const vector<double>&);
	double fit(const vector<vector<double> >&, const vector<double>&, size_t = 1, size_t = 1);
	friend ostream& operator<<(ostream&, const Unit&);
};

Unit::Unit(size_t length, Activation& activation) {
	__length = length;
	random_device seed_gen;
	default_random_engine engine(seed_gen());
	normal_distribution<> dist(0., 1.);
	__bias = dist(engine);
	vector<double> weight;
	for (size_t d = 0; d < __length; d++)
		weight.push_back(dist(engine));
	__init__(weight, dist(engine), activation);
}
void Unit::__init__(const vector<double>& weight, double bias, Activation& activation) {
	__weight = weight;
	__activation = &activation;
	__length = weight.size();
}

double Unit::__sum(const vector<double>& x) const {
	return Dot(__weight, x) + __bias;
}
double Unit::predict(const vector<double>& x) const {
	Activation& activation = *__activation;
	return activation(__sum(x));
}
vector<double> Unit::predict(const vector<vector<double>>& X) const {
	vector<double> P(X.size());
	for (size_t n = 0; n < X.size(); n++) P[n] = predict(X[n]);
	return P;
}
void Unit::compile(Optimizer& optimizer, Error& error) {
	optimizer.set_length(__length);
	__optimizer = &optimizer;
	__error = &error;
}

double Unit::fit(const vector<vector<double> >& X, const vector<double>& Y, size_t epochs, size_t verbose) {
	Error& error = *__error;
	for (size_t e = 0; e < epochs; e++){
		double error = train_on_batch(X, Y);
		if(verbose>0) cout << (e + 1) << ": \t" << error << "\n";
	}
	return error(Y, predict(X));
}
double Unit::train_on_batch(const vector<vector<double> >& X, const vector<double>& Y) {
	Error& error = *__error;
	Activation& activation = *__activation;
	vector<double> P = predict(X);
	vector<double> grad_w(__length, 0.);
	double grad_b = 0.;
	for (size_t n = 0; n < Y.size(); n++) {
		double grad = error.df(Y[n], P[n]) * activation.df(__sum(X[n]));
		grad_w = grad_w + grad * X[n];
		grad_b = grad_b + grad;
	}
	__update(grad_w, grad_b);
	return error(Y, predict(X));
}
void Unit::__update(const vector<double >& grad_w, double grad_b) {
	Optimizer& optimizer = *__optimizer;
	__weight = __weight - optimizer(grad_w);
	__bias = __bias - optimizer(grad_b);
}

ostream& operator<<(ostream& out, const Unit& unit) {
	out << "weights\n";
	for (size_t w = 0; w < unit.__length; w++)
		out << unit.__weight[w] << "\t";
	out << "\n";
	out << "bias: \t" << unit.__bias;
	return out;
}

void test(Activation& activation, Error& error, const size_t N = 100, double std = 1.) {

	random_device seed_gen;
	default_random_engine engine(seed_gen());
	normal_distribution<> dist(0., std);

	GD opt;
	Sigmoid sigmoid;

	vector<vector<double> > X;
	vector<double> Y;

	for (int n = 0; n < N; n++) {
		double x = dist(engine), y = dist(engine);
		double sum = x / 2. - y / 2 - 2;
		X.push_back({ x, y });
		Y.push_back(sigmoid(sum));
	}

	Unit model(2, activation);
	model.compile(opt, error);
	model.fit(X, Y, 200, 0);
	cout << model;
}


int main() {
	Linear linear;
	MSE mse;

	test(linear, mse);
	cout << "\n\n";
	Sigmoid sigmoid;
	BCE bce;

	test(sigmoid, bce);
}