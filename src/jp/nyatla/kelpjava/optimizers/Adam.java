package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.IOptimizer;
import jp.nyatla.kelpjava.OptimizeParameter;

//[Serializable]
public class Adam implements IOptimizer {
	final public double Alpha;
	final public double Beta1;
	final public double Beta2;
	final public double Epsilon;

	private long UpdateCount = 1;

	final private double[] m;
	final private double[] v;

	public Adam() {
		this(0.001, 0.9, 0.999, 1e-8, 0);
	}

	public Adam(double alpha, double beta1, double beta2, double epsilon,
			int parameterLength) {
		this.Alpha = alpha;
		this.Beta1 = beta1;
		this.Beta2 = beta2;
		this.Epsilon = epsilon;

		this.m = new double[parameterLength];
		this.v = new double[parameterLength];
	}

	@Override
	public IOptimizer initialise(OptimizeParameter i_parameter) {
		return new Adam(this.Alpha, this.Beta1, this.Beta2, this.Epsilon,
				i_parameter.length());
	}

	@Override
	public void update(OptimizeParameter parameter) {
		double fix1 = 1 - Math.pow(this.Beta1, this.UpdateCount);
		double fix2 = 1 - Math.pow(this.Beta2, this.UpdateCount);
		double lr = this.Alpha * Math.sqrt(fix2) / fix1;

		for (int i = 0; i < parameter.length(); i++) {
			double grad = parameter.grad.data[i];

			this.m[i] += (1 - this.Beta1) * (grad - this.m[i]);
			this.v[i] += (1 - this.Beta2) * (grad * grad - this.v[i]);

			parameter.param.data[i] -= lr * this.m[i]
					/ (Math.sqrt(this.v[i]) + this.Epsilon);
		}

		this.UpdateCount++;
	}
}
