package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.FunctionParameter;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.OptimizerParameter;

/**
 * [Serializable]
 */
public class Adam extends Optimizer {
	private static final long serialVersionUID = 2087321425467232653L;
	final public double Alpha;
	final public double Beta1;
	final public double Beta2;
	final public double Epsilon;

	private long UpdateCount = 1;
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	private Adam(Adam i_src) {
		super(i_src);
		this.Alpha = i_src.Alpha;
		this.Beta1 = i_src.Beta1;
		this.Beta2 = i_src.Beta2;
		this.Epsilon = i_src.Epsilon;
	}

	public Adam() {
		this(0.001, 0.9, 0.999, 1e-8);
	}

	public Adam(double alpha, double beta1, double beta2, double epsilon) {
		super();
		this.Alpha = alpha;
		this.Beta1 = beta1;
		this.Beta2 = beta2;
		this.Epsilon = epsilon;
	}

	public void initilise(FunctionParameter[] i_functionParameters) {
		this.optimizerParameters = new OptimizerParameter[i_functionParameters.length];

		for (int i = 0; i < this.optimizerParameters.length; i++) {
			this.optimizerParameters[i] = new AdamParameter(
					i_functionParameters[i], this);
		}
	}
	@Override
	public Optimizer deepCopy() {
		return new Adam(this);
	}
	private class AdamParameter extends OptimizerParameter {
		private static final long serialVersionUID = -2950834706490917841L;

		final private Adam optimiser;

		private final double[] m;
		private final double[] v;

		public AdamParameter(AdamParameter i_src) {
			super(i_src);
			this.optimiser = (Adam) i_src.optimiser.deepCopy();
			this.m = i_src.m.clone();
			this.v = i_src.v.clone();
		}

		public AdamParameter(FunctionParameter parameter, Adam optimiser) {
			super(parameter);
			this.m = new double[parameter.length()];
			this.v = new double[parameter.length()];
			this.optimiser = optimiser;
		}

		@Override
		public void updateFunctionParameters() {
			double fix1 = 1 - Math.pow(this.optimiser.Beta1,this.optimiser.UpdateCount);
			double fix2 = 1 - Math.pow(this.optimiser.Beta2,this.optimiser.UpdateCount);
			double lr = this.optimiser.Alpha * Math.sqrt(fix2) / fix1;

			for (int i = 0; i < this.functionParameters.length(); i++) {
				double grad = this.functionParameters.grad.data[i];

				this.m[i] += (1 - this.optimiser.Beta1) * (grad - this.m[i]);
				this.v[i] += (1 - this.optimiser.Beta2)
						* (grad * grad - this.v[i]);

				this.functionParameters.param.data[i] -= lr * this.m[i]
						/ (Math.sqrt(this.v[i]) + this.optimiser.Epsilon);
			}
			return;
		}

		@Override
		public Object deepCopy() {
			return new AdamParameter(this);
		}
	}


}
