package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.OptimizerParameter;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.optimizers.common.Optimizer;


public class RMSprop extends Optimizer{
	/**
	 * 
	 */
	private static final long serialVersionUID = 4696404453338019276L;
	final public double LearningRate;
	final public double Alpha;
	final public double Epsilon;

	public RMSprop(RMSprop i_src) {
		super(i_src);
		this.LearningRate = i_src.LearningRate;
		this.Alpha = i_src.Alpha;
		this.Epsilon = i_src.Epsilon;
	}

	public RMSprop() {
		this(0.01, 0.99, 1e-8);
	}

	public RMSprop(double i_learningRate, double i_alpha, double i_epsilon) {
		super();
		this.LearningRate = i_learningRate;
		this.Alpha = i_alpha;
		this.Epsilon = i_epsilon;
	}



	@Override
	public void addFunctionParameters(FunctionParameter[] i_functionParameters) {
		this.optimizerParameters = new OptimizerParameter[i_functionParameters.length];

		for (int i = 0; i < this.optimizerParameters.length; i++) {
			this.optimizerParameters[i] = new RMSpropParameter(
					i_functionParameters[i], this);
		}
	}

	@Override
	public Optimizer deepCopy() {
		return new RMSprop(this);
	}


	private class RMSpropParameter extends OptimizerParameter {

		private static final long serialVersionUID = 1861252226788441316L;
		final private RMSprop optimiser;
		final private double[] ms;

		public RMSpropParameter(RMSpropParameter i_src) {
			super(i_src);
			this.optimiser = (RMSprop) i_src.deepCopy();
			this.ms = i_src.ms.clone();
		}

		public RMSpropParameter(FunctionParameter parameter, RMSprop optimiser) {
			super(parameter);
			this.optimiser = optimiser;
			this.ms = new double[parameter.length()];
		}

		public void updateFunctionParameters() {
			for (int i = 0; i < this.functionParameters.length(); i++) {
				double grad = this.functionParameters.grad.data[i];
				this.ms[i] *= this.optimiser.Alpha;
				this.ms[i] += (1 - this.optimiser.Alpha) * grad * grad;

				this.functionParameters.param.data[i] -= this.optimiser.LearningRate
						* grad
						/ (Math.sqrt(this.ms[i]) + this.optimiser.Epsilon);
			}
		}

		@Override
		public Object deepCopy() {
			return new RMSpropParameter(this);
		}
	}

}