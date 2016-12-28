package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.OptimizerParameter;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.optimizers.common.Optimizer;

public class AdaDelta extends Optimizer {
	
	private static final long serialVersionUID = 2884677339340543950L;
	public double Rho;
	public double Epsilon;

	public AdaDelta(AdaDelta i_src) {
		super(i_src);
		this.Rho = i_src.Rho;
		this.Epsilon = i_src.Epsilon;
	}

	public AdaDelta() {
		this(0.95, 1e-6);
	}

	public AdaDelta(double i_rho, double i_epsilon) {
		super();
		this.Rho = i_rho;
		this.Epsilon = i_epsilon;
	}

	@Override
	public void addFunctionParameters(FunctionParameter[] i_functionParameters) {
		this.optimizerParameters = new OptimizerParameter[i_functionParameters.length];

		for (int i = 0; i < this.optimizerParameters.length; i++) {
			this.optimizerParameters[i] = new AdaDeltaParameter(
					i_functionParameters[i], this);
		}
	}

	@Override
	public Optimizer deepCopy() {
		return new AdaDelta(this);
	}

	private class AdaDeltaParameter extends OptimizerParameter {
		private static final long serialVersionUID = 5472656307658516720L;
		final private double[] msg;
		final private double[] msdx;
		final private AdaDelta optimiser;

		public AdaDeltaParameter(AdaDeltaParameter i_src) {
			super(i_src);
			this.msg = i_src.msg.clone();
			this.msdx = i_src.msdx.clone();
			this.optimiser = (AdaDelta) i_src.optimiser.deepCopy();
		}

		public AdaDeltaParameter(FunctionParameter functionParameter,
				AdaDelta optimiser) {
			super(functionParameter);
			this.msg = new double[functionParameter.length()];
			this.msdx = new double[functionParameter.length()];
			this.optimiser = optimiser;
		}

		@Override
		public void updateFunctionParameters() {
			for (int i = 0; i < this.functionParameters.length(); i++) {
				double grad = this.functionParameters.grad.data[i];
				this.msg[i] *= this.optimiser.Rho;
				this.msg[i] += (1 - this.optimiser.Rho) * grad * grad;

				double dx = Math.sqrt((this.msdx[i] + this.optimiser.Epsilon)
						/ (this.msg[i] + this.optimiser.Epsilon))
						* grad;

				this.msdx[i] *= this.optimiser.Rho;
				this.msdx[i] += (1 - this.optimiser.Rho) * dx * dx;

				this.functionParameters.param.data[i] -= dx;
			}
		}

		@Override
		public Object deepCopy() {
			return new AdaDeltaParameter(null);
		}
	}

}
