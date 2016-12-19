package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.FunctionParameter;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.OptimizerParameter;

/**
 * 与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
 * 
 */
public class GradientClipping extends Optimizer {
	private static final long serialVersionUID = -7390615912880588714L;
	public double Threshold;

	public GradientClipping(GradientClipping i_src) {
		super(i_src);
		this.Threshold = i_src.Threshold;
	}

	public GradientClipping(double threshold) {
		this.Threshold = threshold;
	}

	@Override
	public void initilise(FunctionParameter[] functionParameters) {
		this.optimizerParameters = new OptimizerParameter[functionParameters.length];

		for (int i = 0; i < this.optimizerParameters.length; i++) {
			this.optimizerParameters[i] = new GradientClippingParameter(
					functionParameters[i], this);
		}
	}

	@Override
	public Optimizer deepCopy() {
		return new GradientClipping(this);
	}

	private class GradientClippingParameter extends OptimizerParameter {
		private static final long serialVersionUID = -955421061998475596L;
		private final GradientClipping optimiser;

		public GradientClippingParameter(FunctionParameter i_functionParameter,
				GradientClipping i_optimiser) {
			super(i_functionParameter);
			this.optimiser = i_optimiser;
		}

		protected GradientClippingParameter(GradientClippingParameter i_src) {
			super(i_src);
			this.optimiser = (GradientClipping) i_src.optimiser.deepCopy();
		}

		@Override
		public void updateFunctionParameters() {
			// _sum_sqnorm
			double s = 0.0;

			for (int i = 0; i < this.functionParameters.length(); i++) {
				s += Math.pow(this.functionParameters.grad.data[i], 2);
			}

			double norm = Math.sqrt(s);
			double rate = this.optimiser.Threshold / norm;

			if (rate < 1) {
				for (int i = 0; i < this.functionParameters.length(); i++) {
					this.functionParameters.grad.data[i] *= rate;
				}
			}
		}

		@Override
		public Object deepCopy() {
			return new GradientClippingParameter(this);
		}
	}

}
