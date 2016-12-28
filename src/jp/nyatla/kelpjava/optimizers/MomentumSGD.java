package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.OptimizerParameter;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.optimizers.common.Optimizer;

/**
 * [Serializable]
 * 
 */
final public class MomentumSGD extends Optimizer {
	private static final long serialVersionUID = -470212701546866706L;
	final private double LearningRate;
	final private double momentum;
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	private MomentumSGD(MomentumSGD i_src) {
		super(i_src);
		this.LearningRate=i_src.LearningRate;
		this.momentum=i_src.momentum;
	}

	public MomentumSGD() {
		this(0.01, 0.9);
	}

	public MomentumSGD(double i_learningRate, double i_momentum) {
		this.LearningRate = i_learningRate;
		this.momentum = i_momentum;
	}

	public void addFunctionParameters(FunctionParameter[] i_functionParameters) {
		this.optimizerParameters = new OptimizerParameter[i_functionParameters.length];

		for (int i = 0; i < this.optimizerParameters.length; i++) {
			this.optimizerParameters[i] = new MomentumSGDParameter(
					i_functionParameters[i], this);
		}
	}
	@Override
	public Optimizer deepCopy() {
		return new MomentumSGD(this);
	}
	/**
	 * [Serializable]
	 * 
	 */
	private class MomentumSGDParameter extends OptimizerParameter {
		private static final long serialVersionUID = 2286314412322217159L;
		private MomentumSGD optimiser;
		private double[] v;

		public MomentumSGDParameter(MomentumSGDParameter i_src) {
			super(i_src);
			this.v = i_src.v.clone();
			this.optimiser = (MomentumSGD) i_src.optimiser.deepCopy();
		}

		public MomentumSGDParameter(FunctionParameter i_functionParameter,
				MomentumSGD i_optimiser) {
			super(i_functionParameter);
			this.v = new double[i_functionParameter.length()];
			this.optimiser = i_optimiser;
		}

		@Override
		public void updateFunctionParameters() {
			for (int i = 0; i < this.functionParameters.length(); i++) {
				this.v[i] *= this.optimiser.momentum;
				this.v[i] -= this.optimiser.LearningRate
						* this.functionParameters.grad.data[i];

				this.functionParameters.param.data[i] += this.v[i];
			}
		}

		@Override
		public Object deepCopy() {
			return new MomentumSGDParameter(this);
		}
	}

}
