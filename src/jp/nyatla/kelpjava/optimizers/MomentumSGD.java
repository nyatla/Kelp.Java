package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.IOptimizer;
import jp.nyatla.kelpjava.OptimizeParameter;

/**
 * [Serializable]
 * 
 */
final public class MomentumSGD implements IOptimizer
{
	final private double LearningRate;
	final private double momentum;
	final private double[] v;

	// public MomentumSGD(MomentumSGD i_src)
	// {
	// super(i_src);
	// this.LearningRate=i_src.LearningRate;
	// this.momentum=i_src.momentum;
	// this.v=new double[i_src.v.length][];
	// for(int i=0;i<this.v.length;i++){
	// this.v[i]=i_src.v[i].clone();
	// }
	// }

	public MomentumSGD() {
		this(0.01, 0.9, 0);
	}

	public MomentumSGD(double i_learningRate, double i_momentum,
			int i_parameterLength) {
		this.LearningRate = i_learningRate;
		this.momentum = i_momentum;
		this.v = new double[i_parameterLength];
	}

	@Override
	public IOptimizer initialise(OptimizeParameter parameter) {
		return new MomentumSGD(this.LearningRate, this.momentum,
				parameter.length());
	}

	@Override
	public void update(OptimizeParameter i_parameter) {
		for (int i = 0; i < i_parameter.length(); i++) {

			this.v[i] *= this.momentum;
			this.v[i] -= this.LearningRate * i_parameter.grad.data[i];

			i_parameter.param.data[i] += this.v[i];
		}
	}
}
