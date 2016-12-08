package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.IOptimizer;
import jp.nyatla.kelpjava.OptimizeParameter;

/**
 * [Serializable]
 * 
 */
public class SGD implements IOptimizer
{
	final public double LearningRate;

	public SGD(double learningRate) {
		this.LearningRate = learningRate;
	}

	public SGD() {
		this(0.1);
	}

	@Override
	public IOptimizer initialise(OptimizeParameter i_parameter) {
		return this;
	}

	@Override
	public void update(OptimizeParameter i_parameter) {
		for (int j = 0; j < i_parameter.length(); j++) {
			i_parameter.param.data[j] -= this.LearningRate* i_parameter.grad.data[j];
		}
	}
}
