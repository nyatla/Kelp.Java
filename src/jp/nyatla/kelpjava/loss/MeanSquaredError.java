package jp.nyatla.kelpjava.loss;

import jp.nyatla.kelpjava.common.NdArray;

final public class MeanSquaredError extends SingleLossFunction {

	@Override
	protected Result evaluate(NdArray i_input, NdArray i_teachSignal,Result o_loss) {
		double loss = 0.0;

		double[] diff = new double[i_teachSignal.length()];
		double coeff = 2.0 / diff.length;

		for (int i = 0; i < i_input.length(); i++) {
			diff[i] = i_input.data[i] - i_teachSignal.data[i];
			loss += Math.pow(diff[i], 2);

			diff[i] *= coeff;
		}

		o_loss.loss = loss / diff.length;
		o_loss.data = new NdArray(diff, i_teachSignal.shape.clone(),false);
		return o_loss;
	}
}
