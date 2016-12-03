package jp.nyatla.kelpjava.loss;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;

public class SoftmaxCrossEntropy extends SingleLossFunction {

	protected Result evaluate(NdArray i_input, NdArray i_teachSignal,Result o_loss) {
		int maxIndex = (int) Math.max(JavaUtils.max(i_teachSignal.data), 0.0);

		double[] logY = SoftmaxLog(i_input.data);
		double loss = -logY[maxIndex];

		double[] gx = new double[logY.length];

		for (int i = 0; i < logY.length; i++) {
			gx[i] = Math.exp(logY[i]);
		}

		gx[maxIndex] -= 1;
		o_loss.loss = loss;
		o_loss.data = new NdArray(gx, i_input.shape.clone(),false);
		return o_loss;
	}

	private static double[] SoftmaxLog(double[] x) {
		double[] result = new double[x.length];

		double[] y = new double[x.length];
		double m = JavaUtils.max(x);

		for (int i = 0; i < x.length; i++) {
			y[i] = Math.exp(x[i] - m);
		}

		m += Math.log(JavaUtils.sum(y));

		for (int i = 0; i < x.length; i++) {
			result[i] = x[i] - m;
		}

		return result;
	}
}
