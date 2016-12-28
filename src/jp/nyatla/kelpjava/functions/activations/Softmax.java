package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.functions.common.NeedPreviousOutputFunction;

public class Softmax extends NeedPreviousOutputFunction {
	/**
		 * 
		 */
	private static final long serialVersionUID = 167528661534372458L;

	protected Softmax(Softmax i_src) {
		super(i_src);
	}

	public Softmax() {
		this("Softmax");
	}

	public Softmax(String i_name) {
		super(i_name);
		this.parameters=new FunctionParameter[0];
	}

	@Override
	protected NdArray needPreviousForward(NdArray x) {
		double[] y = new double[x.length()];

		double maxval = JavaUtils.max(x.data);
		double sumval = 0.0;

		for (int i = 0; i < y.length; i++) {
			y[i] = Math.exp(x.data[i] - maxval);
			sumval += y[i];
		}

		for (int i = 0; i < y.length; i++) {
			y[i] /= sumval;
		}

		return new NdArray(y, x.shape.clone(), false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevOutput) {
		double[] gx = new double[gy.length()];
		double sumdx = 0.0;

		for (int i = 0; i < gx.length; i++) {
			gx[i] = prevOutput.data[i] * gy.data[i];
			sumdx += gx[i];
		}

		for (int i = 0; i < gx.length; i++) {
			gx[i] -= prevOutput.data[i] * sumdx;
		}

		return new NdArray(gx, gy.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new Softmax(this);
	}



}
