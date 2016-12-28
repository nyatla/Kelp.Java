package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.NeedPreviousOutputFunction;

public class LeakyReLU extends NeedPreviousOutputFunction {
	private static final long serialVersionUID = -453265116046979469L;
	final private double _slope;

	public LeakyReLU(LeakyReLU i_src) {
		super(i_src);
		this._slope = i_src._slope;
	}

	public LeakyReLU() {
		this(0.2, "LeakyReLU");
	}

	public LeakyReLU(double i_slope, String i_name) {
		super(i_name);
		this._slope = i_slope;
	}

	@Override
	protected NdArray needPreviousForward(NdArray x) {
		double[] y = new double[x.length()];

		for (int i = 0; i < x.length(); i++) {
			y[i] = x.data[i] < 0 ? x.data[i] *= this._slope : x.data[i];
		}

		return new NdArray(y, x.shape.clone(), false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevOutput) {
		double[] gx = new double[gy.length()];

		for (int i = 0; i < gx.length; i++) {
			gx[i] = prevOutput.data[i] > 0 ? gy.data[i] : prevOutput.data[i]
					* this._slope;
		}

		return new NdArray(gx, gy.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new LeakyReLU(this);
	}
}
