package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.NeedPreviousOutputFunction;

public class Softplus extends NeedPreviousOutputFunction {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5972824634581149478L;
	final private double _beta;
	final private double _betaInv;

	public Softplus(Softplus i_src) {
		super(i_src);
		this._beta = i_src._beta;
		this._betaInv = i_src._betaInv;
	}

	public Softplus() {
		this(1.0, "Softplus");
	}

	public Softplus(double beta, String i_name) {
		super(i_name);
		this._beta = beta;
		this._betaInv = 1.0 / beta;
	}

	@Override
	protected NdArray needPreviousForward(NdArray x) {
		double[] y = new double[x.length()];

		for (int i = 0; i < y.length; i++) {
			y[i] = x.data[i] * this._beta;
		}

		double maxval = Math.max(JavaUtils.max(y), 0);

		for (int i = 0; i < y.length; i++) {
			y[i] = (maxval + Math.log(1.0 + Math.exp(-Math.abs(x.data[i]
					* this._beta))))
					* this._betaInv;
		}

		return new NdArray(y, x.shape.clone(), false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevOutput) {
		double[] gx = new double[gy.length()];

		for (int i = 0; i < gx.length; i++) {
			gx[i] = (1 - 1 / (1 + Math.exp(this._beta * prevOutput.data[i])))
					* gy.data[i];
		}

		return new NdArray(gx, gy.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new Softplus(this);
	}

}
