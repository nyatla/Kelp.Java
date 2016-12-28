package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.NeedPreviousDataFunction;

public class ELU extends NeedPreviousDataFunction {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7452129809367265092L;
	private double _alpha;

	public ELU(ELU i_src) {
		super(i_src);
	}
	
	public ELU() {
		this(1.0, "ELU");
	}

	public ELU(double alpha, String name) {
		super(name);
		this._alpha = alpha;
	}



	@Override
	protected NdArray needPreviousForward(NdArray x) {
		double[] y = new double[x.length()];

		for (int i = 0; i < x.length(); i++) {
			y[i] = x.data[i] >= 0 ? x.data[i] : this._alpha
					* (Math.exp(x.data[i]) - 1);
		}

		return new NdArray(y, x.shape.clone(), false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevInput,
			NdArray prevOutput) {
		double[] gx = new double[gy.length()];

		for (int i = 0; i < gx.length; i++) {
			gx[i] = prevOutput.data[i] >= 0 ? gy.data[i] : gy.data[i]
					* this._alpha * Math.exp(prevInput.data[i]);
		}

		return new NdArray(gx, gy.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new ELU(this);
	}
}
