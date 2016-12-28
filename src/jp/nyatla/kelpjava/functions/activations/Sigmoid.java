package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.functions.common.NeedPreviousOutputFunction;

/**
 * シグモイド関数
 * [Serializable]
 *
 */
final public class Sigmoid extends NeedPreviousOutputFunction {
	private static final long serialVersionUID = -3943737017157421431L;

	/**
	 * コピーコンストラクタ
	 * 
	 * @param i_src
	 */
	public Sigmoid(Sigmoid i_src) {
		super(i_src);
	}

	public Sigmoid() {
		this("Sigmoid");
	}

	public Sigmoid(String i_name) {
		super(i_name);
		this.parameters = new FunctionParameter[0];
	}

	@Override
	protected NdArray needPreviousForward(NdArray i_x) {
		double[] y = new double[i_x.length()];

		for (int i = 0; i < i_x.length(); i++) {
			y[i] = 1 / (1 + Math.exp(-i_x.data[i]));
		}

		return new NdArray(y, i_x.shape.clone(), false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevOutput) {
		double[] gx = new double[gy.length()];

		for (int i = 0; i < gy.length(); i++) {
			gx[i] = gy.data[i] * prevOutput.data[i] * (1 - prevOutput.data[i]);
		}

		return new NdArray(gx, gy.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new Sigmoid(this);
	}
}
