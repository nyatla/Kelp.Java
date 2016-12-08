package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.OptimizeParameter;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.NeedPreviousOutputFunction;

/**
 * ランプ関数
 * [Serializable]
 */
public class ReLU extends NeedPreviousOutputFunction {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4242585829660176536L;

	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	public ReLU(ReLU i_src) {
		super(i_src);
	}

	public ReLU() {
		this("ReLU");
	}

	public ReLU(String i_name) {
		super(i_name);
		this.parameters = new OptimizeParameter[0];
	}

	@Override
	protected NdArray needPreviousForward(NdArray i_x) {
		double[] y = new double[i_x.length()];

		for (int i = 0; i < i_x.length(); i++) {
			y[i] = Math.max(0, i_x.data[i]);
		}

		return new NdArray(y, i_x.shape.clone(), false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray i_gy, NdArray i_prevOutput) {
		double[] gx = new double[i_gy.length()];

		for (int i = 0; i < i_gy.length(); i++) {
			gx[i] = i_prevOutput.data[i] > 0 ? i_gy.data[i] : 0;
		}

		return new NdArray(gx, i_gy.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new ReLU(this);
	}
}
