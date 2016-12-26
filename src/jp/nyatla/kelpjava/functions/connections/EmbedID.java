package jp.nyatla.kelpjava.functions.connections;

import jp.nyatla.kelpjava.FunctionParameter;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.NeedPreviousInputFunction;

public class EmbedID extends NeedPreviousInputFunction {
	private static final long serialVersionUID = 2281978012463953715L;
	final public NdArray W;
	final public NdArray gW;

	private EmbedID(EmbedID i_src) {
		super(i_src);
		this.W = (NdArray) i_src.W.deepCopy();
		this.gW = (NdArray) i_src.gW.deepCopy();
	}

	public EmbedID(int i_inputCount, int outputCount) {
		this(i_inputCount, outputCount, null, null);
	}
	public EmbedID(int i_inputCount, int outputCount,String i_name) {
		this(i_inputCount, outputCount, null, i_name);
	}

	public EmbedID(int inputCount, int outputCount, NdArray initialW,String name)
	{
		super(name, inputCount, outputCount);

		if (initialW == null) {
			this.W = NdArray.zeros(inputCount, outputCount);
			initWeight(this.W);
		} else {
			this.W = initialW;
		}
		this.gW = NdArray.zerosLike(this.W);

		this.parameters = new FunctionParameter[] { new FunctionParameter(
				this.W, this.gW, this.name + " W") };
	}

	@Override
	protected NdArray needPreviousForward(NdArray x) {
		double[] result = new double[x.length() * this.outputCount];

		for (int i = 0; i < x.length(); i++) {
			for (int j = 0; j < this.outputCount; j++) {
				result[i * this.outputCount + j] = this.W.data[(int) x.data[i]
						* this.outputCount + j];
			}
		}

		return new NdArray(result, new int[] { x.length(), this.outputCount },
				false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevInput) {
		for (int i = 0; i < prevInput.length(); i++) {
			for (int j = 0; j < this.outputCount; j++) {
				this.gW.data[(int) prevInput.data[i] * this.outputCount + j] += gy.data[i
						+ j];
			}
		}

		return null;
	}

	@Override
	public Object deepCopy() {
		return new EmbedID(this);
	}
}
