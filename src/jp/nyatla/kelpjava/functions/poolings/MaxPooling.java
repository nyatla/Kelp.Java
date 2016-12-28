package jp.nyatla.kelpjava.functions.poolings;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.functions.common.NeedPreviousDataFunction;

/**
 * [Serializable]
 * 
 */
public class MaxPooling extends NeedPreviousDataFunction
{
	private static final long serialVersionUID = -4905695215302617823L;
	final private int _kSize;
	final private int _stride;
	final private int _pad;
	protected MaxPooling(MaxPooling i_src)
	{
		super(i_src);
		this._kSize=i_src._kSize;
		this._stride=i_src._stride;
		this._pad=i_src._pad;
	}

	public MaxPooling(int i_ksize) {
		this(i_ksize, 1, 0, "MaxPooling");
	}
	public MaxPooling(int i_ksize, int i_stride, String i_name) {
		this(i_ksize,i_stride,0,i_name);
	}
	public MaxPooling(int i_ksize, int i_stride, int i_pad, String i_name) {
		super(i_name);
		this.parameters = new FunctionParameter[0];		
		this._kSize = i_ksize;
		this._stride = i_stride;
		this._pad = i_pad;
	}

	@Override
	protected NdArray needPreviousForward(NdArray input) {
		int outputSize = (int) Math
				.floor((input.shape[2] - this._kSize + this._pad * 2.0)
						/ this._stride) + 1;
		double[] result = JavaUtils.fill(new double[input.shape[0] * outputSize * outputSize],Double.MIN_VALUE);

		int resultIndex = 0;

		for (int i = 0; i < input.shape[0]; i++) {
			int inputIndexOffset = i * input.shape[1] * input.shape[2];

			for (int y = 0; y < outputSize; y++) {
				for (int x = 0; x < outputSize; x++) {
					for (int dy = 0; dy < this._kSize; dy++) {
						int inputIndexY = y * this._stride + dy - this._pad;

						if (inputIndexY >= 0 && inputIndexY < input.shape[1]) {
							for (int dx = 0; dx < this._kSize; dx++) {
								int inputIndexX = x * this._stride + dx
										- this._pad;

								if (inputIndexX >= 0
										&& inputIndexX < input.shape[2]) {
									int inputIndex = inputIndexOffset
											+ inputIndexY * input.shape[2]
											+ inputIndexX;
									result[resultIndex] = Math.max(
											result[resultIndex],
											input.data[inputIndex]);
								}
							}
						}
					}

					resultIndex++;
				}
			}
		}

		return new NdArray(result, new int[] { input.shape[0], outputSize,
				outputSize }, false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevInput,
			NdArray prevOutput) {
		double[] result = new double[prevInput.length()];

		int index = 0;

		for (int i = 0; i < prevInput.shape[0]; i++) {
			int prevInputIndexOffset = i * prevInput.shape[1]
					* prevInput.shape[2];
			for (int y = 0; y < prevOutput.shape[1]; y++) {
				for (int x = 0; x < prevOutput.shape[2]; x++) {
					// 前回の入力値と出力値を比較して、同じ値のものを見つける
					this.SetResult(prevInputIndexOffset, y, x, gy.data[index],
							prevInput, prevOutput.data[index], result);
					index++;
				}
			}
		}

		return new NdArray(result, prevInput.shape.clone(), false);
	}

	// 同じ値を複数持つ場合、左上優先にして処理を打ち切る
	// 他のライブラリの実装では乱数を取って同じ値の中からどれかを選ぶ物が多い
	void SetResult(int i_prevInputIndexOffset, int i_y, int i_x, double i_data,
			NdArray i_prevInput, double i_prevOutputData, double[] i_result) {
		for (int dy = 0; dy < this._kSize; dy++) {
			int outputIndexY = i_y * this._stride + dy - this._pad;

			if (outputIndexY >= 0 && outputIndexY < i_prevInput.shape[1]) {
				for (int dx = 0; dx < this._kSize; dx++) {
					int outputIndexX = i_x * this._stride + dx - this._pad;

					if (outputIndexX >= 0
							&& outputIndexX < i_prevInput.shape[2]) {
						int prevInputIndex = i_prevInputIndexOffset
								+ outputIndexY * i_prevInput.shape[2]
								+ outputIndexX;

						if (i_prevInput.data[prevInputIndex] == i_prevOutputData) {
							i_result[prevInputIndex] = i_data;
							return;
						}
					}
				}
			}
		}
	}
	@Override
	public Object deepCopy() {
		return new MaxPooling(this);
	}
}
