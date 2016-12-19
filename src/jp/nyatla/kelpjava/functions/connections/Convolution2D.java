package jp.nyatla.kelpjava.functions.connections;

import jp.nyatla.kelpjava.FunctionParameter;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.NeedPreviousInputFunction;

public class Convolution2D extends NeedPreviousInputFunction {
	private static final long serialVersionUID = -4713515794669987999L;
	final public NdArray W;
	final public NdArray b;

	final public NdArray gW;
	final public NdArray gb;

	private final int _kSize;
	private final int _stride;
	private final int _pad;
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	protected Convolution2D(Convolution2D i_src)
	{
		super(i_src);
		this.W=(NdArray) i_src.W.deepCopy();
		this.b=(NdArray)i_src.b.deepCopy();
		this.gW=(NdArray) i_src.gW.deepCopy();
		this.gb=(NdArray)i_src.gb.deepCopy();
		this._kSize=i_src._kSize;
		this._stride=i_src._stride;
		this._pad=i_src._pad;
	}
	/**
	 * 
	 * @param i_inputChannels
	 * @param i_outputChannels
	 * @param i_kSize
	 * @param i_pad
	 */
	public Convolution2D(int i_inputChannels, int i_outputChannels, int i_kSize,int i_pad)
	{
		this(i_inputChannels,i_outputChannels,i_kSize,i_pad,"Conv2D");
	}

	/**
	 * 
	 * @param i_inputChannels
	 * @param i_outputChannels
	 * @param i_kSize
	 * @param i_pad
	 * @param i_name
	 */
	public Convolution2D(int i_inputChannels, int i_outputChannels, int i_kSize,int i_pad, String i_name) {
		this(	i_inputChannels,i_outputChannels,i_kSize,
				1,i_pad, false,
				initWeight(NdArray.zeros(i_outputChannels, i_inputChannels,i_kSize, i_kSize)),
				NdArray.zeros(i_outputChannels),
				i_name);
	}


	/**
	 * 
	 * @param i_inputChannels
	 * @param i_outputChannels
	 * @param i_kSize
	 * @param i_stride
	 * @param i_pad
	 * @param i_noBias
	 * @param i_initialW
	 * @param i_initialb
	 * @param i_name
	 */
	public Convolution2D(int i_inputChannels, int i_outputChannels,int i_kSize, int i_stride, int i_pad, boolean i_noBias,NdArray i_initialW, NdArray i_initialb, String i_name) {
		super(i_name, i_inputChannels, i_outputChannels);
		this._kSize = i_kSize;
		this._stride = i_stride;
		this._pad = i_pad;

		this.W = i_initialW;
		this.gW = NdArray.zerosLike(this.W);

		this.parameters = new FunctionParameter[i_noBias ? 1 : 2];
		this.parameters[0] = new FunctionParameter(this.W, this.gW, this.name + " W");

		// noBias=trueでもbiasを用意して更新しない
		this.b = i_initialb;
		this.gb = NdArray.zerosLike(this.b);

		if (!i_noBias) {
			this.parameters[1] = new FunctionParameter(this.b, this.gb,this.name + " b");
		}
		return;
	}

	@Override
	protected NdArray needPreviousForward(NdArray i_input) {
		int outputSize = (int) Math.floor((i_input.shape[2] - this._kSize + this._pad * 2.0)/ this._stride) + 1;

		double[] result = new double[this.outputCount * outputSize * outputSize];
		int resultIndex = 0;

		for (int och = 0; och < this.outputCount; och++) {
			// Wインデックス用
			int outChOffset = och * this.inputCount * this._kSize * this._kSize;

			for (int oy = 0; oy < outputSize; oy++) {
				for (int ox = 0; ox < outputSize; ox++) {
					for (int ich = 0; ich < i_input.shape[0]; ich++) {
						// Wインデックス用
						int inChOffset = ich * this._kSize * this._kSize;

						// inputインデックス用
						int inputOffset = ich * i_input.shape[1] * i_input.shape[2];
						for (int ky = 0; ky < this._kSize; ky++) {
							int iy = oy * this._stride + ky - this._pad;

							if (iy >= 0 && iy < i_input.shape[1]) {
								for (int kx = 0; kx < this._kSize; kx++) {
									int ix = ox * this._stride + kx - this._pad;

									if (ix >= 0 && ix < i_input.shape[2]) {
										int wIndex = outChOffset + inChOffset+ ky * this._kSize + kx;
										int inputIndex = inputOffset + iy * i_input.shape[2] + ix;
										result[resultIndex] += i_input.data[inputIndex] * this.W.data[wIndex];
									}
								}
							}
						}
					}
					result[resultIndex] += this.b.data[och];
					resultIndex++;
				}
			}
		}
		return new NdArray(result, new int[] { this.outputCount, outputSize,
				outputSize }, false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray i_gy, NdArray i_prevInput) {
		double[] gx = new double[i_prevInput.length()];

		int gyIndex = 0;

		for (int och = 0; och < i_gy.shape[0]; och++) {
			// gWインデックス用
			int outChOffset = och * this.inputCount * this._kSize * this._kSize;

			for (int oy = 0; oy < i_gy.shape[1]; oy++) {
				for (int ox = 0; ox < i_gy.shape[2]; ox++) {
					double gyData = i_gy.data[gyIndex++]; // gyIndex = ch * x * y

					for (int ich = 0; ich < i_prevInput.shape[0]; ich++) {
						// gWインデックス用
						int inChOffset = ich * this._kSize * this._kSize;

						// inputインデックス用
						int inputOffset = ich * i_prevInput.shape[1] * i_prevInput.shape[2];

						for (int ky = 0; ky < this._kSize; ky++) {
							int iy = oy * this._stride + ky - this._pad;

							if (iy >= 0 && iy < i_prevInput.shape[1]) {
								for (int kx = 0; kx < this._kSize; kx++) {
									int ix = ox * this._stride + kx - this._pad;

									if (ix >= 0 && ix < i_prevInput.shape[2]) {
										// WとgWのshapeは等しい
										int wIndex = outChOffset + inChOffset + ky * this._kSize + kx;

										// prevInputとgxのshapeは等しい
										int inputIndex = inputOffset + iy * i_prevInput.shape[2] + ix;

										this.gW.data[wIndex] += i_prevInput.data[inputIndex] * gyData;

										gx[inputIndex] += this.W.data[wIndex] * gyData;
									}
								}
							}
						}
					}

					this.gb.data[och] += gyData;
				}
			}
		}

		return new NdArray(gx, i_prevInput.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new Convolution2D(this);
	}
}
