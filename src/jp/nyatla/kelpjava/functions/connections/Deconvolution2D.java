package jp.nyatla.kelpjava.functions.connections;

import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.functions.common.NeedPreviousInputFunction;

public class Deconvolution2D extends NeedPreviousInputFunction {
	/**
		 * 
		 */
	private static final long serialVersionUID = 5973166043348098375L;
	final public NdArray W;
	final public NdArray b;

	final public NdArray gW;
	final public NdArray gb;

	final private int _kSize;
	final private int _subSample;
	final private int _trim;

	public Deconvolution2D(Deconvolution2D i_src) {
		super(i_src);
		this.W = (NdArray) i_src.W.deepCopy();
		this.b = (NdArray) i_src.b.deepCopy();
		this.gW = (NdArray) i_src.gW.deepCopy();
		this.gb = (NdArray) i_src.gb.deepCopy();
		this._kSize = i_src._kSize;
		this._subSample = i_src._subSample;
		this._trim = i_src._trim;
	}

	public Deconvolution2D(int i_inputChannels, int i_outputChannels,int i_kSize) {
		this(i_inputChannels, i_outputChannels, i_kSize, 1);
	}
	public Deconvolution2D(int i_inputChannels, int i_outputChannels,int i_kSize, int i_subSample) {
		this(i_inputChannels,i_outputChannels,i_kSize,i_subSample,"Deconv2D");
	}
	public Deconvolution2D(int i_inputChannels, int i_outputChannels,int i_kSize, int i_subSample,int i_trim) {
		this(i_inputChannels,i_outputChannels,i_kSize,i_subSample,i_trim,"Deconv2D");
	}

	public Deconvolution2D(int i_inputChannels, int i_outputChannels,int i_kSize, int i_subSample, String i_name) {
		this(i_inputChannels, i_outputChannels, i_kSize, i_subSample,0,i_name);
	}
	public Deconvolution2D(int i_inputChannels, int i_outputChannels,int i_kSize, int i_subSample,int i_trim,String i_name) {
		this(i_inputChannels, i_outputChannels, i_kSize, i_subSample,i_trim, false,initWeight(NdArray.zeros(i_outputChannels, i_inputChannels,i_kSize, i_kSize)), NdArray.zeros(i_outputChannels),i_name);
	}
	public Deconvolution2D(int i_inputChannels, int i_outputChannels,
			int i_kSize, int i_subSample, int i_trim, boolean i_noBias,NdArray i_initialW, NdArray i_initialb)
	{
		this(i_inputChannels,i_outputChannels,i_kSize, i_subSample, i_trim, i_noBias,i_initialW,i_initialb,"Deconv2D");
	}

	public Deconvolution2D(int i_inputChannels, int i_outputChannels,
			int i_kSize, int i_subSample, int i_trim, boolean i_noBias,NdArray i_initialW, NdArray i_initialb, String i_name)
	{
		super(i_name, i_inputChannels, i_outputChannels);

		this._kSize = i_kSize;
		this._subSample = i_subSample;
		this._trim = i_trim;

		this.parameters = new FunctionParameter[i_noBias ? 1 : 2];

		this.W = i_initialW;
		this.gW = NdArray.zerosLike(this.W);

		this.parameters[0] = new FunctionParameter(this.W, this.gW, this.name
				+ " W");

		// noBias=trueでもbiasを用意して更新しない
		this.b = NdArray.zeros(i_outputChannels);
		this.gb = NdArray.zerosLike(this.b);

		if (!i_noBias) {
			this.parameters[1] = new FunctionParameter(this.b, this.gb,
					this.name + " b");
		}
	}

	@Override
	protected NdArray needPreviousForward(NdArray i_input) {
		int outputSize = (i_input.shape[2] - 1) * this._subSample + this._kSize
				- this._trim * 2;

		double[] result = new double[this.outputCount * outputSize * outputSize];

		int outSizeOffset = outputSize * outputSize;

		int inputSizeOffset = i_input.shape[1] * i_input.shape[2];
		int kSizeOffset = this.W.shape[2] * this.W.shape[3];

		for (int och = 0; och < this.W.shape[0]; och++) {
			for (int ich = 0; ich < i_input.shape[0]; ich++) // ich = kich
																// input.Shape[0]
																// =
																// this.W.Shape[1]
			{
				for (int iy = 0; iy < i_input.shape[1]; iy++) {
					for (int ix = 0; ix < i_input.shape[2]; ix++) {
						int inputIndex = ich * inputSizeOffset + iy
								* i_input.shape[2] + ix;

						for (int ky = 0; ky < this.W.shape[2]; ky++) {
							int outIndexY = iy * this._subSample + ky
									- this._trim;

							for (int kx = 0; kx < this.W.shape[3]; kx++) {
								int outIndexX = ix * this._subSample + kx
										- this._trim;

								int outputIndex = och * outSizeOffset
										+ outIndexY * outputSize + outIndexX;

								int kernelIndex = och * this.W.shape[1]
										* kSizeOffset + ich * kSizeOffset + ky
										* this.W.shape[3] + kx;

								if (outIndexY >= 0 && outIndexY < outputSize
										&& outIndexX >= 0
										&& outIndexX < outputSize) {
									result[outputIndex] += i_input.data[inputIndex]
											* this.W.data[kernelIndex];
								}
							}
						}
					}
				}
			}

			for (int oy = 0; oy < outputSize; oy++) {
				for (int ox = 0; ox < outputSize; ox++) {
					int outputIndex = och * outSizeOffset + oy * outputSize
							+ ox;
					result[outputIndex] += this.b.data[och];
				}
			}
		}

		return new NdArray(result, new int[] { this.outputCount, outputSize,
				outputSize }, false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray gy, NdArray prevInput) {
		double[] gx = new double[prevInput.length()];

		for (int och = 0; och < this.gW.shape[0]; och++) {
			// Wインデックス用
			int outChOffset = och * this.gW.shape[1] * this.gW.shape[2]
					* this.gW.shape[3];

			// inputインデックス用
			int inputOffset = och * gy.shape[1] * gy.shape[2];

			for (int ich = 0; ich < this.gW.shape[1]; ich++) {
				// Wインデックス用
				int inChOffset = ich * this.gW.shape[2] * this.gW.shape[3];

				int pinputOffset = ich * prevInput.shape[1]
						* prevInput.shape[2];

				for (int gwy = 0; gwy < this.gW.shape[2]; gwy++) {
					for (int gwx = 0; gwx < this.gW.shape[3]; gwx++) {
						for (int py = 0; py < prevInput.shape[1]; py++) {
							int gyy = py * this._subSample + gwy - this._trim;

							for (int px = 0; px < prevInput.shape[2]; px++) {
								int gyx = px * this._subSample + gwx
										- this._trim;

								int gwIndex = outChOffset + inChOffset + gwy
										* this.gW.shape[3] + gwx;
								int gyIndex = inputOffset + gyy * gy.shape[2]
										+ gyx;

								int pInIndex = pinputOffset + py
										* prevInput.shape[2] + px;

								if (gyy >= 0 && gyy < gy.shape[1] && gyx >= 0
										&& gyx < gy.shape[2]) {
									this.gW.data[gwIndex] += prevInput.data[pInIndex]
											* gy.data[gyIndex];
									gx[pInIndex] += this.W.data[gwIndex]
											* gy.data[gyIndex];
								}
							}
						}
					}
				}
			}

			for (int oy = 0; oy < gy.shape[1]; oy++) {
				for (int ox = 0; ox < gy.shape[2]; ox++) {
					int gyIndex = inputOffset + oy * gy.shape[2] + ox;
					this.gb.data[och] += gy.data[gyIndex];
				}
			}
		}

		return new NdArray(gx, prevInput.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new Deconvolution2D(this);
	}
}
