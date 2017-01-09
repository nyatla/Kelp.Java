package jp.nyatla.kelpjava.functions.connections;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;
import jp.nyatla.kelpjava.functions.common.NeedPreviousInputFunction;

public class Convolution2D_O1 extends NeedPreviousInputFunction {
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
	protected Convolution2D_O1(Convolution2D_O1 i_src)
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
	public Convolution2D_O1(int i_inputChannels, int i_outputChannels, int i_kSize,int i_pad)
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
	public Convolution2D_O1(int i_inputChannels, int i_outputChannels, int i_kSize,int i_pad, String i_name) {
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
	public Convolution2D_O1(int i_inputChannels, int i_outputChannels,int i_kSize, int i_stride, int i_pad, boolean i_noBias,NdArray i_initialW, NdArray i_initialb, String i_name) {
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
	protected NdArray needPreviousForward(NdArray i_input)
	{
		int outputSize = (int) Math.floor((i_input.shape[2] - this._kSize + this._pad * 2.0)/ this._stride) + 1;

		double[] result = new double[this.outputCount * outputSize * outputSize];
		int resultIndex = 0;
		int d0=i_input.shape[0];//ch
		int d1=i_input.shape[1];//y
		int d2=i_input.shape[2];//x
		int ks=this._kSize;
		int stride=this._stride;
		int pad=this._pad;
		double[] w_data=this.W.data;
		double[] input_data= i_input.data;

//		double p=0;
		for (int och = 0; och < this.outputCount; och++) {
			// Wインデックス用
			int outChOffset = och * this.inputCount * ks*ks;
			for (int oy = 0; oy < outputSize; oy++) {
				//yのインデクス
				int spy=oy * stride - pad;
				int my=ks+spy;if(my>d1){my=d1;}
				int ky=spy<0?0:spy;
				for (int ox = 0; ox < outputSize; ox++) {
					double wret=0;
					//xのインデクス
					int spx=ox * stride - pad;
					int mx=ks+spx;if(mx>d2){mx=d2;}
					int kx=((spx<0)?0:spx);
					for (int ich = 0; ich < d0; ich++) {
						// Wインデックス用
						int wIndexOffset = ich * ks*ks+outChOffset;
						// inputインデックス用
						int inputOffset = ich * d1 * d2;
						for (int iy=ky; iy < my; iy++) {
							int inputIndex = inputOffset + iy * d2;
							int wIndex = wIndexOffset+ (iy-spy) * ks - spx ;
							for (int ix=kx; ix < mx; ix++) {
								wret += input_data[inputIndex+ix] * w_data[wIndex+ ix];
							}
						}
					}
//					p+=wret;
					result[resultIndex] = wret+this.b.data[och];
					resultIndex++;
				}
			}
		}
//		if(55.6459842585469==p){
//			System.out.println();
//		}
		return new NdArray(result, new int[] { this.outputCount, outputSize,outputSize }, false);
	}

	@Override
	protected NdArray needPreviousBackward(NdArray i_gy, NdArray i_prevInput) {
		double[] gx = new double[i_prevInput.length()];

		int gyIndex = 0;
		int pi0=i_prevInput.shape[0];
		int pi1=i_prevInput.shape[1];
		int pi2=i_prevInput.shape[2];
		int ks=this._kSize;
//		int stride=this._stride;
//		int pad=this._pad;
		double[] w_data=this.W.data;
		double[] gW_data=this.gW.data;
		double[] prevInput_data=i_prevInput.data;		
		
		for (int och = 0; och < i_gy.shape[0]; och++) {
			// gWインデックス用
			int outChOffset = och * this.inputCount * ks * ks;

			for (int oy = 0; oy < i_gy.shape[1]; oy++) {
				for (int ox = 0; ox < i_gy.shape[2]; ox++) {
					double gyData = i_gy.data[gyIndex++]; // gyIndex = ch * x * y

					int sx=ox * this._stride - this._pad;
					int sy= oy * this._stride - this._pad;
					int ky=sy<0?0:sy;
					int my=(ks+sy);if(my>pi2){my=pi1;};
					int kx=sx<0?0:sx;
					int mx=(ks+sx);if(mx>pi2){mx=pi2;};
					for (int ich = 0; ich < pi0; ich++) {
						// gWインデックス用
						int inChOffset = ich * ks * ks;
						// inputインデックス用
						int inputOffset = ich * pi1 * pi2;

						for (int iy = ky; iy < my; iy++)
						{
							for (int ix = kx; ix < mx; ix++) {
								// WとgWのshapeは等しい
								int wIndex = outChOffset + inChOffset + (iy-sy) * ks + (ix-sx);
								// prevInputとgxのshapeは等しい
								int inputIndex = inputOffset + iy * pi2 + ix;
								gW_data[wIndex] += prevInput_data[inputIndex] * gyData;
								gx[inputIndex] += w_data[wIndex] * gyData;
							}
						}
					}

					this.gb.data[och] += gyData;
				}
			}
		}
//		double s=JavaUtils.sum(gW_data)+JavaUtils.sum(gx)+JavaUtils.sum(this.gb.data);
//		if(s==-45.331699669989185){
//			System.out.println();
//		}
		

		return new NdArray(gx, i_prevInput.shape.clone(), false);
	}

	@Override
	public Object deepCopy() {
		return new Convolution2D_O1(this);
	}
}
