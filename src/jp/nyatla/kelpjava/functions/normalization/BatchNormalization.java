package jp.nyatla.kelpjava.functions.normalization;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.Function;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;

/**
 * Chainerより移植　finetuningは未実装
 */
public class BatchNormalization extends Function {
	/**
		 * 
		 */
	private static final long serialVersionUID = -327313252405712460L;
	private boolean IsTrain;
	final private NdArray Gamma;
	final private NdArray gGamma;
	final private NdArray Beta;
	final private NdArray gBeta;
	final private double Decay;
	final private double Eps;
	final private NdArray AvgMean;
	final private NdArray AvgVar;

	private NdArray gMean;
	private NdArray gVariance;
	private double[] Std;
	private double[][] Xhat;

	private double[] Mean;
	private double[] Variance;
	final private int ChannelSize;

	public BatchNormalization(BatchNormalization i_src) {
		super(i_src);
		this.IsTrain = i_src.IsTrain;
		this.Gamma = (NdArray) i_src.Gamma.deepCopy();
		this.gGamma = (NdArray) i_src.gGamma.deepCopy();
		this.Beta = (NdArray) i_src.Beta.deepCopy();
		this.gBeta = (NdArray) i_src.gBeta.deepCopy();
		this.Decay = i_src.Decay;
		this.Eps = i_src.Eps;
		this.AvgMean = (NdArray) i_src.AvgMean.deepCopy();
		this.AvgVar = (NdArray) i_src.AvgVar.deepCopy();
		this.gMean = (NdArray) i_src.gMean.deepCopy();
		this.gVariance = (NdArray) i_src.gVariance.deepCopy();
		this.Std = i_src.Std.clone();
		this.Xhat = JavaUtils.cloneArray(i_src.Xhat);
		this.Mean = i_src.Mean.clone();
		this.Variance = i_src.Variance.clone();
		this.ChannelSize = i_src.ChannelSize;
	}
	public BatchNormalization(int i_channelSize,String i_name) {
		this(i_channelSize, 0.9, 1e-5, true, i_name);
	}

	public BatchNormalization(int i_channelSize) {
		this(i_channelSize,"BatchNorm");
	}

	public BatchNormalization(int i_channelSize, double i_decay, double i_eps,
			boolean i_isTrain, String i_name) {
		super(i_name);
		this.ChannelSize = i_channelSize;
		this.Decay = i_decay;
		this.Eps = i_eps;
		this.IsTrain = i_isTrain;

		this.Gamma = NdArray.ones(i_channelSize);
		this.Beta = NdArray.zeros(i_channelSize);

		this.gGamma = NdArray.zerosLike(this.Gamma);
		this.gBeta = NdArray.zerosLike(this.Beta);

		this.parameters = new FunctionParameter[this.IsTrain ? 2 : 4];

		// 学習対象のParameterを登録
		this.parameters[0] = new FunctionParameter(this.Gamma, this.gGamma,
				this.name + " Gamma");
		this.parameters[1] = new FunctionParameter(this.Beta, this.gBeta,
				this.name + " Beta");

		this.AvgMean = NdArray.zeros(i_channelSize);
		this.AvgVar = NdArray.zeros(i_channelSize);

		if (!this.IsTrain) {
			this.gMean = NdArray.zeros(i_channelSize);
			this.gVariance = NdArray.zeros(i_channelSize);

			this.parameters[2] = new FunctionParameter(this.AvgMean,
					this.gMean, this.name + " Mean");
			this.parameters[3] = new FunctionParameter(this.AvgVar,
					this.gVariance, this.name + " Variance");
		}
	}

	@Override
	protected NdArray[] forwardSingle(NdArray[] x) {
		NdArray[] y = new NdArray[x.length];

		// 計算用パラメータの取得
		if (this.IsTrain) {
			// メンバのMeanとVarianceを設定する
			this.CalcVariance(x);
		} else {
			this.Mean = this.AvgMean.data;
			this.Variance = this.AvgVar.data;
		}

		this.Std = new double[this.Variance.length];
		for (int i = 0; i < this.Variance.length; i++) {
			this.Std[i] = Math.sqrt(this.Variance[i]);
		}

		// 結果を計算
		this.Xhat = new double[x.length][this.ChannelSize];

		for (int i = 0; i < x.length; i++) {
			y[i] = NdArray.zerosLike(x[i]);

			for (int j = 0; j < this.ChannelSize; j++) {
				this.Xhat[i][j] = (x[i].data[j] - this.Mean[j]) / this.Std[j];
				y[i].data[j] = this.Gamma.data[j] * this.Xhat[i][j]
						+ this.Beta.data[j];
			}
		}

		// パラメータを更新
		if (this.IsTrain) {
			int m = x.length;
			double adjust = m / Math.max(m - 1.0, 1.0); // unbiased estimation

			for (int i = 0; i < this.AvgMean.length(); i++) {
				this.AvgMean.data[i] *= this.Decay;
				this.Mean[i] *= 1 - this.Decay; // reuse buffer as a temporary
				this.AvgMean.data[i] += this.Mean[i];

				this.AvgVar.data[i] *= this.Decay;
				this.Variance[i] *= (1 - this.Decay) * adjust; // reuse buffer
																// as a
																// temporary
				this.AvgVar.data[i] += this.Variance[i];
			}
		}

		return y;
	}

	public void CalcVariance(NdArray... values) {
		this.Variance = new double[this.ChannelSize];
		for (int i = 0; i < this.Variance.length; i++) {
			this.Variance[i] = 0;
		}

		this.Mean = new double[this.ChannelSize];
		for (int i = 0; i < this.Mean.length; i++) {
			for (NdArray value : values) {
				this.Mean[i] += value.data[i];
			}

			this.Mean[i] /= values.length;
		}

		for (int i = 0; i < this.Mean.length; i++) {
			for (NdArray value : values) {
				this.Variance[i] += Math.pow(value.data[i] - this.Mean[i], 2);
			}

			this.Variance[i] /= values.length;
		}

		for (int i = 0; i < this.Variance.length; i++) {
			this.Variance[i] += this.Eps;
		}
	}

	@Override
	protected NdArray[] backwardSingle(NdArray[] gy) {
		NdArray[] gx = new NdArray[gy.length];
		for (int i = 0; i < gy.length; i++) {
			gx[i] = NdArray.zeros(this.ChannelSize);
		}

		this.gBeta.fill(0);
		this.gGamma.fill(0);

		for (int i = 0; i < this.ChannelSize; i++) {
			for (int j = 0; j < gy.length; j++) {
				this.gBeta.data[i] += gy[j].data[i];
				this.gGamma.data[i] += gy[j].data[i] * this.Xhat[j][i];
			}
		}

		if (!this.IsTrain) {
			// 学習なし
			for (int i = 0; i < this.ChannelSize; i++) {
				double gs = this.Gamma.data[i] / this.Std[i];
				this.gMean.data[i] = -gs * this.gBeta.data[i];
				this.gVariance.data[i] = -0.5 * this.Gamma.data[i]
						/ this.AvgVar.data[i] * this.gGamma.data[i];

				for (int j = 0; j < gy.length; j++) {
					gx[j].data[i] = gs * gy[j].data[i];
				}
			}
		} else {
			int m = gy.length;

			for (int i = 0; i < this.ChannelSize; i++) {
				double gs = this.Gamma.data[i] / this.Std[i];

				for (int j = 0; j < gy.length; j++) {
					double val = (this.Xhat[j][i] * this.gGamma.data[i] + this.gBeta.data[i])
							/ m;

					gx[j].data[i] = gs * (gy[j].data[i] - val);
				}
			}
		}

		return gx;
	}

	@Override
	public NdArray[] predict(NdArray[] input) {
		NdArray[] result;

		if (this.IsTrain) {
			// Predictはトレーニングしない
			this.IsTrain = false;

			result = this.forwardSingle(input);

			// フラグをリセット
			this.IsTrain = true;
		} else {
			result = this.forwardSingle(input);
		}

		return result;
	}

	@Override
	public Object deepCopy() {
		return new BatchNormalization(this);
	}
}
