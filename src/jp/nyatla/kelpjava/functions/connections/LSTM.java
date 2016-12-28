package jp.nyatla.kelpjava.functions.connections;

import java.util.ArrayList;

import jp.nyatla.kelpjava.common.IDuplicatable;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.Function;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;

/**
 * [Serializable]
 */
public class LSTM extends Function {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2707911009160412463L;
	final public Linear upward0;
	final public Linear upward1;
	public Linear upward2;
	public Linear upward3;

	public Linear lateral0;
	public Linear lateral1;
	public Linear lateral2;
	public Linear lateral3;

	private DoubleList[] aParam;
	private DoubleList[] iParam;
	private DoubleList[] fParam;
	private DoubleList[] oParam;
	private DoubleList[] cParam;

	private double[][] hParam;

	private NdArray[] gxPrev0;
	private NdArray[] gxPrev1;
	private NdArray[] gxPrev2;
	private NdArray[] gxPrev3;
	private double[][] gcPrev;

	private LSTM(LSTM i_src) {
		super(i_src);
		this.upward0 = (Linear) i_src.upward0.deepCopy();
		this.upward1 = (Linear) i_src.upward1.deepCopy();
		this.upward2 = (Linear) i_src.upward2.deepCopy();
		this.upward3 = (Linear) i_src.upward3.deepCopy();

		this.lateral0 = (Linear) i_src.lateral0.deepCopy();
		this.lateral1 = (Linear) i_src.lateral1.deepCopy();
		this.lateral2 = (Linear) i_src.lateral2.deepCopy();
		this.lateral3 = (Linear) i_src.lateral3.deepCopy();

		this.aParam = DoubleList.cloneArray(i_src.aParam);
		this.iParam = DoubleList.cloneArray(i_src.iParam);
		this.fParam = DoubleList.cloneArray(i_src.fParam);
		this.oParam = DoubleList.cloneArray(i_src.oParam);
		this.cParam = DoubleList.cloneArray(i_src.cParam);

		this.hParam = i_src.hParam.clone();
		this.gxPrev0 = NdArray.deepCopy(i_src.gxPrev0);
		this.gxPrev1 = NdArray.deepCopy(i_src.gxPrev1);
		this.gxPrev2 = NdArray.deepCopy(i_src.gxPrev2);
		this.gxPrev3 = NdArray.deepCopy(i_src.gxPrev3);

		this.gcPrev = i_src.gcPrev.clone();

	}

	public LSTM(int i_inSize, int i_outSize, String i_name) {
		this(i_inSize, i_outSize, null, null, null, "LSTM");
	}

	public LSTM(int i_inSize, int i_outSize, NdArray i_initialUpwardW,
			NdArray i_initialUpwardb, NdArray i_initialLateralW, String i_name) {
		super(i_name, i_inSize, i_outSize);
		this.parameters = new FunctionParameter[12];

		this.upward0 = new Linear(i_inSize, i_outSize, false, i_initialUpwardW,
				i_initialUpwardb, "upward0");
		this.upward1 = new Linear(i_inSize, i_outSize, false, i_initialUpwardW,
				i_initialUpwardb, "upward1");
		this.upward2 = new Linear(i_inSize, i_outSize, false, i_initialUpwardW,
				i_initialUpwardb, "upward2");
		this.upward3 = new Linear(i_inSize, i_outSize, false, i_initialUpwardW,
				i_initialUpwardb, "upward3");
		this.parameters[0] = this.upward0.parameters[0];
		this.parameters[1] = this.upward0.parameters[1];
		this.parameters[2] = this.upward1.parameters[0];
		this.parameters[3] = this.upward1.parameters[1];
		this.parameters[4] = this.upward2.parameters[0];
		this.parameters[5] = this.upward2.parameters[1];
		this.parameters[6] = this.upward3.parameters[0];
		this.parameters[7] = this.upward3.parameters[1];

		// lateralはBiasは無し
		this.lateral0 = new Linear(i_outSize, i_outSize, true,
				i_initialLateralW, null, "lateral0");
		this.lateral1 = new Linear(i_outSize, i_outSize, true,
				i_initialLateralW, null, "lateral1");
		this.lateral2 = new Linear(i_outSize, i_outSize, true,
				i_initialLateralW, null, "lateral2");
		this.lateral3 = new Linear(i_outSize, i_outSize, true,
				i_initialLateralW, null, "lateral3");
		this.parameters[8] = this.lateral0.parameters[0];
		this.parameters[9] = this.lateral1.parameters[0];
		this.parameters[10] = this.lateral2.parameters[0];
		this.parameters[11] = this.lateral3.parameters[0];
		return;
	}

	@Override
	protected NdArray[] forwardSingle(NdArray[] x) {
		NdArray[] result = new NdArray[x.length];

		NdArray[] upwards0 = this.upward0.forward(x);
		NdArray[] upwards1 = this.upward1.forward(x);
		NdArray[] upwards2 = this.upward2.forward(x);
		NdArray[] upwards3 = this.upward3.forward(x);

		if (this.hParam == null) {
			// 値がなければ初期化
			this.InitBatch(x.length);
		} else {
			NdArray[] prevInput = new NdArray[this.hParam.length];
			for (int i = 0; i < prevInput.length; i++) {
				prevInput[i] = NdArray.fromArray(this.hParam[i]);
			}

			NdArray[] laterals0 = this.lateral0.forward(prevInput);
			NdArray[] laterals1 = this.lateral1.forward(prevInput);
			NdArray[] laterals2 = this.lateral2.forward(prevInput);
			NdArray[] laterals3 = this.lateral3.forward(prevInput);

			for (int i = 0; i < laterals0.length; i++) {
				for (int j = 0; j < laterals0[i].length(); j++) {
					upwards0[i].data[j] += laterals0[i].data[j];
					upwards1[i].data[j] += laterals1[i].data[j];
					upwards2[i].data[j] += laterals2[i].data[j];
					upwards3[i].data[j] += laterals3[i].data[j];
				}
			}
		}

		for (int i = 0; i < x.length; i++) {
			if (this.cParam[i].size() == 0) {
				this.cParam[i].add(new double[this.outputCount]);
			}

			// 再配置
			double[][] r = this.extractGates(upwards0[i].data,
					upwards1[i].data, upwards2[i].data, upwards3[i].data);

			double[] la = new double[this.outputCount];
			double[] li = new double[this.outputCount];
			double[] lf = new double[this.outputCount];
			double[] lo = new double[this.outputCount];
			double[] cPrev = this.cParam[i].get(this.cParam[i].size() - 1);
			double[] cResult = new double[cPrev.length];

			for (int j = 0; j < this.hParam[i].length; j++) {
				la[j] = Math.tanh(r[0][j]);
				li[j] = Sigmoid(r[1][j]);
				lf[j] = Sigmoid(r[2][j]);
				lo[j] = Sigmoid(r[3][j]);

				cResult[j] = la[j] * li[j] + lf[j] * cPrev[j];
				this.hParam[i][j] = lo[j] * Math.tanh(cResult[j]);
			}

			// Backward用
			this.cParam[i].add(cResult);
			this.aParam[i].add(la);
			this.iParam[i].add(li);
			this.fParam[i].add(lf);
			this.oParam[i].add(lo);

			result[i] = NdArray.fromArray(this.hParam[i]);
		}

		return result;
	}

	@Override
	protected NdArray[] backwardSingle(NdArray[] gh) {
		NdArray[] result = new NdArray[gh.length];

		if (this.gxPrev0 == null) {
			// 値がなければ初期化
			this.gxPrev0 = new NdArray[gh.length];
			this.gxPrev1 = new NdArray[gh.length];
			this.gxPrev2 = new NdArray[gh.length];
			this.gxPrev3 = new NdArray[gh.length];
		} else {
			NdArray[] ghPre0 = this.lateral0.backward(this.gxPrev0);
			NdArray[] ghPre1 = this.lateral1.backward(this.gxPrev1);
			NdArray[] ghPre2 = this.lateral2.backward(this.gxPrev2);
			NdArray[] ghPre3 = this.lateral3.backward(this.gxPrev3);

			for (int j = 0; j < ghPre0.length; j++) {
				for (int k = 0; k < ghPre0[j].length(); k++) {
					gh[j].data[k] += ghPre0[j].data[k];
					gh[j].data[k] += ghPre1[j].data[k];
					gh[j].data[k] += ghPre2[j].data[k];
					gh[j].data[k] += ghPre3[j].data[k];
				}
			}
		}

		{
			for (int i = 0; i < gh.length; i++) {
				this.CalcgxPrev(gh[i].data, i);
			}

			NdArray[] gArray0 = this.upward0.backward(this.gxPrev0);
			NdArray[] gArray1 = this.upward1.backward(this.gxPrev1);
			NdArray[] gArray2 = this.upward2.backward(this.gxPrev2);
			NdArray[] gArray3 = this.upward3.backward(this.gxPrev3);

			for (int i = 0; i < gh.length; i++) {
				double[] gx = new double[this.inputCount];

				for (int j = 0; j < gx.length; j++) {
					gx[j] += gArray0[i].data[j];
					gx[j] += gArray1[i].data[j];
					gx[j] += gArray2[i].data[j];
					gx[j] += gArray3[i].data[j];
				}

				result[i] = new NdArray(gx);
			}
		}

		return result;
	}

	public void CalcgxPrev(double[] gh, int i) {
		double[] ga = new double[this.inputCount];
		double[] gi = new double[this.inputCount];
		double[] gf = new double[this.inputCount];
		double[] go = new double[this.inputCount];

		double[] lcParam = this.cParam[i].get(this.cParam[i].size() - 1);
		this.cParam[i].remove(this.cParam[i].size() - 1);

		double[] laParam = this.aParam[i].get(this.aParam[i].size() - 1);
		this.aParam[i].remove(this.aParam[i].size() - 1);

		double[] liParam = this.iParam[i].get(this.iParam[i].size() - 1);
		this.iParam[i].remove(this.iParam[i].size() - 1);

		double[] lfParam = this.fParam[i].get(this.fParam[i].size() - 1);
		this.fParam[i].remove(this.fParam[i].size() - 1);

		double[] loParam = this.oParam[i].get(this.oParam[i].size() - 1);
		this.oParam[i].remove(this.oParam[i].size() - 1);

		double[] cPrev = this.cParam[i].get(this.cParam[i].size() - 1);

		for (int j = 0; j < this.inputCount; j++) {
			double co = Math.tanh(lcParam[j]);

			this.gcPrev[i][j] += gh[j] * loParam[j] * GradTanh(co);
			ga[j] = this.gcPrev[i][j] * liParam[j] * GradTanh(laParam[j]);
			gi[j] = this.gcPrev[i][j] * laParam[j] * GradSigmoid(liParam[j]);
			gf[j] = this.gcPrev[i][j] * cPrev[j] * GradSigmoid(lfParam[j]);
			go[j] = gh[j] * co * GradSigmoid(loParam[j]);

			this.gcPrev[i][j] *= lfParam[j];
		}

		NdArray[] r = this.RestoreGates(ga, gi, gf, go);

		this.gxPrev0[i] = r[0];
		this.gxPrev1[i] = r[1];
		this.gxPrev2[i] = r[2];
		this.gxPrev3[i] = r[3];
		return;
	}

	@Override
	public void resetState() {
		this.hParam = null;
		this.gxPrev0 = null;
		this.gxPrev1 = null;
		this.gxPrev2 = null;
		this.gxPrev3 = null;
	}

	private static class DoubleList extends ArrayList<double[]> implements
			IDuplicatable {
		private static final long serialVersionUID = 4249178667165812610L;

		@Override
		public Object deepCopy() {
			DoubleList r = new DoubleList();
			for (int i = 0; i < this.size(); i++) {
				r.add(this.get(i).clone());
			}
			return r;
		}

		private static DoubleList[] cloneArray(DoubleList[] i_src) {
			DoubleList[] r = new DoubleList[i_src.length];
			for (int i = 0; i < i_src.length; i++) {
				r[i] = (DoubleList) i_src[i].deepCopy();
			}
			return r;
		}
	}

	// バッチ実行時にバッティングするメンバをバッチ数分用意
	private void InitBatch(int batchCount) {
		this.aParam = new DoubleList[batchCount];
		this.iParam = new DoubleList[batchCount];
		this.fParam = new DoubleList[batchCount];
		this.oParam = new DoubleList[batchCount];
		this.cParam = new DoubleList[batchCount];
		this.hParam = new double[batchCount][];
		this.gcPrev = new double[batchCount][this.inputCount];

		for (int i = 0; i < batchCount; i++) {
			this.aParam[i] = new DoubleList();
			this.iParam[i] = new DoubleList();
			this.fParam[i] = new DoubleList();
			this.oParam[i] = new DoubleList();
			this.cParam[i] = new DoubleList();
			this.hParam[i] = new double[this.outputCount];
		}
	}

	private static double Sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	private static double GradSigmoid(double x) {
		return x * (1 - x);
	}

	private static double GradTanh(double x) {
		return 1 - x * x;
	}

	// Forward用
	double[][] extractGates(double[]... x) {
		double[][] r = new double[4][this.outputCount];

		for (int i = 0; i < this.outputCount; i++) {
			int index = i * 4;

			r[0][i] = x[index / this.outputCount][index % this.outputCount];
			r[1][i] = x[++index / this.outputCount][index % this.outputCount];
			r[2][i] = x[++index / this.outputCount][index % this.outputCount];
			r[3][i] = x[++index / this.outputCount][index % this.outputCount];
		}

		return r;
	}

	// Backward用
	NdArray[] RestoreGates(double[]... x) {
		NdArray[] result = { NdArray.zeros(this.outputCount),
				NdArray.zeros(this.outputCount),
				NdArray.zeros(this.outputCount),
				NdArray.zeros(this.outputCount) };

		for (int i = 0; i < this.outputCount * 4; i++) {
			// 暗黙的に切り捨て
			result[i / this.outputCount].data[i % this.outputCount] = x[i % 4][i / 4];
		}

		return result;
	}

	@Override
	public Object deepCopy() {
		return new LSTM(this);
	}
}
