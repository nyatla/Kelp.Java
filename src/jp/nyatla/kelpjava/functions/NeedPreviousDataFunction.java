package jp.nyatla.kelpjava.functions;

import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.Function;
import jp.nyatla.kelpjava.common.NdArray;

/**
 * 前回の入出力を自動的に扱うクラステンプレート [Serializable]
 * 
 */
public abstract class NeedPreviousDataFunction extends Function {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6440373924822849939L;
	// 後入れ先出しリスト
	private final List<NdArray[]> _prevInput = new ArrayList<NdArray[]>();
	private final List<NdArray[]> _prevOutput = new ArrayList<NdArray[]>();


	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	protected NeedPreviousDataFunction(NeedPreviousDataFunction i_src)
	{
		super(i_src);
		for(int i=0;i<this._prevInput.size();i++){
			this._prevInput.add(NdArray.deepCopy(i_src._prevInput.get(i)));
		}
		for(int i=0;i<this._prevOutput.size();i++){
			this._prevOutput.add(NdArray.deepCopy(i_src._prevOutput.get(i)));
		}
	}
	
	protected NeedPreviousDataFunction(String i_name) {
		this(i_name, 0, 0);
	}

	protected NeedPreviousDataFunction(String i_name, int i_inputCount,int i_oututCount) {
		super(i_name, i_inputCount, i_oututCount);
	}

	protected abstract NdArray needPreviousForward(NdArray i_x);

	protected abstract NdArray needPreviousBackward(NdArray i_gy,NdArray i_prevInput, NdArray i_prevOutput);

	@Override
	protected NdArray forwardSingle(NdArray i_x) {
		this._prevInput.add(new NdArray[] { i_x });
		NdArray result = this.needPreviousForward(i_x);
		this._prevOutput.add(new NdArray[] { result });

		return result;
	}

	@Override
	protected NdArray[] forwardSingle(NdArray[] x) {
		this._prevInput.add(x);

		NdArray[] prevoutput = new NdArray[x.length];

		for (int i = 0; i < x.length; i++) {
			prevoutput[i] = this.needPreviousForward(x[i]);
		}

		this._prevOutput.add(prevoutput);

		return prevoutput;
	}

	@Override
	protected NdArray backwardSingle(NdArray gy) {
		NdArray prevInput = this._prevInput.get(this._prevInput.size() - 1)[0];
		this._prevInput.remove(this._prevInput.size() - 1);

		NdArray prevOutput = this._prevOutput.get(this._prevOutput.size() - 1)[0];
		this._prevOutput.remove(this._prevOutput.size() - 1);

		return this.needPreviousBackward(gy, prevInput, prevOutput);
	}

	@Override
	protected NdArray[] backwardSingle(NdArray[] i_gy) {
		NdArray[] prevInput = this._prevInput.get(this._prevInput.size() - 1);
		this._prevInput.remove(this._prevInput.size() - 1);

		NdArray[] prevOutput = this._prevOutput
				.get(this._prevOutput.size() - 1);
		this._prevOutput.remove(this._prevOutput.size() - 1);

		NdArray[] result = new NdArray[i_gy.length];

		for (int i = 0; i < i_gy.length; i++) {
			result[i] = this.needPreviousBackward(i_gy[i], prevInput[i],
					prevOutput[i]);
		}

		return result;
	}

	@Override
	public NdArray predict(NdArray i_input) {
		return this.needPreviousForward(i_input);
	}

	@Override
	public NdArray[] predict(NdArray[] i_x) {
		NdArray[] prevoutput = new NdArray[i_x.length];
		for (int i = 0; i < i_x.length; i++) {
			prevoutput[i] = this.needPreviousForward(i_x[i]);
		}
		return prevoutput;
	}
}
