package jp.nyatla.kelpjava.functions;

import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.Function;
import jp.nyatla.kelpjava.common.NdArray;

/**
 * 前回の入出力を自動的に扱うクラステンプレート [Serializable]
 */
public abstract class NeedPreviousInputFunction extends Function
{
	private static final long serialVersionUID = -5125646377622756151L;
	// 後入れ先出しリスト
	final private List<NdArray[]> _prevInput = new ArrayList<NdArray[]>();

	protected abstract NdArray needPreviousForward(NdArray x);

	protected abstract NdArray needPreviousBackward(NdArray gy,NdArray prevInput);
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	protected NeedPreviousInputFunction(NeedPreviousInputFunction i_src)
	{
		super(i_src);
		for(int i=0;i<this._prevInput.size();i++){
			this._prevInput.add(NdArray.deepCopy(i_src._prevInput.get(i)));
		}
	}
	
	protected NeedPreviousInputFunction(String i_name)
	{
		this(i_name, 0, 0);
	}

	protected NeedPreviousInputFunction(String i_name, int i_inputCount,int i_oututCount)
	{
		super(i_name, i_inputCount, i_oututCount);
	}

	
	@Override
	protected NdArray forwardSingle(NdArray i_x) {
		this._prevInput.add(new NdArray[] {i_x});

		return this.needPreviousForward(i_x);
	}

	@Override
	protected NdArray[] forwardSingle(NdArray[] i_x)
	{
		this._prevInput.add(i_x);

		NdArray[] prevoutput = new NdArray[i_x.length];

		for (int i = 0; i < i_x.length; i++) {
			prevoutput[i] = this.needPreviousForward(i_x[i]);
		}
		return prevoutput;
	}

	protected NdArray backwardSingle(NdArray i_gy) {
		NdArray prevInput = this._prevInput.get(this._prevInput.size() - 1)[0];
		this._prevInput.remove(this._prevInput.size() - 1);

		return this.needPreviousBackward(i_gy, prevInput);
	}

	@Override
	protected NdArray[] backwardSingle(NdArray[] i_gy) {
		NdArray[] prevInput = this._prevInput.get(this._prevInput.size() - 1);
		this._prevInput.remove(this._prevInput.size() - 1);

		NdArray[] result = new NdArray[i_gy.length];

		for (int i = 0; i < i_gy.length; i++) {
			result[i] = this.needPreviousBackward(i_gy[i], prevInput[i]);
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
