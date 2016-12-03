package jp.nyatla.kelpjava;

import java.io.Serializable;

import jp.nyatla.kelpjava.common.IDuplicatable;
import jp.nyatla.kelpjava.common.Mother;
import jp.nyatla.kelpjava.common.NdArray;

/**
 * FunctionStackに積み上げるFunctionの基底クラス [Serializable]
 */
public abstract class Function implements IDuplicatable,Serializable
{
	private static final long serialVersionUID = 1622805076763179719L;
	final public String name;
	final protected int outputCount;
	final protected int inputCount;
	public OptimizeParameter[] parameters;// = new ArrayList<OptimizeParameter>();
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	protected Function(Function i_src)
	{
		this.name=i_src.name;
		this.parameters=new OptimizeParameter[i_src.parameters.length];
		for(int i=0;i<this.parameters.length;i++){
			this.parameters[i]=((OptimizeParameter)i_src.parameters[i].deepCopy());
		}
		this.outputCount=i_src.outputCount;
		this.inputCount=i_src.inputCount;
	}
	
	protected Function(String i_name)
	{
		this(i_name, 0, 0);
	}
	/**
	 * コンストラクタ
	 * @param i_name
	 * @param i_inputCount
	 * @param i_oututCount
	 */
	protected Function(String i_name, int i_inputCount, int i_oututCount)
	{
		this.name = i_name;
		this.inputCount = i_inputCount;
		this.outputCount = i_oututCount;
		this.parameters=null;
	}


	/**
	 * 外部公開用
	 * @param i_x
	 * @return
	 */
	public NdArray[] forward(NdArray[] i_x) {
		return this.forwardSingle(i_x);
	}

	/**
	 * 
	 * @param i_gy
	 * @return
	 */
	public NdArray[] backward(NdArray[] i_gy)
	{
		// バッチは内部で割引を行うためgy.Lengthでの加算の必要がない
		for (int i = 0; i < this.parameters.length; i++) {
			this.parameters[i].trainCount++;
		}
		return this.backwardSingle(i_gy);
	}

	/**
	 * 通常であれば非バッチ呼び出しを仮想とするが、バッチ専用関数がスタンダードで非バッチ関数がイレギュラーであるため
	 * @param x
	 * @return
	 */
	protected abstract NdArray[] forwardSingle(NdArray[] x);

	protected abstract NdArray[] backwardSingle(NdArray[] gy);

	/**
	 * 外部公開用非バッチ関数
	 * @param i_x
	 * @return
	 */
	public NdArray forward(NdArray i_x) {
		return this.forwardSingle(i_x);
	}

	public NdArray backward(NdArray i_gy) {
		for (int i = 0; i < this.parameters.length; i++) {
			this.parameters[i].trainCount++;
		}
		return this.backwardSingle(i_gy);
	}

	/**
	 * 任意で個別に非バッチ関数が書けるように用意
	 * @param x
	 * @return
	 */
	protected NdArray forwardSingle(NdArray i_x) {
		NdArray[] na = { i_x };
		return this.forwardSingle(na)[0];
	}
	/**
	 * 
	 * @param gy
	 * @return
	 */
	protected NdArray backwardSingle(NdArray i_gy) {
		NdArray[] na = { i_gy };
		return this.backwardSingle(na)[0];
	}

	/**
	 * 評価関数
	 * @param input
	 * @return
	 */
	public NdArray[] predict(NdArray[] i_input) {
		return this.forwardSingle(i_input);
	}

	public NdArray predict(NdArray i_input) {
		return this.forwardSingle(i_input);
	}

	/**
	 * ある処理実行後に特定のデータを初期値に戻す処理
	 */
	public void resetState() {
		return;
	}

	/**
	 * 名前を返します。
	 * @return
	 */
	@Override
	public String toString() {
		return this.name;
	}

	protected void initWeight(NdArray i_array) {
		this.initWeight(i_array, 1.0);
	}

	/**
	 * 初期値が入力されなかった場合、この関数で初期化を行う
	 * @param array
	 * @param masterScale
	 */
	protected void initWeight(NdArray i_array, double i_masterScale) {
		double localScale = 1 / Math.sqrt(2);
		int fanIn = this.getFans(i_array.shape);
		double s = localScale * Math.sqrt(2.0 / fanIn);

		for (int i = 0; i < i_array.length(); i++) {
			i_array.data[i] = this.normal(s) * i_masterScale;
		}
	}

	private double normal(double i_scale) {
		Mother.Sigma = i_scale;
		return Mother.RandomNormal();
	}

	private int getFans(int[] i_shape) {
		int result = 1;

		for (int i = 1; i < i_shape.length; i++) {
			result *= i_shape[i];
		}

		return result;
	}
}
