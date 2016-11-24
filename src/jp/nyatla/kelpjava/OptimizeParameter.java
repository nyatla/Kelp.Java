package jp.nyatla.kelpjava;

import jp.nyatla.kelpjava.common.IDuplicatable;
import jp.nyatla.kelpjava.common.NdArray;

final public class OptimizeParameter implements IDuplicatable
{
	final public String name;
	final public NdArray param;
	final public NdArray grad;

	/** Updateを行わずに実行されたBackwardの回数をカウントし、バッチ更新時に使用する*/
	public int trainCount;

	public int length()
	{
		return this.param.length();
	}

	public OptimizeParameter(NdArray param, NdArray grad, String name) {
		this.param = param;
		this.grad = grad;
		this.name = name;
	}
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	protected OptimizeParameter(OptimizeParameter i_src)
	{
		this.name=i_src.name;
		this.grad=(NdArray) i_src.grad.deepCopy();
		this.param=(NdArray) i_src.param.deepCopy();
		this.trainCount=i_src.trainCount;
	}

	/**
	 * 傾き・バッチカウントを初期化する。
	 */
	public void clearGrad() {
		// 0埋め
		this.grad.fill(0);

		// バッチカウントもリセット
		this.trainCount = 0;
	}

	@Override
	public String toString() {
		return this.name;
	}

	@Override
	public Object deepCopy() {
		return this.deepCopy();
	}

}
