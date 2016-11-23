package jp.nyatla.kelpjava;

import jp.nyatla.kelpjava.common.NdArray;

public class OptimizeParameter
{
	final public String name;
	final public NdArray param;
	final public NdArray grad;

	/** Updateを行わずに実行されたBackwardの回数をカウントし、バッチ更新時に使用する*/
	public int trainCount;

	public int Length() {
		return this.param.length();
	}

	public OptimizeParameter(NdArray param, NdArray grad, String name) {
		this.param = param;
		this.grad = grad;
		this.name = name;
	}

	/**
	 * 傾き・バッチカウントを初期化する。
	 */
	public void ClearGrad() {
		// 0埋め
		this.grad.fill(0);

		// バッチカウントもリセット
		this.trainCount = 0;
	}

	@Override
	public String toString() {
		return this.name;
	}
}
