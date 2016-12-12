package jp.nyatla.kelpjava;

import java.io.Serializable;

import jp.nyatla.kelpjava.common.IDuplicatable;
import jp.nyatla.kelpjava.common.NdArray;

final public class FunctionParameter implements IDuplicatable,Serializable
{
	private static final long serialVersionUID = 7870851322738478448L;
	final public String name;
	final public NdArray param;
	final public NdArray grad;
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */
	protected FunctionParameter(FunctionParameter i_src)
	{
		this.name=i_src.name;
		this.grad=(NdArray) i_src.grad.deepCopy();
		this.param=(NdArray) i_src.param.deepCopy();
		this.trainCount=i_src.trainCount;
	}
	/** Updateを行わずに実行されたBackwardの回数をカウントし、バッチ更新時に使用する*/
	public int trainCount;
	

	public int length()
	{
		return this.param.length();
	}

	public FunctionParameter(NdArray param, NdArray grad, String name) {
		this.param = param;
		this.grad = grad;
		this.name = name;
	}
    //傾きの補正
    public void reduce()
    {
        for (int j = 0; j < this.grad.length(); j++)
        {
            this.grad.data[j] /= this.trainCount;
        }

        //カウンタをリセット
        this.trainCount = 0;
    }

	/**
	 * 傾き・バッチカウントを初期化する。
	 */
	public void clearGrad() {
		// 0埋め
		this.grad.fill(0);
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
