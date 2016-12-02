package jp.nyatla.kelpjava.loss;

import jp.nyatla.kelpjava.common.NdArray;


/**
 * Kelp.netのpacial実装をインスタンス付きのクラスにしています。
 *
 */
public abstract class LossFunctions
{
	final static public class Results
	{
		public Results(NdArray[] i_array, double i_loss) {
			this.data=i_array;
			this.loss=i_loss;
		}
		final public double loss;
		final public NdArray[] data;
	}
	final private SingleLossFunction _fn;
	protected LossFunctions(SingleLossFunction i_function)
	{
		this._fn=i_function;
	}
	
    public Results evaluate(NdArray[] i_input, NdArray[] i_teachSignal)
    {
    	NdArray[] array = new NdArray[i_input.length];
    	double sum=0;
    	SingleLossFunction.Result t=new SingleLossFunction.Result();
        for (int i = 0; i < i_input.length; i++)
        {
        	this._fn.evaluate(i_input[i], i_teachSignal[i],t);
        	sum+=t.loss;
        	array[i]=t.data;
        }
        //Average
        return new Results(array,sum/i_input.length);
    }
}
