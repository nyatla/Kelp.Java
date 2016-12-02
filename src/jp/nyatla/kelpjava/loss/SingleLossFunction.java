package jp.nyatla.kelpjava.loss;

import jp.nyatla.kelpjava.common.NdArray;

abstract public class SingleLossFunction
{
	static public class Result{
		public double loss;
		public NdArray data;
	}
    abstract protected Result evaluate(NdArray i_input, NdArray i_teachSignal,Result o_loss);
    public Result evaluate(NdArray i_input, NdArray i_teachSignal)
    {
    	return this.evaluate(i_input, i_teachSignal,new Result());
    }
}
