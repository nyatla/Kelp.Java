package jp.nyatla.kelpjava.loss;

import jp.nyatla.kelpjava.common.NdArray;

abstract public class LossFunction
{
	static public class Result
	{
		public double loss;
		public NdArray data;
	}
	static public class Results
	{
		public Results(NdArray[] i_array, double i_loss) {
			this.data=i_array;
			this.loss=i_loss;
		}
		final public double loss;
		final public NdArray[] data;
	}
	
    abstract protected Result evaluate(NdArray i_input, NdArray i_teachSignal,Result o_loss);
    public Result evaluate(NdArray i_input, NdArray i_teachSignal)
    {
    	return this.evaluate(i_input, i_teachSignal,new Result());
    }
    public Results evaluate(NdArray[] i_input, NdArray[] i_teachSignal)
    {
    	NdArray[] array = new NdArray[i_input.length];
    	double sum=0;
    	LossFunction.Result t=new LossFunction.Result();
        for (int i = 0; i < i_input.length; i++)
        {
        	this.evaluate(i_input[i], i_teachSignal[i],t);
        	sum+=t.loss;
        	array[i]=t.data;
        }
        //Average
        return new Results(array,sum/i_input.length);
    }    
}
