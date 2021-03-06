package jp.nyatla.kelpjava.optimizers.common;

import java.io.Serializable;

import jp.nyatla.kelpjava.OptimizerParameter;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;

/**
 * Optimizerの素となるクラスでパラメータを持つ
 */
public abstract class Optimizer implements Serializable
{
	private static final long serialVersionUID = 6130067286765706726L;
	public long updateCount = 1;
	protected OptimizerParameter[] optimizerParameters;
	protected Optimizer()
	{
	}
	protected Optimizer(Optimizer i_src)
	{
		this.optimizerParameters=new OptimizerParameter[i_src.optimizerParameters.length];
		for(int i=0;i<this.optimizerParameters.length;i++){
			this.optimizerParameters[i]=(OptimizerParameter) i_src.optimizerParameters[i].deepCopy();
		}
		this.updateCount=i_src.updateCount;
	}

	
	public abstract void addFunctionParameters(FunctionParameter[] functionParameters);

	public void update() {
		for (OptimizerParameter i : this.optimizerParameters) {
			i.updateFunctionParameters();
		}

		this.updateCount++;
	}

	public abstract Optimizer deepCopy();
}
