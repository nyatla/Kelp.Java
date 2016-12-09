package jp.nyatla.kelpjava;

import java.io.Serializable;

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

	
	public abstract void initilise(OptimizeParameter[] functionParameters);

	public void update() {
		for (OptimizerParameter i : this.optimizerParameters) {
			i.update();
		}

		this.updateCount++;
	}

	public abstract Optimizer deepCopy();
}
