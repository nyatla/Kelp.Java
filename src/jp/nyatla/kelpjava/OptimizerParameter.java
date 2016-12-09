package jp.nyatla.kelpjava;

import java.io.Serializable;

import jp.nyatla.kelpjava.common.IDuplicatable;


/**
 * このクラスはFunctionParameterと1:1で作成される
 * 
 */
public abstract class OptimizerParameter implements IDuplicatable, Serializable {
	private static final long serialVersionUID = 3444889240336483434L;
	protected OptimizeParameter functionParameters;
	protected OptimizerParameter(OptimizerParameter i_src)
	{
		this.functionParameters = (OptimizeParameter) i_src.functionParameters.deepCopy();
	}

	protected OptimizerParameter(OptimizeParameter i_functionParameter) {
		this.functionParameters = i_functionParameter;
	}

	public abstract void update();
}
