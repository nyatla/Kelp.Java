package jp.nyatla.kelpjava;

import java.io.Serializable;

import jp.nyatla.kelpjava.common.IDuplicatable;


/**
 * このクラスはFunctionParameterと1:1で作成される
 * 
 */
public abstract class OptimizerParameter implements IDuplicatable, Serializable {
	private static final long serialVersionUID = 3444889240336483434L;
	protected FunctionParameter functionParameters;
	protected OptimizerParameter(OptimizerParameter i_src)
	{
		this.functionParameters = (FunctionParameter) i_src.functionParameters.deepCopy();
	}

	protected OptimizerParameter(FunctionParameter i_functionParameter) {
		this.functionParameters = i_functionParameter;
	}

	public abstract void updateFunctionParameters();
}
