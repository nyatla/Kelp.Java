package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.OptimizeParameter;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.OptimizerParameter;

/**
 * [Serializable]
 * 
 */
public class SGD extends Optimizer
{
	private static final long serialVersionUID = 5440602882523432289L;
	final public double learningRate;
	/**
	 * コピーコンストラクタ
	 * @param i_src
	 */	
	private SGD(SGD i_src) {
		super(i_src);		
		this.learningRate = i_src.learningRate;
	}
	public SGD(double i_learningRate) {
		this.learningRate = i_learningRate;
	}

	public SGD() {
		this(0.1);
	}
	@Override
    public void initilise(OptimizeParameter[] i_functionParameters)
    {
        this.optimizerParameters = new OptimizerParameter[i_functionParameters.length];

        for (int i = 0; i < this.optimizerParameters.length; i++)
        {
            this.optimizerParameters[i] = new SGDParameter(i_functionParameters[i], this);
        }
    }
	@Override
	public Optimizer deepCopy() {
		return new SGD(this);
	}

	/**
	 * [Serializable]
	 * @author nyatla
	 *
	 */
	private class SGDParameter extends OptimizerParameter
	{
		private static final long serialVersionUID = 368508461777294695L;
		private SGD optimiser;
		public SGDParameter(SGDParameter i_src) {
			super(i_src);
			this.optimiser = (SGD) i_src.optimiser.deepCopy();
		}	    
	    public SGDParameter(OptimizeParameter i_functionParameter, SGD i_optimiser)
	    {
	    	super(i_functionParameter);
	        this.optimiser = i_optimiser;
	    }
	
	    @Override
	    public void update()
	    {
	        for (int i = 0; i < this.functionParameters.length(); i++)
	        {
	            this.functionParameters.param.data[i] -= this.optimiser.learningRate * this.functionParameters.grad.data[i];
	        }
	    }

		@Override
		public Object deepCopy() {
			return new SGDParameter(this);
		}
	}

}


