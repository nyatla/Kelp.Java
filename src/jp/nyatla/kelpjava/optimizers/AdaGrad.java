package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.FunctionParameter;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.OptimizerParameter;

    public class AdaGrad extends Optimizer
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = 5190699996067552348L;
		final public double LearningRate;
        final public double Epsilon;
        protected AdaGrad(AdaGrad i_src)
        {
        	super(i_src);
        	this.Epsilon=i_src.Epsilon;
        	this.LearningRate=i_src.LearningRate;
        }

        public AdaGrad()
        {
        	this(0.01,1e-8);
        }
        public AdaGrad(double i_learningRate, double i_epsilon)
        {
            this.LearningRate = i_learningRate;
            this.Epsilon = i_epsilon;
        }

        @Override
        public void initilise(FunctionParameter[] functionParameters)
        {
            this.optimizerParameters = new OptimizerParameter[functionParameters.length];

            for (int i = 0; i < this.optimizerParameters.length; i++)
            {
                this.optimizerParameters[i] = new AdaGradParameter(functionParameters[i], this);
            }
        }
    	@Override
    	public Optimizer deepCopy() {
    		// TODO Auto-generated method stub
    		return null;
    	}    

    /**
     * [Serializable]
     *
     */
    private class AdaGradParameter extends OptimizerParameter
    {
		private static final long serialVersionUID = 1302322013532525608L;
		private final AdaGrad optimiser;
        private final double[] h;
        private AdaGradParameter(AdaGradParameter i_src)
        {
        	super(i_src);
        	this.optimiser=(AdaGrad) i_src.optimiser.deepCopy();
        	this.h=i_src.h.clone();
        }

        public AdaGradParameter(FunctionParameter i_functionParameter, AdaGrad i_optimiser)
        {
        	super(i_functionParameter);
            this.h = new double[i_functionParameter.length()];
            this.optimiser = i_optimiser;
        }

        @Override
        public void updateFunctionParameters()
        {
            for (int i = 0; i < this.functionParameters.length(); i++)
            {
                double grad = this.functionParameters.grad.data[i];

                this.h[i] += grad * grad;

                this.functionParameters.param.data[i] -= this.optimiser.LearningRate * grad / (Math.sqrt(this.h[i]) + this.optimiser.Epsilon);
            }
        }

		@Override
		public Object deepCopy() {
			return new AdaGradParameter(this);
		}
    }




    }
