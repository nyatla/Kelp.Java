package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.OptimizeParameter;
import jp.nyatla.kelpjava.Optimizer;

    /**
     * [Serializable]
     *
     */
    final public class MomentumSGD extends Optimizer
    {
        final private double LearningRate;
        final private double momentum;
        final private double[][] v;
        
//        public MomentumSGD(MomentumSGD i_src)
//        {
//        	super(i_src);
//        	this.LearningRate=i_src.LearningRate;
//        	this.momentum=i_src.momentum;
//        	this.v=new double[i_src.v.length][];
//        	for(int i=0;i<this.v.length;i++){
//        		this.v[i]=i_src.v[i].clone();
//        	}
//        }

        public MomentumSGD(OptimizeParameter[] i_parameters)
        {
        	this(i_parameters,0.01,0.9);
        }
        public MomentumSGD(OptimizeParameter[] i_parameters,double i_learningRate, double i_momentum)
        {
        	super(i_parameters);
            this.LearningRate = i_learningRate;
            this.momentum = i_momentum;
            this.v = new double[i_parameters.length][];
            for (int i = 0; i < this.v.length; i++)
            {
                this.v[i] = new double[parameters[i].param.length()];
            }            
        }

        @Override
        protected void doUpdate()
        {
            for (int i = 0; i < this.parameters.length; i++)

            {
                for (int k = 0; k < this.parameters[i].length(); k++)
                {
                    this.v[i][k] *= this.momentum;
                    this.v[i][k] -= this.LearningRate * this.parameters[i].grad.data[k];

                    this.parameters[i].param.data[k] += this.v[i][k];
                }
            }
        }
    }

