package jp.nyatla.kelpjava.optimizers;

import jp.nyatla.kelpjava.Optimizer;

    /**
     * [Serializable]
     *
     */
    final public class MomentumSGD extends Optimizer
    {
        final private double LearningRate;
        final private double momentum;
        private double[][] v;
        
        public MomentumSGD(MomentumSGD i_src)
        {
        	super(i_src);
        	this.LearningRate=i_src.LearningRate;
        	this.momentum=i_src.momentum;
        	this.v=new double[i_src.v.length][];
        	for(int i=0;i<this.v.length;i++){
        		this.v[i]=i_src.v[i].clone();
        	}
        }

        public MomentumSGD()
        {
        	this(0.01,0.9);
        }
        public MomentumSGD(double i_learningRate, double i_momentum)
        {
        	super();
            this.LearningRate = i_learningRate;
            this.momentum = i_momentum;
        }

        @Override
        protected void doUpdate()
        {
            for (int i = 0; i < this.parameters.size(); i++)

            {
                for (int k = 0; k < this.parameters.get(i).length(); k++)
                {
                    this.v[i][k] *= this.momentum;
                    this.v[i][k] -= this.LearningRate * this.parameters.get(i).grad.data[k];

                    this.parameters.get(i).param.data[k] += this.v[i][k];
                }
            }
        }
        @Override
        protected void initialize()
        {
            this.v = new double[this.parameters.size()][];

            for (int i = 0; i < this.v.length; i++)
            {
                this.v[i] = new double[this.parameters.get(i).param.length()];
            }
        }
		@Override
		public Object deepCopy()
		{
			return new MomentumSGD(this);
		}
    }

