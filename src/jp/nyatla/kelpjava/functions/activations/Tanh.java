package jp.nyatla.kelpjava.functions.activations;

import jp.nyatla.kelpjava.OptimizeParameter;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.NeedPreviousOutputFunction;

    /**
     * [Serializable]
     *
     */
    public class Tanh extends NeedPreviousOutputFunction
    {
		private static final long serialVersionUID = -9168668285732504457L;
		protected Tanh(Tanh i_src)
        {
        	super(i_src);
		}
    	
        public Tanh(String i_name)
        {
        	super(i_name);
    		this.parameters = new OptimizeParameter[0];        	
        }    	
        public Tanh()
        {
        	this("Tanh");
        }

		@Override
        protected NdArray needPreviousForward(NdArray i_x)
        {
            double[] y = new double[i_x.length()];

            for (int i = 0; i < y.length; i++)
            {
                y[i] = Math.tanh(i_x.data[i]);
            }

            return new NdArray(y, i_x.shape.clone(),false);
        }

        @Override
        protected NdArray needPreviousBackward(NdArray i_gy, NdArray i_prevOutput)
        {
            double[] gx = new double[i_gy.length()];

            for (int i = 0; i < gx.length; i++)
            {
                gx[i] = i_gy.data[i] * (1 - i_prevOutput.data[i] * i_prevOutput.data[i]);
            }

            return new NdArray(gx, i_gy.shape.clone(),false);
        }
		@Override
		public Object deepCopy() {
			return new Tanh(this);
		}
    }
